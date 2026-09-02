"""The entity side of retrieval: a second vector collection holding the entities
extracted from each memory, and the memory ids each one appears in.

It lives here rather than on ``Memory`` because both ``Memory`` and
``AsyncMemory`` need it and this code has no reason to exist twice. Everything
below is blocking, so the async side reaches it through one
``asyncio.to_thread`` per call instead of one per store hit.

NOTE: the operations are all fail-open. An entity is a retrieval boost, never
the record itself, so a store that is down must not break add, update or
delete. Failures are logged and swallowed on purpose.
"""

import concurrent.futures
import logging
import uuid
import weakref

from mem0.memory.utils import _safe_deepcopy_config, _vector_store_list_rows
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
from mem0.utils.factory import VectorStoreFactory
from mem0.utils.scoring import ENTITY_BOOST_WEIGHT

logger = logging.getLogger(__name__)

_SCOPE_KEYS = ("user_id", "agent_id", "run_id")

# An entity match below this contributes no boost at all.
_BOOST_FLOOR = 0.5

# Entity rows are small and a scope holds few of them, so one list() is cheaper
# than paging. NOTE: a scope that ever exceeds this silently loses cleanup.
_LIST_CAP = 10000


def entity_collection_name(provider: str, collection_name: str) -> str:
    separator = "-" if provider == "s3_vectors" else "_"
    return f"{collection_name}{separator}entities"


def _scope_of(filters):
    return {k: v for k, v in filters.items() if k in _SCOPE_KEYS and v}


class EntityStore:
    """Entity records for one ``Memory``/``AsyncMemory``.

    NOTE: reads the owner's embedder and config on each call rather than
    snapshotting them, because both are replaced after construction and a
    snapshot would freeze whichever object happened to exist first.

    NOTE: the reference is weak, and must stay weak. A strong one makes
    ``Memory`` part of a cycle, so it survives until the cycle collector runs
    instead of dying on its last reference. With embedded Qdrant that means the
    RocksDB lock outlives the ``Memory`` that took it, and the next instance
    fails to open the store.
    """

    def __init__(self, owner):
        self._owner = weakref.proxy(owner)
        self._store = None

    @property
    def store(self):
        """The underlying vector store, created on first use."""
        if self._store is None:
            config = self._owner.config.vector_store
            entity_config = _safe_deepcopy_config(config.config)
            collection = entity_collection_name(config.provider, self._owner.collection_name)
            if hasattr(entity_config, "collection_name"):
                entity_config.collection_name = collection
            elif isinstance(entity_config, dict):
                entity_config["collection_name"] = collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if config.provider == "qdrant" and hasattr(self._owner.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self._owner.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self._owner.vector_store.client
            self._store = VectorStoreFactory.create(config.provider, entity_config)
        return self._store

    @property
    def initialized(self):
        """Whether the store has been touched in this process.

        Cleanup paths check this: never having listed entities is not the same
        as having none, and creating the collection just to empty it is waste.
        """
        return self._store is not None

    @staticmethod
    def normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _dedup_threshold(self):
        return self._owner.config.dedup_similarity_threshold

    def existing_by_text(self, filters):
        """Return existing entity rows keyed by normalized payload data."""
        try:
            listed = self.store.list(filters=filters, top_k=_LIST_CAP)
        except Exception as e:
            logger.debug(f"Exact entity lookup failed, falling back to semantic dedup: {e}")
            return {}

        rows_by_text = {}
        for row in _vector_store_list_rows(listed):
            payload = getattr(row, "payload", None) or {}
            text = payload.get("data")
            if not isinstance(text, str):
                continue
            normalized = self.normalize(text)
            if normalized and normalized not in rows_by_text:
                rows_by_text[normalized] = row
        return rows_by_text

    def upsert(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self._owner.embedding_model.embed(entity_text, "add")
            scope = _scope_of(filters)
            exact_match = self.existing_by_text(scope).get(self.normalize(entity_text))

            existing = []
            if exact_match is None:
                existing = self.store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=1,
                    filters=scope,
                )

            threshold = self._dedup_threshold()
            semantic_match = existing[0] if existing and existing[0].score >= threshold else None
            match = exact_match or semantic_match
            if match:
                # Update existing entity's linked_memory_ids
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                self.store.insert(
                    vectors=[entity_embedding],
                    ids=[str(uuid.uuid4())],
                    payloads=[{
                        "data": entity_text,
                        "entity_type": entity_type,
                        "linked_memory_ids": [memory_id],
                        **scope,
                    }],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def link_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id`, scoped to
        `filters`. Single-memory variant of `link_batch`: one
        search-then-update-or-insert per entity. Non-fatal on any failure.
        """
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = self.normalize(entity_text)
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self.upsert(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    def link_batch(self, records, filters):
        """Link every memory in `records` to its entities in one pass.

        `records` are the add pipeline's `(memory_id, text, embedding, payload,
        contradicted)` tuples. The whole point is batching: one entity
        extraction, one embed, one search and one insert for the batch, rather
        than the per-memory round trips `link_memory` does.
        """
        scope = _scope_of(filters)
        try:
            all_entities = extract_entities_batch([r[1] for r in records])

            # Global dedup — one row per distinct entity across all the memories
            global_entities = {}  # normalized_key -> [entity_type, entity_text, {memory_ids}]
            for idx, (memory_id, _text, _embedding, _payload, _contradicted) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = self.normalize(entity_text)
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if not global_entities:
                return

            ordered_keys = list(global_entities.keys())
            entity_texts = [global_entities[k][1] for k in ordered_keys]

            try:
                entity_embeddings = self._owner.embedding_model.embed_batch(entity_texts, "add")
            except Exception:
                # Fallback: embed individually, use None for failures
                entity_embeddings = []
                for t in entity_texts:
                    try:
                        entity_embeddings.append(self._owner.embedding_model.embed(t, "add"))
                    except Exception:
                        entity_embeddings.append(None)

            if len(entity_embeddings) != len(ordered_keys):
                logger.warning(
                    "embed_batch returned %d vectors for %d entity texts — "
                    "padding/truncating to avoid dropping entity links",
                    len(entity_embeddings),
                    len(ordered_keys),
                )
                entity_embeddings = list(entity_embeddings[: len(ordered_keys)])
                entity_embeddings += [None] * (len(ordered_keys) - len(entity_embeddings))

            # Entities whose embedding failed cannot be searched or inserted
            valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
            if not valid:
                return

            valid_indices, valid_keys = zip(*valid)
            valid_vectors = [entity_embeddings[i] for i in valid_indices]
            exact_matches = self.existing_by_text(scope)

            existing_matches = self.store.search_batch(
                queries=[global_entities[k][1] for k in valid_keys],
                vectors_list=valid_vectors,
                top_k=1,
                filters=scope,
            )

            threshold = self._dedup_threshold()
            to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
            for j, key in enumerate(valid_keys):
                entity_type, entity_text, memory_ids = global_entities[key]
                matches = existing_matches[j] if j < len(existing_matches) else []

                semantic_match = matches[0] if matches and matches[0].score >= threshold else None
                match = exact_matches.get(key) or semantic_match
                if match:
                    payload = match.payload or {}
                    linked = set(payload.get("linked_memory_ids", []))
                    linked |= memory_ids
                    payload["linked_memory_ids"] = sorted(linked)
                    try:
                        self.store.update(vector_id=match.id, vector=None, payload=payload)
                    except Exception as e:
                        logger.debug(f"Entity update failed for '{entity_text}': {e}")
                else:
                    to_insert_vectors.append(valid_vectors[j])
                    to_insert_ids.append(str(uuid.uuid4()))
                    to_insert_payloads.append({
                        "data": entity_text,
                        "entity_type": entity_type,
                        "linked_memory_ids": sorted(memory_ids),
                        **scope,
                    })

            if to_insert_vectors:
                try:
                    self.store.insert(
                        vectors=to_insert_vectors,
                        ids=to_insert_ids,
                        payloads=to_insert_payloads,
                    )
                except Exception as e:
                    logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

    def unlink_memory(self, memory_id, filters):
        """Strip `memory_id` from every entity record scoped to `filters`.

        For each entity whose `linked_memory_ids` contains `memory_id`:
          - remove the id; if the list becomes empty, delete the entity record.
          - otherwise re-embed the entity text and update the payload
            (the vector store's update() requires a vector).

        No-op if the entity store has never been initialized in this process.
        Errors on individual entities are swallowed at debug level; outer
        failures are swallowed at warning level so the primary delete/update
        path is never broken by entity cleanup.
        """
        if not self.initialized:
            return
        try:
            listed = self.store.list(filters=_scope_of(filters), top_k=_LIST_CAP)
            for row in _vector_store_list_rows(listed):
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            self.store.delete(vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                        continue

                    entity_text = payload.get("data")
                    if not isinstance(entity_text, str) or not entity_text:
                        logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                        continue
                    try:
                        vec = self._owner.embedding_model.embed(entity_text, "update")
                    except Exception as e:
                        logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                        continue
                    try:
                        self.store.update(
                            vector_id=row.id,
                            vector=vec,
                            payload={**payload, "linked_memory_ids": remaining},
                        )
                    except Exception as e:
                        logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def bulk_clear(self, filters):
        """Delete all entity records matching the given scope filters.

        Used by delete_all, which would otherwise clean the entity store inside
        every _delete_memory: each of those lists the whole entity collection,
        so deleting N memories meant N full scans. On the async side it also
        avoids the read-modify-write race between concurrent _delete_memory
        coroutines touching the same linked_memory_ids list.
        """
        if not self.initialized:
            return
        try:
            listed = self.store.list(filters=_scope_of(filters), top_k=_LIST_CAP)
            for row in _vector_store_list_rows(listed):
                try:
                    self.store.delete(vector_id=row.id)
                except Exception as e:
                    logger.debug(f"Bulk entity delete failed for id={row.id}: {e}")
        except Exception as e:
            logger.warning(f"Bulk entity store cleanup failed: {e}")

    def boosts_for(self, query_entities, filters):
        """Per-memory retrieval boosts from the entities in a query.

        Each query entity is embedded and searched against the entity store;
        every match above `_BOOST_FLOOR` boosts the memories it links to. The
        boost is damped by how many memories an entity links to, so an entity
        that appears everywhere says nothing about any one memory.

        Returns: dict of memory_id (str) -> max boost, in [0, ENTITY_BOOST_WEIGHT].
        """
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = self.normalize(entity_text)
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        scope = _scope_of(filters)
        memory_boosts = {}

        try:
            entity_texts = [text for _, text in deduped]
            embeddings = self._owner.embedding_model.embed_batch(entity_texts, "search")

            if len(embeddings) != len(entity_texts):
                logger.warning(
                    "embed_batch returned %d vectors for %d texts — skipping entity boost",
                    len(embeddings),
                    len(entity_texts),
                )
                return memory_boosts

            store = self.store

            def _search_entity(entity_text, embedding):
                return store.search(query=entity_text, vectors=embedding, top_k=500, filters=scope)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(_search_entity, text, emb)
                    for text, emb in zip(entity_texts, embeddings)
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        matches = future.result()
                    except Exception as e:
                        logger.warning("Entity boost search failed for one entity: %s", e)
                        continue

                    for match in matches:
                        similarity = getattr(match, "score", 0.0)
                        if similarity < _BOOST_FLOOR:
                            continue

                        payload = getattr(match, "payload", {})
                        linked_memory_ids = payload.get("linked_memory_ids", [])
                        if not isinstance(linked_memory_ids, list):
                            continue

                        num_linked = max(len(linked_memory_ids), 1)
                        memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                        boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                        for memory_id in linked_memory_ids:
                            if memory_id:
                                memory_key = str(memory_id)
                                memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    def reset(self):
        """Drop the entity collection and forget it, so the next use rebuilds it."""
        if not self.initialized:
            return
        try:
            self._store.reset()
        except Exception as e:
            logger.warning(f"Failed to reset entity store: {e}")
        self._store = None
