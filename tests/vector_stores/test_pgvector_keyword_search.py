"""Defect ledger for PGVector's BM25 arm.

Every case here is a bug that reached production because `keyword_search` is
only ever mocked elsewhere in the suite, so no test ever ran its SQL.
"""

import os
import uuid

import pytest

PGVECTOR_HOST = os.environ.get("PGVECTOR_HOST", "localhost")
PGVECTOR_PORT = int(os.environ.get("PGVECTOR_PORT", "5432"))
PGVECTOR_USER = os.environ.get("PGVECTOR_USER", "mem0")
PGVECTOR_PASS = os.environ.get("PGVECTOR_PASSWORD", "mem0test")
PGVECTOR_DB = os.environ.get("PGVECTOR_DB", "mem0_test")

DIMS = 8


def _pgvector_reachable():
    try:
        import psycopg

        conn = psycopg.connect(
            host=PGVECTOR_HOST, port=PGVECTOR_PORT,
            user=PGVECTOR_USER, password=PGVECTOR_PASS, dbname=PGVECTOR_DB,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _pgvector_reachable(),
    reason=f"pgvector not reachable at {PGVECTOR_HOST}:{PGVECTOR_PORT} with user {PGVECTOR_USER}",
)
class TestKeywordSearchMatching:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mem0.vector_stores.pgvector import PGVector

        self.collection = f"test_kw_{uuid.uuid4().hex[:8]}"
        self.store = PGVector(
            collection_name=self.collection,
            embedding_model_dims=DIMS,
            host=PGVECTOR_HOST,
            port=PGVECTOR_PORT,
            user=PGVECTOR_USER,
            password=PGVECTOR_PASS,
            dbname=PGVECTOR_DB,
            diskann=False,
            hnsw=True,
        )
        texts = [
            "user prefer local inference for the homelab",
            "the deploy pipeline build a container on merge",
            "postgres hold the memory store on ocean",
        ]
        self.store.insert(
            vectors=[[0.1] * DIMS for _ in texts],
            payloads=[{"data": t, "text_lemmatized": t} for t in texts],
            ids=[str(uuid.uuid4()) for _ in texts],
        )
        yield
        self.store.delete_col()

    def test_a_multi_term_query_does_not_require_every_term(self):
        """plainto_tsquery ANDs its terms, so a query of N words only matched a
        memory containing all N. Real queries are long -- the recall hook sends
        seven words -- so the keyword arm returned an empty set and BM25
        contributed nothing to any ranking. Measured against the live store:
        one term matched 66 of 2821 rows, seven terms matched 0.
        """
        results = self.store.keyword_search(query="homelab ocean deploy", top_k=10)

        assert results, "a query whose terms are spread across memories matched nothing"
        assert len(results) >= 3, (
            f"expected all three memories, each holding one term, got {len(results)}"
        )


@pytest.mark.skipif(
    not _pgvector_reachable(),
    reason=f"pgvector not reachable at {PGVECTOR_HOST}:{PGVECTOR_PORT} with user {PGVECTOR_USER}",
)
class TestKeywordSearchScoping:
    @pytest.fixture(autouse=True)
    def setup(self):
        from mem0.vector_stores.pgvector import PGVector

        self.collection = f"test_kw_scope_{uuid.uuid4().hex[:8]}"
        self.store = PGVector(
            collection_name=self.collection,
            embedding_model_dims=DIMS,
            host=PGVECTOR_HOST,
            port=PGVECTOR_PORT,
            user=PGVECTOR_USER,
            password=PGVECTOR_PASS,
            dbname=PGVECTOR_DB,
            diskann=False,
            hnsw=True,
        )
        # Same keyword, two owners. Only the filter separates them.
        rows = [
            ("alice", "alice keep her password in a vault"),
            ("bob", "bob keep his password in a notebook"),
        ]
        self.store.insert(
            vectors=[[0.1] * DIMS for _ in rows],
            payloads=[
                {"data": text, "text_lemmatized": text, "user_id": user}
                for user, text in rows
            ],
            ids=[str(uuid.uuid4()) for _ in rows],
        )
        yield
        self.store.delete_col()

    def test_a_filtered_keyword_search_does_not_leak_another_users_memories(self):
        """The keyword SQL binds the query and the filter values positionally.
        Nothing else covers this path: every other pgvector test mocks the
        cursor, so a filter bound to the wrong placeholder would still look
        correct in the suite while returning another user's memories.
        """
        results = self.store.keyword_search(
            query="password", top_k=10, filters={"user_id": "alice"}
        )

        assert results, "the filtered search matched nothing at all"
        owners = {r.payload.get("user_id") for r in results}
        assert owners == {"alice"}, f"expected only alice's memories, got owners {owners}"
