"""Retrieval-quality and context-size regression gate for mem0 search.

Exists so a change that makes search less accurate, or quietly pushes more text
into an agent's context, fails CI instead of shipping. Numbers are compared to
baseline.json; a deliberate change rewrites it and shows up in review.

    python -m tests.quality.harness                    # print the report
    python -m tests.quality.harness --update-baseline  # accept the current numbers
    python -m tests.quality.harness --regen            # re-record embeddings from ollama, then rebaseline

All three need MEM0_QUALITY_PG, a connection string for a pgvector database.
"""

import argparse
import base64
import json
import math
import os
import struct
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DIR = Path(__file__).parent
CORPUS_PATH = _DIR / "corpus.json"
EMBEDDINGS_PATH = _DIR / "embeddings.json"
BASELINE_PATH = _DIR / "baseline.json"

EMBEDDING_DIMS = 768

# NOTE: the recall hook's own settings. Accuracy is measured through the same
# cutoff an agent sees, so a gold memory pushed under the threshold counts as a
# miss, which it is in practice.
SEARCH_LIMIT = 10
SEARCH_THRESHOLD = 0.45

# NOTE: tight on purpose. Runs are bit-for-bit deterministic, so tolerance is
# not absorbing noise, only deciding how small a regression may pass unreviewed.
# Re-introducing the two ranking bugs fixed in PRs #18 and #19 moved MRR by
# 0.0011 and 0.0045 and unanswerable-query tokens by 2.4% and 3.0%; looser
# bounds let both through.
ACCURACY_TOLERANCE = {"recall@5": 0.001, "mrr": 0.001, "ndcg@10": 0.001}
SIZE_TOLERANCE = {
    "context_tokens_mean": 0.02,
    "context_tokens_p95": 0.02,
    "negative_context_tokens_mean": 0.02,
    "add_prompt_tokens_mean": 0.02,
}


class MissingEmbedding(KeyError):
    pass


class FrozenEmbedder:
    """Serves embeddings recorded from nomic-embed-text, so CI needs no model and
    every run sees identical vectors. Regenerate with `python -m tests.quality.harness --regen`.
    """

    def __init__(self, vectors):
        self.vectors = vectors
        self.missing = set()

    def embed(self, text, memory_action=None):
        if text not in self.vectors:
            self.missing.add(text)
            raise MissingEmbedding(text)
        return self.vectors[text]

    def embed_batch(self, texts, memory_action="add"):
        unknown = [t for t in texts if t not in self.vectors]
        if unknown:
            self.missing.update(unknown)
            raise MissingEmbedding(unknown)
        return [self.vectors[t] for t in texts]


class _RecordingEmbedder:
    def __init__(self, real):
        self.real = real
        self.vectors = {}

    def embed(self, text, memory_action=None):
        self.vectors[text] = self.real.embed(text, memory_action)
        return self.vectors[text]

    def embed_batch(self, texts, memory_action="add"):
        out = self.real.embed_batch(texts, memory_action)
        self.vectors.update(zip(texts, out))
        return out


class _StubLLM:
    """Records every extraction prompt and extracts nothing, so add probes
    measure prompt size without changing the corpus."""

    def __init__(self):
        self.calls = []

    def generate_response(self, messages, **kwargs):
        self.calls.append(messages)
        return '{"memory": []}'


def ranking_metrics(ranked_ids, gold):
    hits = [i for i, mem_id in enumerate(ranked_ids) if mem_id in gold]
    dcg = sum(1 / math.log2(i + 2) for i in hits if i < 10)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return {
        "recall@5": sum(1 for i in hits if i < 5) / len(gold),
        "mrr": 1 / (hits[0] + 1) if hits else 0.0,
        "ndcg@10": dcg / ideal,
    }


def estimate_tokens(text):
    # chars/4: deterministic and dependency-free. Only ever compared against
    # itself, so the absolute error does not matter.
    return len(text) / 4


def _p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _encode(vector):
    # int8 with a per-vector scale keeps the fixture near 1 KB per string. The
    # rounding is fine because baseline and every run read the same values.
    scale = max(abs(x) for x in vector) or 1.0
    quantized = [round(x / scale * 127) for x in vector]
    return base64.b64encode(struct.pack(f"<f{len(vector)}b", scale, *quantized)).decode()


def _decode(blob):
    raw = base64.b64decode(blob)
    scale, *quantized = struct.unpack(f"<f{len(raw) - 4}b", raw)
    return [q / 127 * scale for q in quantized]


def load_vectors():
    return {text: _decode(blob) for text, blob in json.loads(EMBEDDINGS_PATH.read_text()).items()}


def load_baseline():
    return json.loads(BASELINE_PATH.read_text())


def _build_memory(connection_string, collection, history_path):
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    config = MemoryConfig(
        vector_store={
            "provider": "pgvector",
            "config": {
                "connection_string": connection_string,
                "collection_name": collection,
                "embedding_model_dims": EMBEDDING_DIMS,
                # NOTE: exact search. pgvector's HNSW picks graph levels at
                # random, so the same inserts gave different neighbours from run
                # to run and the gate failed on clean code.
                "hnsw": False,
            },
        },
        # Never called: the embedder and LLM are swapped out below. openai is
        # chosen because it constructs without a network round trip.
        embedder={"provider": "openai", "config": {"api_key": "unused", "embedding_dims": EMBEDDING_DIMS}},
        llm={"provider": "openai", "config": {"api_key": "unused"}},
        history_db_path=history_path,
    )
    return Memory(config)


def _drop_collections(connection_string, collection):
    import psycopg

    with psycopg.connect(connection_string, autocommit=True) as conn:
        for table in (collection, f"{collection}_entities"):
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')


def _ingest(memory, corpus, now):
    user_id = corpus["user_id"]
    corpus_id_of = {}
    for mem in corpus["memories"]:
        created_at = (now - timedelta(days=mem["age_days"])).isoformat()
        [added] = memory.add(
            [{"role": "user", "content": mem["text"]}], user_id=user_id, infer=False, timestamp=created_at
        )["results"]
        corpus_id_of[added["id"]] = mem["id"]
        # NOTE: infer=False stores the text but skips entity linking, which the
        # production add pipeline always does. Without this the entity boost
        # would never fire and the gate could not see it regress.
        memory.entities.link_batch([(added["id"], mem["text"], None, {}, [])], {"user_id": user_id})
    return corpus_id_of


def _search(memory, corpus, corpus_id_of):
    results = []
    for query in corpus["queries"]:
        found = memory.search(
            query["text"],
            filters={"user_id": corpus["user_id"]},
            top_k=SEARCH_LIMIT,
            threshold=SEARCH_THRESHOLD,
        )["results"]
        results.append(
            {
                "id": query["id"],
                "text": query["text"],
                "gold": query["gold"],
                "ranked": [corpus_id_of[r["id"]] for r in found],
                "tokens": sum(estimate_tokens(r["memory"]) for r in found),
            }
        )
    return results


def _probe_adds(memory, corpus, llm):
    for text in corpus["add_probes"]:
        memory.add(text, user_id=corpus["user_id"])
    return [sum(estimate_tokens(m["content"]) for m in call) for call in llm.calls]


def run(connection_string, embedder):
    corpus = json.loads(CORPUS_PATH.read_text())
    collection = f"quality_{uuid.uuid4().hex[:12]}"
    llm = _StubLLM()
    with tempfile.TemporaryDirectory() as tmp:
        memory = _build_memory(connection_string, collection, f"{tmp}/history.db")
        memory.embedding_model = embedder
        memory.llm = llm
        try:
            corpus_id_of = _ingest(memory, corpus, datetime.now(timezone.utc))
            queries = _search(memory, corpus, corpus_id_of)
            prompt_tokens = _probe_adds(memory, corpus, llm)
        finally:
            memory.close()
            _drop_collections(connection_string, collection)
    return summarize(queries, prompt_tokens)


def summarize(queries, prompt_tokens):
    positives = [q for q in queries if q["gold"]]
    negatives = [q for q in queries if not q["gold"]]
    per_query = [ranking_metrics(q["ranked"], set(q["gold"])) for q in positives]
    metrics = {k: sum(m[k] for m in per_query) / len(per_query) for k in ACCURACY_TOLERANCE}
    metrics["context_tokens_mean"] = sum(q["tokens"] for q in queries) / len(queries)
    metrics["context_tokens_p95"] = _p95([q["tokens"] for q in queries])
    metrics["negative_context_tokens_mean"] = sum(q["tokens"] for q in negatives) / len(negatives)
    metrics["add_prompt_tokens_mean"] = sum(prompt_tokens) / len(prompt_tokens)
    misses = [
        {"id": q["id"], "text": q["text"], "gold": q["gold"], "top5": q["ranked"][:5]}
        for q, m in zip(positives, per_query)
        if m["recall@5"] < 1
    ]
    return {"metrics": {k: round(v, 4) for k, v in metrics.items()}, "misses": misses}


def regressions(current, baseline):
    found = []
    for key, tolerance in ACCURACY_TOLERANCE.items():
        if current[key] < baseline[key] - tolerance:
            found.append(f"{key} fell from {baseline[key]} to {current[key]} (tolerance {tolerance})")
    for key, growth in SIZE_TOLERANCE.items():
        # A zero baseline would make any growth infinite; allow one token.
        limit = baseline[key] * (1 + growth) if baseline[key] else 1
        if current[key] > limit:
            found.append(f"{key} grew from {baseline[key]} to {current[key]} (limit {limit:.1f})")
    return found


def format_report(report, baseline=None):
    lines = ["metric                          current   baseline"]
    for key, value in report["metrics"].items():
        base = baseline.get(key, "-") if baseline else "-"
        lines.append(f"{key:30} {value:>9} {base:>10}")
    lines.append(f"\nqueries with a gold memory outside the top 5: {len(report['misses'])}")
    for miss in report["misses"]:
        lines.append(f"  {miss['id']} {miss['text']!r}: gold {miss['gold']}, top5 {miss['top5']}")
    return "\n".join(lines)


def _regen(connection_string):
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.ollama import OllamaEmbedding

    recorder = _RecordingEmbedder(
        OllamaEmbedding(BaseEmbedderConfig(model="nomic-embed-text", embedding_dims=EMBEDDING_DIMS))
    )
    run(connection_string, recorder)
    EMBEDDINGS_PATH.write_text(
        json.dumps({text: _encode(vec) for text, vec in sorted(recorder.vectors.items())}, indent=0) + "\n"
    )
    print(f"recorded {len(recorder.vectors)} embeddings to {EMBEDDINGS_PATH.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--regen", action="store_true")
    args = parser.parse_args(argv)

    connection_string = os.environ["MEM0_QUALITY_PG"]
    if args.regen:
        _regen(connection_string)

    embedder = FrozenEmbedder(load_vectors())
    report = run(connection_string, embedder)
    if embedder.missing:
        sys.exit(f"no frozen embedding for {sorted(embedder.missing)}; rerun with --regen")

    baseline = load_baseline() if BASELINE_PATH.exists() else None
    print(format_report(report, baseline))
    if args.regen or args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(report["metrics"], indent=2) + "\n")
        print(f"\nwrote {BASELINE_PATH.name}")


if __name__ == "__main__":
    main()
