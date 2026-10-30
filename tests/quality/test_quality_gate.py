import os

import pytest

from tests.quality.harness import (
    FrozenEmbedder,
    MissingEmbedding,
    format_report,
    load_baseline,
    load_vectors,
    ranking_metrics,
    regressions,
    run,
)


def test_search_accuracy_and_context_size_have_not_regressed():
    """The gate. See tests/quality/harness.py for what it measures and how to
    accept a deliberate change (`--update-baseline`)."""
    connection_string = os.environ.get("MEM0_QUALITY_PG")
    if not connection_string:
        # NOTE: CI sets MEM0_REQUIRE_QUALITY so a missing database fails the run.
        # A skip is how the spaCy tests went unrun for months.
        if os.environ.get("MEM0_REQUIRE_QUALITY"):
            pytest.fail("MEM0_REQUIRE_QUALITY is set but MEM0_QUALITY_PG is not")
        pytest.skip("MEM0_QUALITY_PG not set; the quality gate needs a pgvector database")

    embedder = FrozenEmbedder(load_vectors())
    report = run(connection_string, embedder)
    baseline = load_baseline()

    assert not embedder.missing, (
        f"no frozen embedding for {sorted(embedder.missing)}; run `python -m tests.quality.harness --regen`"
    )
    found = regressions(report["metrics"], baseline)
    assert not found, "\n".join(found) + "\n\n" + format_report(report, baseline)


def test_ranking_metrics_score_a_known_ranking():
    """recall@5, MRR and nDCG@10 on a ranking worked out by hand.

    Gold is {a, b}. `a` sits at rank 2 and `b` at rank 7, so recall@5 sees one
    of two, the first hit gives MRR 1/2, and nDCG@10 is
    (1/log2(3) + 1/log2(8)) / (1/log2(2) + 1/log2(3)).
    """
    ranked = ["x", "a", "x2", "x3", "x4", "x5", "b"]

    m = ranking_metrics(ranked, {"a", "b"})

    assert m["recall@5"] == pytest.approx(0.5)
    assert m["mrr"] == pytest.approx(0.5)
    assert m["ndcg@10"] == pytest.approx((1 / 1.584963 + 1 / 3) / (1 + 1 / 1.584963), rel=1e-5)


def test_frozen_embedder_records_strings_it_has_no_vector_for():
    """An unknown string must surface, not vanish.

    EntityStore.link_batch and boosts_for swallow embedder exceptions, so
    raising alone would let a spaCy upgrade that extracts a new entity
    silently drop entity linking from the gate. The miss is recorded so the
    gate can fail on it after the run.
    """
    embedder = FrozenEmbedder({"known": [1.0, 0.0]})

    assert embedder.embed("known") == [1.0, 0.0]
    with pytest.raises(MissingEmbedding):
        embedder.embed_batch(["known", "unknown"])
    assert embedder.missing == {"unknown"}
