<p align="center">
  <img src="docs/images/personal-fork.jpeg" width="800px" alt="MEMO, the personal fork by terracnosaur: a mem0 logo in a purple sombrero and heart-shaped sunglasses">
</p>

# mem0, the personal fork

This is my fork of [mem0ai/mem0](https://github.com/mem0ai/mem0), and it runs the memory layer behind my own agents: a self-hosted server on my homelab, pgvector for storage, a local embedder, and a frontier model doing fact extraction. Upstream is a good memory library with a hosted platform attached. This fork is what happened after I started reading my own memories back and noticed which ones were wrong, stale, or missing, and then went and fixed the reasons why.

Everything below is a difference from upstream. For what mem0 is and how to use it in general, the [upstream docs](https://docs.mem0.ai) still apply. I merge upstream main regularly (merges, not rebases, because this main is published), so the fork tracks their fixes and adds mine on top.

## Memories that stay true

Upstream's add pipeline only ever adds. "User is vegetarian" from 2024 and "User eats meat again" from last week both come back in the same search, and nothing decides between them. Here the extraction step reports which existing memories a new fact makes untrue, and those get stamped `superseded_by` and `superseded_at`, hidden from `search()` and `get_all()`, and logged as a `SUPERSEDE` entry in `history()`. Pass `show_superseded=True` if you want to see them anyway.

The flip side is restatements. Saying the same thing twice no longer stores it twice: anything at or above `dedup_similarity_threshold` (cosine, default 0.95) against an existing memory is rejected. A flagged contradiction is let through that gate on purpose, because a correction is usually worded almost exactly like the thing it corrects. "Scrub runs Tuesday, before the rebalance" corrected to "Thursday, after" used to vanish without a word, leaving the stale memory serving.

How many existing memories the extraction model gets to reason over is `add_context_top_k` (default 10). On a scope with a lot of memories about one topic, a small window quietly misses contradictions, so it's a knob now instead of a hardcoded ten.

## Time actually means something

Upstream writes `created_at` and `updated_at` on every memory and then never reads them for ranking. Search here blends in an exponential freshness term, `0.5 ** (age_days / recency_half_life_days)`, with a 180 day default half-life. It carries the smallest weight in the blend (0.08), so it breaks ties and demotes stale memories without ever beating relevance. Set the half-life to 0 to turn it off.

`add(timestamp=...)` works in the OSS SDK. Upstream raises "not supported" there, which means importing a year-old transcript dates every "last week" to last week. The timestamp sets both the extraction prompt's observation date and the memory's `created_at`, so recency measures when the thing happened rather than when I imported it. It takes ISO dates, datetimes, or Unix epoch seconds, and the REST server's `POST /memories` accepts it too.

## Retrieval that finds what's there

Hybrid search got a pile of fixes, mostly found by watching real queries on my own server miss:

The pgvector keyword path ranks matches instead of requiring every term to appear, so one missing word no longer zeroes out an otherwise perfect hit. The BM25-style curve is fit to the score scale pgvector actually returns, and keyword-only hits make it into the candidate pool, which is also wider before reranking. Scores use fixed blend weights instead of a divisor derived from whatever else was in the batch, so a memory's score doesn't change depending on its neighbours. A keyword-only candidate is charged the lowest semantic score in the pool, so it can beat weak semantic hits with a strong term match but can't leapfrog results it never actually outscored. The score you get back is the one the threshold gated on, and the entity boost floor sits at 0.75 instead of 0.5.

## Config that tells you when you're wrong

`MemoryConfig` uses `extra="forbid"`. Upstream silently drops keys it doesn't recognise, which means a misspelt knob reads as accepted and does nothing, and so does the `graph_store` key that v3 removed. Here both fail validation at config time, which is the only point where the mistake is still cheap to find.

The three tuning knobs above (`recency_half_life_days`, `dedup_similarity_threshold`, `add_context_top_k`) are real config fields with documented defaults. The dedup bar in particular wants retuning per embedder, since embedding models don't share a cosine scale.

## No phoning home

Upstream's `notices.py` evaluated a PostHog feature flag on every add, search, get, update and delete, fired a telemetry event carrying the operation and its parameters, and printed whatever text the flag held to stderr, with A/B variants. Some error messages came from the flag too. I took it out: 1582 lines of module, 52 call sites, and the tests that went with them. Error text in this fork lives in this repository.

## The server

The REST server image builds from this repository's own `mem0/` package. Upstream's Dockerfile built from `server/` alone and pulled `mem0ai` from PyPI, so in a fork every change under `mem0/` was inert once deployed. The image also bakes in spaCy and `en_core_web_sm`. Without spaCy, BM25 stops matching across word forms, and without the model mem0 downloads it on first use, which a LAN-only service shouldn't be doing.

`MEM0_LLM_BASE_URL` and `MEM0_LLM_API_KEY` split the LLM from the embedder, so extraction can go to a frontier model while embeddings stay on local Ollama (the pgvector collection is fixed at the local model's dimensions anyway). Unset, they fall back to the shared `OPENAI_*` values.

Auth config (`JWT_SECRET`, `ADMIN_API_KEY`, `AUTH_DISABLED`) is read per request rather than frozen at import, so a reconfigure changes the auth posture without a restart.

On merge to main, `.github/workflows/build-and-push.yaml` builds the image, starts it against a real pgvector and reads a memory back out before anything is pushed to `ghcr.io/bluefishforsale/mem0-api-server`. My homelab's deploy poller sees that build job succeed and deploys the commit. The smoke test is there because `docker build` happily exits 0 on an image that can't start, and the homelab deploy halts on an unhealthy service rather than rolling back.

## Measuring quality, not vibes

`tests/quality/` is a retrieval regression gate: a synthetic homelab corpus of 460 memories and 73 labelled queries (10 with no right answer) run through the real `search()` and `add()` paths on pgvector, with frozen embeddings and a stub LLM. It checks recall@5, MRR, nDCG@10, returned-context tokens and extraction-prompt tokens against a committed baseline. Runs are bit-for-bit deterministic, so the tolerances are tight enough that reintroducing the old scoring bugs fails CI.

`tests/quality/nightly/` is the end-to-end view: a throwaway compose stack shaped like production that runs LOCOMO from the pinned `evaluation/` submodule and pushes graded score, per-category accuracy, memory counts and returned tokens to a Pushgateway, so quality shows up as a trend line instead of a feeling.

CI also runs the server tests (they had never executed in CI: fastapi wasn't in any installed extra, so every file skipped itself) and fails if the spaCy tests would skip instead of quietly passing.

## What I don't maintain

The TypeScript SDK in `mem0-ts/` is unmaintained here. Nothing I deploy uses it, it has drifted from `mem0/` on purpose, and fixes don't get ported into it. The same goes for the integrations, CLIs and hosted-platform client, which come along with upstream merges and nothing more.

## License

Apache-2.0, same as upstream. The original work is mem0ai's; the mistakes in the differences above are mine.
