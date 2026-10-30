"""Runs LOCOMO against the nightly stack and turns the result into metrics.

Exists so extraction and end-to-end answer quality, which the per-PR gate in
tests/quality/ cannot see without a real LLM, still get a trend line and an
alert. Pushes to a Prometheus Pushgateway when PUSHGATEWAY_URL is set.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg
import requests

OUT = Path("/out")
CUTOFF = 10
JOB = "mem0_locomo_nightly"


def run_benchmark():
    cmd = [
        sys.executable, "-m", "benchmarks.locomo.run",
        "--project-name", "nightly",
        "--backend", "oss",
        "--mem0-host", "http://mem0:8000",
        "--conversations", os.environ["CONVERSATIONS"],
        "--top-k", str(CUTOFF),
        "--top-k-cutoffs", str(CUTOFF),
        "--provider", "openai",
        "--answerer-model", os.environ["JUDGE_MODEL"],
        "--judge-model", os.environ["JUDGE_MODEL"],
        # llama.cpp serves one request at a time; more workers only queue there.
        "--max-workers", "2",
        "--dataset-path", "/data/locomo10.json",
        "--output-dir", str(OUT),
    ]
    if os.environ.get("MAX_QUESTIONS"):
        cmd += ["--max-questions", os.environ["MAX_QUESTIONS"]]
    subprocess.run(cmd, check=True)


def estimate_tokens(text):
    # chars/4, the same estimate as the per-PR gate.
    return len(text) / 4


def collect():
    [results_path] = sorted(OUT.glob("locomo_results_*.json"))[-1:]
    results = json.loads(results_path.read_text())
    at_cutoff = results["metrics_by_cutoff"][f"top_{CUTOFF}"]
    evaluations = results["evaluations"]

    with psycopg.connect(os.environ["PG_DSN"]) as conn:
        count, mean_chars = conn.execute(
            "SELECT count(*), coalesce(avg(length(payload->>'data')), 0) "
            "FROM mem0_local WHERE payload->>'user_id' LIKE 'locomo_%'"
        ).fetchone()

    returned = [
        sum(estimate_tokens(r.get("memory", "")) for r in e["retrieval"]["search_results"][:CUTOFF])
        for e in evaluations
    ]
    return {
        "score": {"overall": at_cutoff["overall"]["avg_score"],
                  **{k: v["avg_score"] for k, v in at_cutoff["by_category"].items()}},
        "accuracy": {"overall": at_cutoff["overall"]["accuracy"],
                     **{k: v["accuracy"] for k, v in at_cutoff["by_category"].items()}},
        "questions": len(evaluations),
        "memories_extracted": count,
        "memory_tokens_mean": float(mean_chars) / 4,
        "returned_tokens_mean": sum(returned) / len(returned) if returned else 0.0,
    }


def exposition(metrics, duration):
    commit = os.environ.get("COMMIT_SHA", "unknown")
    judge = os.environ["JUDGE_MODEL"]
    lines = [f'mem0_locomo_run_info{{commit="{commit}",judge_model="{judge}"}} 1']
    for category, value in metrics["score"].items():
        lines.append(f'mem0_locomo_score{{category="{category}"}} {value}')
    for category, value in metrics["accuracy"].items():
        lines.append(f'mem0_locomo_accuracy{{category="{category}"}} {value}')
    lines += [
        f"mem0_locomo_questions {metrics['questions']}",
        f"mem0_locomo_memories_extracted {metrics['memories_extracted']}",
        f"mem0_locomo_memory_tokens_mean {metrics['memory_tokens_mean']}",
        f"mem0_locomo_returned_tokens_mean {metrics['returned_tokens_mean']}",
        f"mem0_locomo_duration_seconds {duration}",
        f"mem0_locomo_last_success_timestamp_seconds {time.time()}",
    ]
    return "\n".join(lines) + "\n"


def main():
    started = time.time()
    run_benchmark()
    metrics = collect()
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    gateway = os.environ.get("PUSHGATEWAY_URL")
    if gateway:
        body = exposition(metrics, time.time() - started)
        # PUT replaces the whole group, so a category that disappears does not
        # linger with yesterday's value.
        requests.put(f"{gateway}/metrics/job/{JOB}", data=body, timeout=30).raise_for_status()
        print(f"pushed to {gateway}")


if __name__ == "__main__":
    main()
