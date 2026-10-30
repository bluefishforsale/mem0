#!/usr/bin/env bash
# Builds the nightly stack from this checkout, runs LOCOMO, copies results out,
# and tears everything down. Used by the homelab nightly workflow and by hand.
#
# Required: EXTRACTION_OPENAI_API_KEY, JUDGE_BASE_URL, JUDGE_API_KEY.
# Optional: JUDGE_MODEL, CONVERSATIONS (default 0,1), MAX_QUESTIONS,
#           PUSHGATEWAY_URL, COMMIT_SHA, OUT_DIR (default ./out next to this file).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
project="mem0-nightly-${RUN_ID:-$$}"
out="${OUT_DIR:-$here/out}"
compose=(docker compose -p "$project" -f "$here/docker-compose.yml")

cleanup() {
  status=$?
  # The stack is torn down with its logs, so a failed run has to say why first.
  if [ "$status" -ne 0 ]; then
    "${compose[@]}" logs --no-color --tail 200 mem0 postgres ollama || true
  fi
  docker rm -f "$project-bench" >/dev/null 2>&1 || true
  "${compose[@]}" --profile bench down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${compose[@]}" --profile bench build
"${compose[@]}" up -d --wait postgres ollama

# Same thread-pinned derivative of nomic-embed-text that ocean runs; the stock
# tag spawns one thread per host core and crawls. See the homelab mem0 playbook.
"${compose[@]}" exec -T ollama ollama pull nomic-embed-text
"${compose[@]}" exec -T ollama sh -c \
  'printf "FROM nomic-embed-text\nPARAMETER num_thread 6\n" > /tmp/Modelfile && ollama create nomic-embed-homelab -f /tmp/Modelfile'

"${compose[@]}" up -d --wait mem0

status=0
"${compose[@]}" run --name "$project-bench" bench || status=$?

mkdir -p "$out"
docker cp "$project-bench:/out/." "$out/" || true
exit "$status"
