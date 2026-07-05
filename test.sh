#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source lib/common.sh
load_env

MODEL="unsloth/Llama-3.2-1B-Instruct-GGUF"
QUANT="Q4_K_M"
PORT=7999
SVC="llama-$(slugify "${MODEL}:${QUANT}")"
OUT="compose/${SVC}.yml"

cleanup() { docker compose -f "$OUT" down >/dev/null 2>&1 || true; }
trap cleanup EXIT

header "end-to-end test"
log_info "Launching tiny model"
./launch.sh --model "$MODEL" --quant "$QUANT" --port "$PORT" --ctx 4096

log_info "Waiting for /health (up to ~5 min for first download)"
for _ in $(seq 1 150); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "${ready:-}" == "1" ]] || die "Server never became healthy"
log_info "Server healthy"

resp=$(curl -fsS "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}]}')
content=$(printf '%s' "$resp" | jq -r '.choices[0].message.content // empty')
[[ -n "$content" ]] || die "Empty completion. Raw: $resp"
log_info "Completion received: $(bold "$content")"
log_info "$(green 'E2E PASS')"
