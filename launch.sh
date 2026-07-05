#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source lib/common.sh
load_env

# Defaults
MODEL="" QUANT="" PORT=8080 CTX=32768 NGL=999 PARALLEL="" \
FLASH_ATTN=on CONTEXT_SHIFT=on ALIAS="" RESTART=unless-stopped EXTRA_FLAGS="" DRY_RUN=0

usage() { grep -E '^# Usage' "$0" | sed 's/^# //'; exit 1; }
# Usage: launch.sh --model <repo> [--quant Q] [--port N] [--ctx N] [--ngl N]
# Usage:   [--parallel N] [--flash-attn on|off] [--context-shift on|off]
# Usage:   [--alias A] [--restart POLICY] [--extra "raw flags"] [--dry-run]

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --quant) QUANT="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --ngl) NGL="$2"; shift 2;;
    --parallel) PARALLEL="$2"; shift 2;;
    --flash-attn) FLASH_ATTN="$2"; shift 2;;
    --context-shift) CONTEXT_SHIFT="$2"; shift 2;;
    --alias) ALIAS="$2"; shift 2;;
    --restart) RESTART="$2"; shift 2;;
    --extra) EXTRA_FLAGS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage;;
    *) die "Unknown option: $1";;
  esac
done

[[ -n "$MODEL" ]] || die "--model is required"

# Auto-pick a quant if none given (menu.sh does interactive selection).
if [[ -z "$QUANT" ]]; then
  if QUANT=$(hf_list_quants "$MODEL" | head -n1) && [[ -n "$QUANT" ]]; then
    log_info "No --quant given; using first available: $QUANT"
  else
    log_warn "Could not discover a quant; launching without one (repo must be non-GGUF or single-file)"
    QUANT=""
  fi
fi

SVC="llama-$(slugify "${MODEL}${QUANT:+:$QUANT}")"
mkdir -p compose

# Resolve MODELS_DIR to an absolute path: the compose file lives in compose/, and
# `docker compose -f compose/x.yml` resolves relative bind paths against that dir —
# so a relative ./.models would wrongly become compose/.models. Absolute keeps every
# model in the one shared repo-root cache.
mkdir -p "${MODELS_DIR:-./.models}"
MODELS_DIR=$(cd "${MODELS_DIR:-./.models}" && pwd)
OUT="compose/${SVC}.yml"

export SVC MODEL QUANT PORT CTX NGL PARALLEL FLASH_ATTN CONTEXT_SHIFT ALIAS EXTRA_FLAGS MODELS_DIR RESTART
render_compose > "$OUT"
log_info "Rendered $OUT"

if [[ "$DRY_RUN" == "1" ]]; then
  log_info "--dry-run: not starting the container"
  exit 0
fi

# Refuse if the port is already listening.
if command -v ss >/dev/null 2>&1 && ss -ltnH 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
  die "Port ${PORT} is already in use — pick another with --port"
fi

docker compose -f "$OUT" up -d
log_info "Started $(bold "${SVC}")"
printf '  %s %s\n' "$(green '➜ server:')" "$(bold "http://localhost:${PORT}")"
printf '  %s %s\n' "$(dim 'manage:')" "$(dim "./manage.sh status   |   ./manage.sh stop ${SVC}")"
