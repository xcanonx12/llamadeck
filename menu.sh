#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source lib/common.sh
load_env

ask() { local prompt="$1" default="${2:-}" ans; read -r -p "$(cyan "$prompt")${default:+ $(dim "[$default]")}: " ans || true; printf '%s' "${ans:-$default}"; }

header "llama.cpp launcher"
MODEL=$(ask "Hugging Face model repo (e.g. unsloth/Qwen3-VL-8B-Instruct-GGUF)")
[[ -n "$MODEL" ]] || die "model is required"

# Quant: discover and pick with fzf_select, else manual.
echo "$(dim 'Looking up available quants...')"
mapfile -t QUANTS < <(hf_list_quants "$MODEL" 2>/dev/null || true)
QUANT=""
if [[ ${#QUANTS[@]} -gt 0 ]]; then
  QUANT=$(printf '%s\n' "${QUANTS[@]}" | fzf_select "quant") || QUANT="${QUANTS[0]}"
  log_info "quant: ${QUANT:-<none>}"
else
  log_warn "No quants discovered (network/repo). Enter manually if needed."
  QUANT=$(ask "Quant (blank for none)")
fi

printf '%s\n' "$(bold '— llama-server settings —')"
PORT=$(ask "Port" 8080)
CTX=$(ask "Context size" 32768)
NGL=$(ask "GPU layers (-ngl, blank=all)" 999)
PARALLEL=$(ask "Parallel slots (blank=default)")
FLASH_ATTN=$(ask "Flash attention on/off" on)
CONTEXT_SHIFT=$(ask "Context shift on/off" on)
ALIAS=$(ask "Model alias for the API (blank=default)")
EXTRA_FLAGS=$(ask "Extra raw llama-server flags (blank=none)")

printf '%s\n' "$(bold '— compose / docker settings —')"
RESTART=$(ask "Restart policy" unless-stopped)

printf '%s\n' "$(bold '— review —')"
printf 'model=%s quant=%s port=%s ctx=%s ngl=%s parallel=%s flash=%s ctx-shift=%s alias=%s restart=%s extra=%s\n' \
  "$MODEL" "$QUANT" "$PORT" "$CTX" "$NGL" "$PARALLEL" "$FLASH_ATTN" "$CONTEXT_SHIFT" "$ALIAS" "$RESTART" "$EXTRA_FLAGS"
GO=$(ask "Launch now? y/n (n = just render the compose file)" y)

ARGS=( --model "$MODEL" --port "$PORT" --ctx "$CTX" --flash-attn "$FLASH_ATTN" --context-shift "$CONTEXT_SHIFT" --restart "$RESTART" )
[[ -n "$QUANT" ]]       && ARGS+=( --quant "$QUANT" )
[[ -n "$NGL" ]]         && ARGS+=( --ngl "$NGL" )
[[ -n "$PARALLEL" ]]    && ARGS+=( --parallel "$PARALLEL" )
[[ -n "$ALIAS" ]]       && ARGS+=( --alias "$ALIAS" )
[[ -n "$EXTRA_FLAGS" ]] && ARGS+=( --extra "$EXTRA_FLAGS" )
[[ "$GO" =~ ^[Yy] ]]    || ARGS+=( --dry-run )

exec ./launch.sh "${ARGS[@]}"
