#!/usr/bin/env bash
# Shared helpers for the llama.cpp docker toolkit. Source, don't execute.

# --- terminal styling (TTY-aware: plain output when piped or NO_COLOR set) ---
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  _C_RESET=$'\033[0m'; _C_BOLD=$'\033[1m'; _C_DIM=$'\033[2m'
  _C_RED=$'\033[31m'; _C_GREEN=$'\033[32m'; _C_YELLOW=$'\033[33m'
  _C_BLUE=$'\033[34m'; _C_MAGENTA=$'\033[35m'; _C_CYAN=$'\033[36m'
else
  _C_RESET='' _C_BOLD='' _C_DIM='' _C_RED='' _C_GREEN='' _C_YELLOW='' _C_BLUE='' _C_MAGENTA='' _C_CYAN=''
fi

bold()   { printf '%s%s%s' "${_C_BOLD}" "$*" "${_C_RESET}"; }
dim()    { printf '%s%s%s' "${_C_DIM}" "$*" "${_C_RESET}"; }
cyan()   { printf '%s%s%s' "${_C_CYAN}" "$*" "${_C_RESET}"; }
green()  { printf '%s%s%s' "${_C_GREEN}" "$*" "${_C_RESET}"; }
yellow() { printf '%s%s%s' "${_C_YELLOW}" "$*" "${_C_RESET}"; }

# hr [width]: a dim horizontal rule
hr() { local w="${1:-40}" line; printf -v line '%*s' "$w" ''; printf '%s%s%s\n' "${_C_DIM}" "${line// /─}" "${_C_RESET}"; }

# _box <title>: ANSI boxed banner sized to the title
_box() {
  local title="$1" w=$(( ${#1} + 2 )) bar
  printf -v bar '%*s' "$w" ''; bar="${bar// /─}"
  printf '%s╭%s╮%s\n' "${_C_CYAN}" "$bar" "${_C_RESET}"
  printf '%s│%s %s%s%s %s│%s\n' "${_C_CYAN}" "${_C_RESET}" "${_C_BOLD}" "$title" "${_C_RESET}" "${_C_CYAN}" "${_C_RESET}"
  printf '%s╰%s╯%s\n' "${_C_CYAN}" "$bar" "${_C_RESET}"
}

# header <title>: figlet banner if available, else an ANSI box
header() {
  if command -v figlet >/dev/null 2>&1; then
    printf '%s' "${_C_CYAN}"; figlet -- "$1"; printf '%s\n' "${_C_RESET}"
  else
    _box "$1"
  fi
}

# fzf_select <prompt>: choose one line from stdin. fzf if present, else `select`.
fzf_select() {
  local prompt="${1:-Select}" input opt
  input=$(cat)
  [[ -z "$input" ]] && return 1
  if command -v fzf >/dev/null 2>&1; then
    printf '%s\n' "$input" | fzf --height=40% --reverse --prompt="${prompt} > "
    return
  fi
  printf '%s\n' "$(dim "${prompt}:")" >&2
  local PS3="#? "
  # ponytail: unquoted $input splits on spaces — fine for single-token options
  # (quant names, container names); use mapfile + array if multi-word ever needed.
  select opt in $input; do
    [[ -n "$opt" ]] && { printf '%s\n' "$opt"; return 0; }
  done < /dev/tty
  return 1  # EOF / Ctrl-D without a pick → signal cancel so callers can fall back
}

log_info() { printf '%s✔%s %s\n' "${_C_GREEN}" "${_C_RESET}" "$*"; }
log_warn() { printf '%s!%s %s\n' "${_C_YELLOW}" "${_C_RESET}" "$*" >&2; }
log_err()  { printf '%s✗%s %s\n' "${_C_RED}" "${_C_RESET}" "$*" >&2; }
die()      { log_err "$*"; exit 1; }

# slugify <string> -> docker-safe lowercase name
slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
}

# load .env from current dir into environment
load_env() {
  [[ -f .env ]] || die ".env not found — copy .env.example to .env first"
  set -a; source .env; set +a
}

# env_set KEY VALUE: set (or add) KEY=VALUE in ./.env
env_set() {
  local k="$1" v="$2"
  if grep -q "^${k}=" .env 2>/dev/null; then
    sed -i.bak -E "s|^${k}=.*|${k}=${v}|" .env
  else
    printf '%s=%s\n' "$k" "$v" >> .env
  fi
  log_info "${k}=${v} in .env"
}

# is_unified_gpu: true when the GPU shares the host's memory pool (DGX Spark
# GB10, Jetson). Those parts have no separate VRAM and nvidia-smi reports the
# memory columns as [N/A]. Keep the name list in sync with infra/docker.go.
is_unified_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
    | grep -qiE 'GB10|Spark|Orin|Xavier|Thor|Tegra'
}

# parse_cuda_tag: read nvidia-smi output on stdin, echo "<major>.<minor>.0"
parse_cuda_tag() {
  local ver
  ver=$(grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | grep -oE '[0-9]+\.[0-9]+' || true)
  if [[ -n "$ver" ]]; then
    printf '%s.0\n' "$ver"
  else
    printf '%s\n' "${DEFAULT_CUDA_TAG:-12.9.0}"
  fi
}

# detect_cuda_tag: run nvidia-smi (if present) through parse_cuda_tag
detect_cuda_tag() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>/dev/null | parse_cuda_tag
  else
    log_warn "nvidia-smi not found — using fallback CUDA tag ${DEFAULT_CUDA_TAG:-12.9.0}"
    printf '%s\n' "${DEFAULT_CUDA_TAG:-12.9.0}"
  fi
}

# parse_cuda_arch: read nvidia-smi compute_cap text on stdin (e.g. "8.9"),
# echo the arch digits ("89"). Echo "default" if no capability is found.
parse_cuda_arch() {
  local cap
  cap=$(grep -oE '[0-9]+\.[0-9]+' | head -n1 | tr -d '.' || true)
  if [[ -n "$cap" ]]; then printf '%s\n' "$cap"; else printf 'default\n'; fi
}

# detect_cuda_arch: query the GPU compute capability via nvidia-smi -> parse_cuda_arch.
# Echoes "default" (build all archs) when nvidia-smi is absent.
detect_cuda_arch() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | parse_cuda_arch
  else
    printf 'default\n'
  fi
}

# parse_quants: read HF model JSON on stdin, echo unique GGUF quant tokens sorted.
# Excludes mmproj/projector files; collapses multi-part shards.
parse_quants() {
  jq -r '.siblings[].rfilename
           | select(endswith(".gguf"))
           | select(test("mmproj"; "i") | not)
           | select(test("projector"; "i") | not)' \
    | grep -oE '(IQ[0-9]+_[A-Z]+|Q[0-9]+_[0-9A-Z_]+|BF16|F16|F32)' \
    | LC_ALL=C sort -u
}

# hf_list_quants <repo>: fetch the HF API and list quants. Non-zero on failure.
hf_list_quants() {
  local repo="$1" json
  json=$(curl -fsSL "https://huggingface.co/api/models/${repo}") \
    || { log_warn "Hugging Face lookup failed for '${repo}'"; return 1; }
  printf '%s' "$json" | parse_quants
}

# render_compose: emit a self-contained per-model compose file from env vars.
# Consumes: SVC MODEL QUANT PORT CTX NGL PARALLEL FLASH_ATTN CONTEXT_SHIFT
#           ALIAS EXTRA_FLAGS MODELS_DIR RESTART
# ponytail: env-var contract instead of 13 positional args — documented above.
render_compose() {
  local hf="$MODEL"
  [[ -n "${QUANT:-}" ]] && hf="${MODEL}:${QUANT}"

  # Build the llama-server argv as a YAML list.
  local -a args=( "-hf" "$hf" "--port" "$PORT" "--host" "0.0.0.0" "--ctx-size" "$CTX" )
  [[ -n "${NGL:-}" ]]      && args+=( "-ngl" "$NGL" )
  [[ -n "${PARALLEL:-}" ]] && args+=( "--parallel" "$PARALLEL" )
  [[ "${FLASH_ATTN:-off}" == "on" ]]    && args+=( "--flash-attn" "on" )
  [[ "${CONTEXT_SHIFT:-off}" == "on" ]] && args+=( "--context-shift" )
  [[ -n "${ALIAS:-}" ]]    && args+=( "--alias" "$ALIAS" )
  # shellcheck disable=SC2206  # intentional word-split of raw passthrough flags
  [[ -n "${EXTRA_FLAGS:-}" ]] && args+=( ${EXTRA_FLAGS} )

  printf 'services:\n'
  printf '  %s:\n' "$SVC"
  printf '    image: local/llama.cpp:server-cuda\n'
  printf '    container_name: %s\n' "$SVC"
  printf '    restart: %s\n' "${RESTART:-unless-stopped}"
  printf '    labels:\n'
  printf '      com.llamacpp.managed: "true"\n'
  printf '      com.llamacpp.model: "%s"\n' "$hf"
  printf '    deploy:\n'
  printf '      resources:\n'
  printf '        reservations:\n'
  printf '          devices:\n'
  printf '            - driver: nvidia\n'
  printf '              count: all\n'
  printf '              capabilities: [gpu]\n'
  printf '    volumes:\n'
  printf '      - "${MODELS_DIR:-./.models}:/root/.cache/huggingface"\n'
  printf '    ports:\n'
  printf '      - "%s:%s"\n' "$PORT" "$PORT"
  printf '    healthcheck:\n'
  printf '      test: ["CMD", "curl", "-f", "http://localhost:%s/health"]\n' "$PORT"
  printf '      interval: 30s\n      timeout: 5s\n      retries: 5\n      start_period: 120s\n'
  printf '    command:\n'
  local a
  for a in "${args[@]}"; do printf '      - "%s"\n' "$a"; done
}
