#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source lib/common.sh

SVC=llama-test MODEL=org/Model-GGUF QUANT=Q4_K_M PORT=7000 CTX=8192 \
NGL=999 PARALLEL= FLASH_ATTN=on CONTEXT_SHIFT=on ALIAS=mymodel \
EXTRA_FLAGS="--no-mmap" MODELS_DIR=./.models RESTART=unless-stopped
export SVC MODEL QUANT PORT CTX NGL PARALLEL FLASH_ATTN CONTEXT_SHIFT ALIAS EXTRA_FLAGS MODELS_DIR RESTART

out=$(render_compose)
grep -q 'image: local/llama.cpp:server-cuda' <<<"$out" || { echo "FAIL image"; exit 1; }
grep -q 'com.llamacpp.managed: "true"'       <<<"$out" || { echo "FAIL label"; exit 1; }
grep -q 'com.llamacpp.model: "org/Model-GGUF:Q4_K_M"' <<<"$out" || { echo "FAIL model label"; exit 1; }
grep -q '"-hf"'                              <<<"$out" || { echo "FAIL -hf"; exit 1; }
grep -q '"org/Model-GGUF:Q4_K_M"'           <<<"$out" || { echo "FAIL hf value"; exit 1; }
grep -q '"7000:7000"'                       <<<"$out" || { echo "FAIL port"; exit 1; }
grep -q '"--flash-attn"'                    <<<"$out" || { echo "FAIL flash"; exit 1; }
grep -q '"--context-shift"'                 <<<"$out" || { echo "FAIL ctx-shift"; exit 1; }
grep -q '"--no-mmap"'                       <<<"$out" || { echo "FAIL extra"; exit 1; }
grep -q '\${MODELS_DIR:-./.models}:/root/.cache/huggingface' <<<"$out" || { echo "FAIL volume"; exit 1; }
grep -q '"on"'                              <<<"$out" || { echo "FAIL flash value"; exit 1; }

# off-path: toggles must NOT appear when disabled
FLASH_ATTN=off; CONTEXT_SHIFT=off
out_off=$(render_compose)
if grep -q '"--flash-attn"'   <<<"$out_off"; then echo "FAIL flash-attn should be absent when off"; exit 1; fi
if grep -q '"--context-shift"' <<<"$out_off"; then echo "FAIL context-shift should be absent when off"; exit 1; fi

# valid YAML if the parser is available (a real parse failure must FAIL, not SKIP)
if command -v python3 >/dev/null && python3 -c 'import yaml' 2>/dev/null; then
  python3 -c 'import sys,yaml; yaml.safe_load(sys.stdin)' <<<"$out" || { echo "FAIL invalid YAML"; exit 1; }
else
  echo "SKIP YAML validation (python3 yaml not available)"
fi
echo "PASS test_render"
