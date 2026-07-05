#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
touch .env
out_file="compose/llama-org-model-gguf-q4-k-m.yml"
rm -f "$out_file"
# 12 prompts, one answer per line (blanks accept the default):
#  1 model  2 quant  3 port  4 ctx  5 ngl  6 parallel  7 flash  8 ctx-shift
#  9 alias  10 extra  11 restart  12 launch?=n
# 'org/Model-GGUF' is a deliberate non-existent repo so quant discovery returns
# nothing and prompt 2 falls through to the manual quant ask (keeps stdin aligned).
./menu.sh <<'EOF'
org/Model-GGUF
Q4_K_M
7200
4096







n
EOF
[[ -f "$out_file" ]] || { echo "FAIL: menu did not render compose"; exit 1; }
grep -q '"7200:7200"' "$out_file" || { echo "FAIL: port not applied"; exit 1; }
# model+quant must flow through to the -hf value (guards against prompt/answer misalignment)
grep -q '"org/Model-GGUF:Q4_K_M"' "$out_file" || { echo "FAIL: -hf model:quant misaligned"; exit 1; }
# alias defaulted to blank -> no --alias flag should appear
if grep -q '"--alias"' "$out_file"; then echo "FAIL: unexpected --alias (misalignment)"; exit 1; fi
rm -f "$out_file"
echo "PASS test_menu"
