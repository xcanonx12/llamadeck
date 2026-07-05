#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
touch .env  # load_env needs it
out_file="compose/llama-org-model-gguf-q4-k-m.yml"
rm -f "$out_file"
./launch.sh --model org/Model-GGUF --quant Q4_K_M --port 7123 --ctx 4096 --dry-run
[[ -f "$out_file" ]] || { echo "FAIL: compose file not rendered"; exit 1; }
grep -q '"7123:7123"' "$out_file" || { echo "FAIL: port not in file"; exit 1; }
grep -q 'com.llamacpp.managed: "true"' "$out_file" || { echo "FAIL: label missing"; exit 1; }
rm -f "$out_file"
echo "PASS test_launch"
