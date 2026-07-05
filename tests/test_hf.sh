#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source lib/common.sh
command -v jq >/dev/null || { echo "SKIP test_hf (jq not installed)"; exit 0; }

got=$(parse_quants < tests/fixtures/hf_model.json | paste -sd, -)
# multi-part Q8_0 collapses to one; mmproj-F16 excluded (not a quant of the model weights)
[[ "$got" == "IQ4_XS,Q4_K_M,Q5_K_M,Q8_0" ]] || { echo "FAIL parse_quants got '$got'"; exit 1; }
echo "PASS test_hf"
