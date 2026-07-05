#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source lib/common.sh

# slugify
[[ "$(slugify 'unsloth/Qwen3-VL-8B:Q5_K_M')" == "unsloth-qwen3-vl-8b-q5-k-m" ]] || { echo "FAIL slugify"; exit 1; }
[[ "$(slugify 'Already-clean_123')" == "already-clean-123" ]] || { echo "FAIL slugify clean"; exit 1; }

# load_env dies without .env
( cd "$(mktemp -d)" && source "$OLDPWD/lib/common.sh" && load_env ) 2>/dev/null \
  && { echo "FAIL load_env should have died"; exit 1; } || true

# parse_cuda_tag reads nvidia-smi text on stdin
out=$(printf '| NVIDIA-SMI 555.42  Driver Version: 555.42  CUDA Version: 12.9 |\n' | parse_cuda_tag)
[[ "$out" == "12.9.0" ]] || { echo "FAIL parse_cuda_tag got '$out'"; exit 1; }
out=$(printf 'no cuda here\n' | DEFAULT_CUDA_TAG=12.9.0 parse_cuda_tag)
[[ "$out" == "12.9.0" ]] || { echo "FAIL parse_cuda_tag fallback got '$out'"; exit 1; }

# parse_cuda_arch: compute_cap "8.9" -> "89"; no capability -> "default"
[[ "$(printf '8.9\n' | parse_cuda_arch)" == "89" ]]      || { echo "FAIL parse_cuda_arch 8.9"; exit 1; }
[[ "$(printf 'no cap\n' | parse_cuda_arch)" == "default" ]] || { echo "FAIL parse_cuda_arch fallback"; exit 1; }

echo "PASS test_common"
