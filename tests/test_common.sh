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

# parse_cuda_arch: DGX Spark reports 12.1 (GB10) -> "121"
[[ "$(printf '12.1\n' | parse_cuda_arch)" == "121" ]] || { echo "FAIL parse_cuda_arch 12.1"; exit 1; }

# env_set adds a missing key and overwrites an existing one
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
(
  cd "$tmp"
  printf 'UBUNTU_VERSION=24.04\n' > .env
  env_set GCC_VERSION 14 >/dev/null
  env_set UBUNTU_VERSION 22.04 >/dev/null
  grep -qx 'GCC_VERSION=14' .env    || { echo "FAIL env_set add"; exit 1; }
  grep -qx 'UBUNTU_VERSION=22.04' .env || { echo "FAIL env_set overwrite"; exit 1; }
  [[ $(grep -c '^UBUNTU_VERSION=' .env) == 1 ]] || { echo "FAIL env_set duplicated key"; exit 1; }
) || exit 1

# is_unified_gpu: matches GB10 (DGX Spark), not a discrete card
fake=$tmp/bin; mkdir -p "$fake"
printf '#!/bin/sh\necho "%s"\n' 'NVIDIA GB10' > "$fake/nvidia-smi"; chmod +x "$fake/nvidia-smi"
PATH="$fake:$PATH" is_unified_gpu || { echo "FAIL is_unified_gpu GB10"; exit 1; }
printf '#!/bin/sh\necho "%s"\n' 'NVIDIA GeForce RTX 4090' > "$fake/nvidia-smi"
PATH="$fake:$PATH" is_unified_gpu && { echo "FAIL is_unified_gpu discrete"; exit 1; } || true

echo "PASS test_common"
