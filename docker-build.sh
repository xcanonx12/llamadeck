#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source lib/common.sh

header "build"

[[ -f .env ]] || { cp .env.example .env; log_info "Created .env from .env.example"; }
load_env

# 1. Pick CUDA tag from the driver and persist it.
TAG=$(detect_cuda_tag)
if grep -q '^CUDA_VERSION=' .env; then
  sed -i.bak -E "s/^CUDA_VERSION=.*/CUDA_VERSION=${TAG}/" .env
else
  echo "CUDA_VERSION=${TAG}" >> .env
fi
log_info "CUDA_VERSION set to ${TAG} in .env"
load_env

# 2. Clone llama.cpp if missing or incomplete.
if [[ ! -f llama.cpp/CMakeLists.txt ]]; then
  [[ -d llama.cpp ]] && { log_warn "Removing incomplete llama.cpp checkout"; rm -rf llama.cpp; }
  log_info "Cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
else
  log_info "llama.cpp present — skipping clone"
fi

# 3. Model cache dir.
mkdir -p "${MODELS_DIR:-./.models}"

[[ "${1:-}" == "--no-build" ]] && { log_info "--no-build: stopping before docker build"; exit 0; }

# 4. Build the server image.
log_info "Building local/llama.cpp:server-cuda (this takes a while)"
ARCH=$(detect_cuda_arch)
log_info "Target CUDA arch: ${ARCH}"
docker build \
  --target server \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg GCC_VERSION="${GCC_VERSION}" \
  --build-arg CUDA_DOCKER_ARCH="${ARCH}" \
  -t local/llama.cpp:server-cuda .

log_info "Done. Launch a model with $(bold ./menu.sh) or $(bold ./launch.sh)"
