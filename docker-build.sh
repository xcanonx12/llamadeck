#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source lib/common.sh

header "build"

[[ -f .env ]] || { cp .env.example .env; log_info "Created .env from .env.example"; }
load_env

# 1. Pick CUDA tag from the driver and persist it.
env_set CUDA_VERSION "$(detect_cuda_tag)"
load_env

# Build args every Dockerfile here expects. Defaulted (not required in .env), so
# an .env written by an older version — or hand-trimmed — can't fail the build
# under `set -u` with "GCC_VERSION: unbound variable".
: "${UBUNTU_VERSION:=24.04}"
: "${GCC_VERSION:=$([[ "${UBUNTU_VERSION}" == 22.04 ]] && echo 12 || echo 14)}"  # newest gcc the base image packages
env_set UBUNTU_VERSION "${UBUNTU_VERSION}"
env_set GCC_VERSION "${GCC_VERSION}"
load_env

# 1b. DGX Spark / any unified-memory ARM64 part: different base image, no x86
# CPU variants, sm_121. Its own Dockerfile; same output tag so the rest of the
# toolkit is unchanged. Override with DOCKERFILE=... to force one.
if [[ -z "${DOCKERFILE:-}" ]]; then
  if [[ "$(uname -m)" == "aarch64" ]] && is_unified_gpu; then
    DOCKERFILE=dgx.Dockerfile
  else
    DOCKERFILE=Dockerfile
  fi
fi
[[ -f "$DOCKERFILE" ]] || die "Dockerfile '$DOCKERFILE' not found"

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
ARCH=$(detect_cuda_arch)
log_info "Building local/llama.cpp:server-cuda from ${DOCKERFILE} (this takes a while)"
log_info "ubuntu=${UBUNTU_VERSION} cuda=${CUDA_VERSION} gcc=${GCC_VERSION} arch=${ARCH}"
docker build -f "$DOCKERFILE" \
  --target server \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg GCC_VERSION="${GCC_VERSION}" \
  --build-arg CUDA_DOCKER_ARCH="${ARCH}" \
  -t local/llama.cpp:server-cuda .

log_info "Done. Launch a model with $(bold ./menu.sh) or $(bold ./launch.sh)"
