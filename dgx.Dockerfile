# llama.cpp server for NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64).
# Differences from the x86 Dockerfile, all forced by the hardware:
#   - CUDA 13 base (sm_121 needs it) and the arm64/sbsa variant of the image.
#   - GGML_CPU_ALL_VARIANTS is x86-only; built natively for this Grace CPU instead.
#   - Unified memory: GPU and CPU share one LPDDR5X pool, so llama.cpp may
#     allocate past "VRAM" without a real copy — enabled at runtime below.
# Built by ./docker-build.sh (auto-selected on aarch64 + unified GPU), producing
# the same local/llama.cpp:server-cuda tag the rest of the toolkit expects.
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.1
ARG GCC_VERSION=14
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
ARG NODE_VERSION=24

# --- WebUI build stage (baked into llama-server at compile time) ---
FROM node:${NODE_VERSION} AS web
WORKDIR /app/tools/ui
COPY llama.cpp/tools/ui/package.json llama.cpp/tools/ui/package-lock.json ./
RUN npm ci
COPY llama.cpp/tools/ui/ ./
RUN npm run build

# --- CUDA build stage ---
FROM ${BASE_CUDA_DEV_CONTAINER} AS build
ARG GCC_VERSION
# GB10 is sm_121. docker-build.sh passes it from nvidia-smi's compute_cap.
ARG CUDA_DOCKER_ARCH=121
RUN apt-get update && apt-get install -y \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake \
    python3 python3-pip git libcurl4-openssl-dev libssl-dev libgomp1
ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}
WORKDIR /app
COPY llama.cpp/ .
COPY --from=web /app/tools/ui/dist tools/ui/dist
# "default" (no compute_cap from the driver) would build every arch — on ARM
# that is a very long build for no gain, so fall back to sm_121 (GB10) instead.
RUN if [ "${CUDA_DOCKER_ARCH}" = "default" ]; then ARCH=121; else ARCH="${CUDA_DOCKER_ARCH}"; fi && \
    cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j"$(nproc)"
RUN mkdir -p /app/lib && find build -name "*.so*" -exec cp -P {} /app/lib \;
RUN mkdir -p /app/full && cp build/bin/* /app/full

# --- runtime base ---
FROM ${BASE_CUDA_RUN_CONTAINER} AS base
RUN apt-get update && apt-get install -y libgomp1 curl ffmpeg \
    && apt autoremove -y && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete
COPY --from=build /app/lib/ /app

# --- server target ---
FROM base AS server
ENV LLAMA_ARG_HOST=0.0.0.0
# Same physical memory behind both budgets, so letting CUDA allocations fall
# back to host pages costs nothing here (it would mean PCIe thrash on a discrete
# GPU). Set to 0 to make over-allocation fail loudly instead.
ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
COPY --from=build /app/full/llama-server /app
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]
