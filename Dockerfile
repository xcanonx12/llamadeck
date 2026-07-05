ARG UBUNTU_VERSION=24.04
# Must generally match the host's max supported CUDA (see docker-build.sh).
ARG CUDA_VERSION=12.9.0
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
ARG CUDA_DOCKER_ARCH=default
RUN apt-get update && apt-get install -y \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake \
    python3 python3-pip git libcurl4-openssl-dev libssl-dev libgomp1
ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}
WORKDIR /app
COPY llama.cpp/ .
COPY --from=web /app/tools/ui/dist tools/ui/dist
RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
      export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} \
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
COPY --from=build /app/full/llama-server /app
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]
