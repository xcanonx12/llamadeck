#!/usr/bin/env bash
# docker-build.sh runs under `set -u`: an .env written by an older version (no
# GCC_VERSION) used to abort with "GCC_VERSION: unbound variable" before docker
# was ever invoked. It must default the build args and fill them into .env.
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$PWD

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp -r lib Dockerfile dgx.Dockerfile docker-build.sh "$tmp/"
mkdir -p "$tmp/llama.cpp" && touch "$tmp/llama.cpp/CMakeLists.txt"  # skip the clone
printf 'UBUNTU_VERSION=22.04\nCUDA_VERSION=12.4.0\n' > "$tmp/.env"   # no GCC_VERSION

out=$(cd "$tmp" && ./docker-build.sh --no-build 2>&1) || { echo "FAIL: $out"; exit 1; }
grep -qx 'GCC_VERSION=12' "$tmp/.env" || { echo "FAIL: GCC_VERSION not defaulted for 22.04: $(cat "$tmp/.env")"; exit 1; }

# 24.04 gets gcc-14 (22.04 has no gcc-14 package, 24.04 has no gcc-12 default)
printf 'UBUNTU_VERSION=24.04\n' > "$tmp/.env"
(cd "$tmp" && ./docker-build.sh --no-build >/dev/null 2>&1) || { echo "FAIL 24.04 run"; exit 1; }
grep -qx 'GCC_VERSION=14' "$tmp/.env" || { echo "FAIL: GCC_VERSION not 14 for 24.04"; exit 1; }

# DOCKERFILE override is honored and a bogus one is rejected before building
out=$(cd "$tmp" && DOCKERFILE=nope.Dockerfile ./docker-build.sh --no-build 2>&1) \
  && { echo "FAIL: missing Dockerfile accepted"; exit 1; }
printf '%s' "$out" | grep -q "not found" || { echo "FAIL: unclear error: $out"; exit 1; }

cd "$repo"
echo "PASS test_build_env"
