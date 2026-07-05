#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
f=Dockerfile
grep -q 'AS web' "$f"            || { echo "FAIL: missing node web stage"; exit 1; }
grep -q 'AS server' "$f"         || { echo "FAIL: missing server target"; exit 1; }
grep -qF 'COPY llama.cpp/ .'     "$f" || { echo "FAIL: must copy cloned subdir"; exit 1; }
grep -q 'tools/ui/dist'          "$f" || { echo "FAIL: WebUI dist not wired in"; exit 1; }
grep -q 'gcc-${GCC_VERSION}'     "$f" || { echo "FAIL: GCC pin missing"; exit 1; }
grep -q 'ffmpeg'                 "$f" || { echo "FAIL: ffmpeg missing"; exit 1; }
grep -q 'libssl-dev'             "$f" || { echo "FAIL: libssl-dev missing (needed for -hf HTTPS / OpenSSL)"; exit 1; }
grep -q 'ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}' "$f" || { echo "FAIL: CC/CXX/CUDAHOSTCXX pin missing"; exit 1; }
grep -q 'HEALTHCHECK'            "$f" || { echo "FAIL: healthcheck missing"; exit 1; }
echo "PASS test_dockerfile"
