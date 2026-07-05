#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CHECK_HW_LIB=1 source ./check-hardware.sh
declare -F check_docker >/dev/null  || { echo "FAIL no check_docker"; exit 1; }
declare -F check_toolkit >/dev/null || { echo "FAIL no check_toolkit"; exit 1; }
declare -F check_driver >/dev/null  || { echo "FAIL no check_driver"; exit 1; }
declare -F ensure_pkg >/dev/null    || { echo "FAIL no ensure_pkg"; exit 1; }
echo "PASS test_hardware"
