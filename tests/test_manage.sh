#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Unknown command exits non-zero with usage.
if ./manage.sh bogus 2>/dev/null; then echo "FAIL: bogus cmd should exit non-zero"; exit 1; fi
# Library mode exposes the functions without running anything.
MANAGE_LIB=1 source ./manage.sh
declare -F list_containers >/dev/null || { echo "FAIL: list_containers missing"; exit 1; }
declare -F show_status >/dev/null     || { echo "FAIL: show_status missing"; exit 1; }
declare -F pick_container >/dev/null  || { echo "FAIL: pick_container missing"; exit 1; }
echo "PASS test_manage"
