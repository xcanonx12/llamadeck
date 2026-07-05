#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source lib/common.sh

# With NO_COLOR (and captured stdout = not a TTY), no ANSI escape bytes are emitted.
out=$(NO_COLOR=1 bash -c 'source lib/common.sh; bold hello; dim x; _box "T"; hr 10')
grep -q $'\033' <<<"$out" && { echo "FAIL: ANSI emitted with NO_COLOR"; exit 1; }
grep -q 'hello' <<<"$out" || { echo "FAIL: bold dropped its text"; exit 1; }
grep -q 'T'     <<<"$out" || { echo "FAIL: _box dropped its title"; exit 1; }
grep -q '╭'     <<<"$out" || { echo "FAIL: _box did not draw a box"; exit 1; }

# fzf_select returns non-zero on empty input
if printf '' | fzf_select "x" >/dev/null 2>&1; then echo "FAIL: empty fzf_select should fail"; exit 1; fi

echo "PASS test_style"
