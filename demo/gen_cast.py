#!/usr/bin/env python3
"""Synthesize an asciinema v2 cast for the llamadeck demo, then it's rendered by agg.
Captures the REAL colored output of each command (CLICOLOR_FORCE=1) and 'types'
the commands with timed events — deterministic, browser-free."""
import json, os, subprocess, sys

W, H = 102, 40
PROMPT = "\x1b[36m$\x1b[0m "      # cyan $
TYPE_DT = 0.028                    # seconds per typed char
AFTER_ENTER = 0.35                 # pause before output appears
SCENE_PAUSE = 3.5                  # pause after output

def run(cmd):
    env = dict(os.environ, CLICOLOR_FORCE="1")
    out = subprocess.run(cmd, capture_output=True, env=env).stdout.decode("utf-8", "replace")
    return out.replace("\r\n", "\n").replace("\n", "\r\n")  # raw terminal needs CR

def clear():
    return "\x1b[2J\x1b[3J\x1b[H"

SCENES = [
    ("llamadeck recommend unsloth/Llama-3.2-1B-Instruct-GGUF --vram-mb 2000 --ram-mb 16000",
     ["llamadeck", "recommend", "unsloth/Llama-3.2-1B-Instruct-GGUF", "--vram-mb", "2000", "--ram-mb", "16000"]),
    ("llamadeck bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M --ctx 32768 --vram-mb 8000 --ram-mb 32000",
     ["llamadeck", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "--ctx", "32768", "--vram-mb", "8000", "--ram-mb", "32000"]),
    ("llamadeck tui unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --ctx 8192 --vram-mb 8000 --ram-mb 32000",
     ["llamadeck", "tui", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", "--ctx", "8192", "--vram-mb", "8000", "--ram-mb", "32000"]),
]

events = []
t = 0.0
def emit(data):
    global t
    events.append([round(t, 3), "o", data])

emit(PROMPT)
for i, (shown, argv) in enumerate(SCENES):
    if i > 0:
        t += 0.6
        emit(clear() + PROMPT)
    for ch in shown:
        t += TYPE_DT
        emit(ch)
    t += 0.25
    emit("\r\n")
    t += AFTER_ENTER
    emit(run(argv))
    t += SCENE_PAUSE

header = {"version": 2, "width": W, "height": H, "title": "llamadeck"}
out_path = sys.argv[1]
with open(out_path, "w") as f:
    f.write(json.dumps(header) + "\n")
    for e in events:
        f.write(json.dumps(e) + "\n")
print(f"wrote {out_path}: {len(events)} events, ~{t:.1f}s")
