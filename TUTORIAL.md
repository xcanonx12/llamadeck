# llamadeck — quickstart & demo

A 5-minute tour, how to record the demo GIF, and a manual-QA checklist before
publishing.

## 1. Build

```bash
cd predictor && go build -o llamadeck ./cmd/llamadeck   # or: make build (from repo root)
export PATH="$PWD:$PATH"                                 # so `llamadeck` is on PATH
```

(Go lives at `~/.local/go` in this environment: `export PATH="$HOME/.local/go/bin:$PATH"`.)

## 2. Predict — does it fit? (no download, no GPU needed)

```bash
llamadeck unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --ctx 8192
```

You get the memory breakdown and a verdict (100% GPU / hybrid / CPU / OOM) plus
a suggested `-ngl`. Override hardware with `--vram-mb` / `--ram-mb` / `--gpus`.

## 3. Recommend — which quant should I pull?

```bash
llamadeck recommend unsloth/Llama-3.2-1B-Instruct-GGUF --ctx 8192
```

Scans every quant, prints size + max all-GPU context + verdict, and stars the
best pick.

## 4. The control center (full TUI)

```bash
llamadeck            # bare = the tabbed app
```

Tabs: **Models · Fit · Monitor · Config** (`tab`/`1`–`4` switch · `q` quit).

- **Models** — live top-50 GGUF repos (scroll) or `/` search · `↑↓` select ·
  `enter` shows metadata + quants + capabilities and a confirm · `enter` again
  jumps to **Fit** · `t` reload top list · `l` quick-launch with defaults
- **Fit** — live VRAM/RAM graph with an inline settings panel beneath it.
  `↑↓` pick a setting · `←→` adjust. Context / U-batch / KV-cache-type update
  the graph instantly; GPU layers default to **auto** (the predicted ngl), and
  a **GPU target** row pins to one device (predicting against that card's free
  VRAM). `c` opens the quant picker; `enter` → confirm → launch.
- **Monitor** — live GPU + containers · `s` stop · `x` remove · a starting
  server shows a spinner + its latest download/load log line
- **Config** — `b` build · `r` rebuild · `d` dev-fork URL · `k` set Hugging Face
  token (for gated/private models · fixes 401s)

## 5. Calibrate against a real launch (optional, needs GPU + Docker)

```bash
# after launching a model in a container named <name>
llamadeck calibrate unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --container <name> --ctx 8192
```

Sharpens future predictions (compute scale + per-GPU base overhead), saved to
`~/.config/llamadeck/profile.json`.

## 6. Record the demo GIF

Browser-free (no headless Chrome needed):

```bash
make demo            # python3 demo/gen_cast.py → agg → demo/llamadeck.gif
```

Needs `agg` and `python3` on PATH (and `ffmpeg`). `make demo-vhs` is the VHS
alternative (needs a display / Chromium).

---

## Manual QA before publishing

Things that can't be checked in a headless sandbox — run these on a real machine
and report anything that looks off:

### Core
- [ ] `make build` succeeds; `llamadeck version` prints a version.
- [ ] `llamadeck unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --ctx 8192` prints a
      sensible breakdown + verdict.
- [ ] `llamadeck recommend unsloth/Llama-3.2-1B-Instruct-GGUF` stars a reasonable quant.

### Interactive TUI (real terminal)
- [ ] `llamadeck` opens full-screen; tabs switch with `1`–`4`/`tab`; `q` quits cleanly.
- [ ] Colours render (green/cyan/mauve bars; coloured verdict badge).
- [ ] Models: the top-50 list loads and scrolls (`↑/↓ N more`); `/` search returns HF results.
- [ ] Models: `enter` shows metadata (arch, layers, VRAM, quants, caps) + a confirm;
      `enter` again jumps to the Fit tab with that model selected.
- [ ] Fit: `↑↓` moves through settings; on Context/U-batch/KV-cache `←→` changes
      the value and the bars/verdict update live.
- [ ] Fit: GPU layers shows `auto (N)` matching the suggested -ngl; `←→` overrides it.
- [ ] Fit (multi-GPU): the GPU-target row pins a single device; the graph then
      predicts against that card's free VRAM and the launch uses `--gpus device=N`.
- [ ] Fit: `c` lists quants with size/max-ctx/verdict; picking one updates the graph.
- [ ] Fit: `enter` shows the launch confirm; `y` starts a container (partial -ngl
      for a HYBRID model, not a forced full offload).
- [ ] Free VRAM on Fit drops when a server launches and recovers when it's removed.
- [ ] Monitor: GPU bars update (~2s); a launched server appears; `s`/`x` work.
- [ ] Monitor: a still-starting server shows a spinner + a live download/load log line.
- [ ] Config: `b` streams the build log live; image status flips to ✓ when done.
- [ ] Config: `k` stores a token; a gated model that 401'd now loads (or env `HF_TOKEN` works).

### Accuracy (with a GPU)
- [ ] Launch a model, run `llamadeck verify <model> --container <name> --ctx N`:
      KV ~0%, VRAM weights ~1%, compute slightly over (all "ok").
- [ ] `calibrate --container` reports a per-GPU base overhead and writes
      `~/.config/llamadeck/profile.json`.

### Degradation
- [ ] On a box with **no GPU**: prediction still works; CLI shows the "CPU/RAM only" note.
- [ ] With **Docker stopped**: the app shows "Docker not found"; prediction still works.
- [ ] On **macOS**: RAM is detected (not 0 → not everything-OOM).

### Docs / repo
- [ ] README renders on the forge; the demo GIF displays and loops.
- [ ] After creating the GitHub repo + first release: the `curl | sh` installer
      and `brew install` actually fetch a working binary (set the owner in
      `.goreleaser.yaml` + `install.sh` first).

### Nice-to-have (no DeepSeek GGUF was available to validate)
- [ ] Predict a DeepSeek-V2/V3 (MLA) model: it's tagged "MLA (KV approx)" and the
      KV number is small (not ~10× inflated). Confirm the verdict is sane.
