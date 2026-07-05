# llamadeck (predictor core)

Predicts whether a GGUF model fits the local hardware — and in what mode
(100% GPU / Hybrid / 100% CPU / OOM) — **without downloading the model**. It
Range-reads only the GGUF header to get the structural keys, then runs a
layer-by-layer allocation against free VRAM/RAM.

![llamadeck — which quant fits, before you download](../demo/llamadeck.gif)

## Install

```bash
# from source (Go 1.24+)
make install                       # → ~/.local/bin/llamadeck
cd predictor && go install ./cmd/llamadeck
```

Prebuilt binaries (`curl … install.sh | sh`) and Homebrew
(`brew install xcanonx12/tap/llamadeck`) ship with the first tagged GitHub
release. Until one is published, install from source with `make install`.

Now a full operational TUI: `llamadeck` (no args) opens a tabbed, nvtop-style
control center — **Models · Fit · Monitor · Config** — built on the same engine.

## The app

```bash
llamadeck            # launch the full TUI
```

- **Models** — the live top-50 GGUF repos on the Hub (scrollable) or `/` search;
  `enter` shows the model's metadata (arch, layers, predicted VRAM, available
  quants, detected capabilities) and a confirm; confirm jumps you to **Fit**.
- **Fit** — the color-coded VRAM/RAM graph with an inline settings panel below it
  (`↑↓` field, `←→` adjust). Context / U-batch / KV-cache-type recompute the graph
  live; GPU layers default to the predicted **auto** ngl; a **GPU target** row
  pins a single device (predicting against that card's free VRAM and launching
  with `--gpus device=N`). Plus port, threads, n-cpu-moe, flash-attn, jinja,
  no-mmap, mlock. `c` is the **quant picker** (size · max-GPU-ctx · verdict);
  `enter` → confirm → launch.
- **Monitor** — live GPU memory/util bars and managed containers (stop/remove).
  A server that's still downloading/loading shows a spinner + its latest log line.
- **Config** — build / rebuild the server image (rm → update llama.cpp → build),
  a Dev option to build from an alternative llama.cpp fork URL, and `k` to store
  a Hugging Face access token (for gated/private models — fixes 401s).

First boot detects whether the server image exists and points you to the Config
tab if not. Predict/monitor/fit work anywhere; build/update need the repo. The
Hub token is read from `HF_TOKEN` or the saved Config-tab key.

## Layout

| Path | What |
|---|---|
| `fit/gguf.go` | GGUF header parser over any `io.ReaderAt` (local file or HTTP range) |
| `fit/engine.go` | Allocation engine: weights + KV + compute + base → fill VRAM, spill to RAM |
| `fit/calib.go` | Calibration: parse real llama-server buffer sizes, persist a compute correction |
| `fit/hf.go` | Hugging Face ref resolution (`owner/repo[:QUANT]` → GGUF shards) |
| `fit/*_test.go` | Offline tests: GGUF fixture, known-model math, log parser, verdict-flip, HF resolver |
| `fit/advise.go` | `MaxCtxForMode` solver — largest context that fits a target mode |
| `tui/view.go` | Pure `RenderView(State) string` — the striking color-coded graph |
| `tui/app.go` | Bubble Tea shell: live ←/→ context, ↑/↓ ubatch |
| `fit/accuracy_test.go` | Golden KV-footprint table for 6 real models — the regression net |
| `app/` | The full tabbed TUI: root + Models/Fit/Monitor/Config tabs, quant picker & launch form |
| `infra/` | Self-contained Docker + nvidia-smi: image check, containers, GPU, launch, build |
| `hub/` | Curated top models, Hugging Face search, and the shared model loader |
| `cmd/llamadeck/main.go` | CLI: bare = TUI; `predict`, `tui`, `recommend`, `calibrate`, `verify` |

## Recommend (the advisor)

Scans every quant in a repo and answers the real question — *which one should I
download?* For each quant it reports size, the largest all-GPU context it allows,
and the verdict at your requested context, then stars the best pick.

```bash
./llamadeck recommend unsloth/Llama-3.2-1B-Instruct-GGUF --ctx 8192
```

Structural keys are shared across quants, so it parses one header and only HEADs
each quant for its size. Rows sort by actual bytes (not name). The pick is the
largest quant that runs fully on GPU at the requested context, falling back to
the largest that avoids OOM.

## Interactive TUI

```bash
./llamadeck tui unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
```

Color-coded VRAM/RAM bars (weights · kv · compute · overhead · spill), a GPU-layer
gauge, and the verdict badge. `←/→` halves/doubles context, `↑/↓` ubatch — the
graph and verdict recompute instantly so you can find the settings that fit.
Piped / non-TTY, it prints a single static frame.

## Calibration (the loop)

llama-server prints its real buffer sizes at load. `calibrate` reads them,
compares to the prediction, and persists per-host corrections — the
compute-buffer scale, plus (with `--container`) the per-GPU base/CUDA-context
overhead — to `~/.config/llamadeck/profile.json`, applied automatically.

```bash
# From a saved log, stdin, or a running container
./llamadeck calibrate model.gguf --log server.log --ctx 8192
./llamadeck calibrate model.gguf --container llama-... --ctx 8192
docker logs my-server 2>&1 | ./llamadeck calibrate model.gguf --log - --ctx 8192
```

Weights and KV are validated as exact against real launches; the compute buffer
(the OOM-causing fuzzy bucket) is the value that gets corrected.

## Accuracy & trust

`verify` reports prediction error against a real launch *without* persisting
anything — the "can I trust this on my box?" check:

```bash
./llamadeck verify unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --container <name> --ctx 8192
```

It prints predicted-vs-real per bucket and flags any over 15%. In practice
weights and KV come out exact; only the compute buffer needs calibration.
`fit/accuracy_test.go` pins KV footprints for six real architectures (diverse
GQA ratios, Gemma's 256 head-dim) so a math regression fails CI, not users.

## Build & test

```bash
export PATH="$HOME/.local/go/bin:$PATH"   # Go installed here this session
go test ./...
go build -o llamadeck ./cmd/llamadeck
```

## Use

```bash
# By Hugging Face ref (resolves the GGUF, sums shards, reads header only)
./llamadeck unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --ctx 8192
./llamadeck unsloth/Llama-3.2-1B-Instruct-GGUF --ctx 8192   # auto-picks a quant

# By direct URL or local file
./llamadeck https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf --ctx 8192
./llamadeck model.gguf --ctx 8192

# Override hardware (0 = probe nvidia-smi / /proc/meminfo)
./llamadeck unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --vram-mb 8000 --ram-mb 32000
```

Source forms accepted everywhere (predict and calibrate): `owner/repo[:QUANT]`,
a direct `.gguf` URL, or a local path.

## Known limitations (next-step backlog)

- **Single global compute scale.** One correction across all models (max ratio).
  Per-arch or regression fitting is a later refinement.
- **Uniform per-layer weight.** Token-embedding / output layers are heavier in
  reality; not yet split out.
- **Auto-pick quant is naive.** With no `:QUANT`, it takes the first sorted tag
  (often BF16, the heaviest). Prefer a sensible default like Q4_K_M.
- **Header read from shard 0.** Multi-shard *weights* are summed (HF path), but
  the header is read from the first shard only; direct-URL/local sharded models
  still size to one file.
- **Single fixed 32 MiB header window** for remote reads. Grow-on-demand if a
  huge-tokenizer model ever truncates.
- **Multi-GPU is pooled.** VRAM is summed across GPUs with per-device base +
  compute overhead (conservative); exact per-device layer placement isn't modelled.
  Use `--gpus N` to model a specific count.
- **MLA models** (DeepSeek-V2/V3) are **detected** and use the compressed-latent
  KV formula, but it's approximate and flagged — verify against a real launch.

Calibrated by `calibrate` (persisted to the profile, applied automatically): the
compute-buffer scale **and** the per-GPU base/CUDA-context overhead (measured as
process VRAM minus the reported buffers).
