# Accuracy validation

llamadeck's predictions are validated against **real `llama-server` launches**, not
just internal math. This is the basis for trusting a "it fits" verdict.

## Method

Each model was launched in the `local/llama.cpp:server-cuda` image pinned to a
**single** RTX 3080 (`--gpus device=0`), fully offloaded (`-ngl 999`). The real
buffer sizes `llama-server` prints at load (`model buffer size`, `KV buffer
size`, `compute buffer size`) were compared to the prediction via
`llamadeck verify <model> --container <name> --ctx <ctx>`.

Reproduce:

```bash
docker run -d --name val --gpus '"device=0"' -p 7991:7991 \
  -v "$PWD/.models:/root/.cache/huggingface" local/llama.cpp:server-cuda \
  -hf unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 7991 --ctx-size 8192 -ngl 999
llamadeck verify unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --container val --ctx 8192
```

## Results (single RTX 3080, ctx 8192, Q4_K_M)

| Model | KV cache | VRAM weights | Compute buffer |
|---|---|---|---|
| Llama-3.2-1B | 256 / 256 MiB · **+0.0%** | 770 / 763 MiB · **+1.0%** | 322 / 303 MiB · **+6.6%** |
| Llama-3.2-3B | 896 / 896 MiB · **+0.0%** | 1.88 / 1.87 GiB · **+0.4%** | 326 / 291 MiB · **+12.4%** |

(predicted / real · error)

## Findings

- **KV cache is exact.** It's pure structural math; the golden test in
  `fit/accuracy_test.go` pins it across six architectures.
- **VRAM weights are within ~1%.** Predicted file size ≈ the GPU-resident model
  buffer.
- **The compute buffer is over-predicted by ~6–12%** — deliberately conservative,
  so the tool errs toward predicting OOM early rather than promising a fit that
  crashes. `calibrate` can tighten it per-host.
- **Host model buffer:** llama.cpp keeps the token-embedding / output tensors on
  the host (~205 MiB for 1B, ~308 MiB for 3B). This counts toward RAM, not VRAM,
  so it doesn't affect the GPU-fit verdict; it is a small, currently-unmodeled
  RAM-side cost (see Roadmap: base/host-buffer modeling).

## Conclusion

For the question that matters — *will this fit on my GPU?* — predictions are
accurate (KV exact, weights ~1%) and conservatively biased on the one estimated
bucket. A green verdict is trustworthy.

## Load-margin campaign (dual RTX 3080, 2026-07-01)

The buffer log alone under-states what `llama-server` really holds: comparing
each probe container's **process VRAM** (`nvidia-smi --query-compute-apps`)
against the sum of its logged buffers exposes the load-time extras (CUDA
context, `alloc_compute_meta`, scheduler reserves). Six probes (dense
Llama-3.2-1B BF16 + hybrid Qwen3.5-9B Q4_K_S; auto and explicit `-ngl`; ubatch
512–2048; f16/q4_0 KV) with free VRAM 3451 / 2633 MiB:

| Run | Config | Outcome | Worst per-device residual (real bufs − pred) | Process VRAM − logged buffers |
|---|---|---|---|---|
| A | Llama auto (999), ctx 8k | ran | +494 MiB (GPU1, auto split noise) | 494 MiB (~250/GPU) |
| B | Llama -ngl 10, ctx 32k | ran | +2 MiB | 494 MiB (~250/GPU) |
| C | Qwen auto (999), ctx 8k | **crashed (139)** | +249 MiB | n/a (died in graph_reserve) |
| D | Qwen -ngl 12, ctx 8k | ran | +43 MiB | **1.37 GiB (~690/GPU)** |
| E | Qwen -ngl 14, ub 1024 | ran (near edge) | +268 MiB | 1.38 GiB (~690/GPU) |
| F | Qwen -ngl 10, ub 2048, ctx 16k | **crashed (139)** | — | — (predicted OOM, over by 1.02 GiB ✓) |

Findings, encoded in the engine:

- **`DefaultLoadMargin = 512 MiB`** — covers the observed residual tail
  (worst near-edge under-prediction ≈ 250–500 MiB/device) without flagging
  roomy fits. The auto fit *reserves* it (mirroring `llama_params_fit`'s own
  free-VRAM target); explicit `-ngl` configs inside it are flagged **TIGHT**.
- **Run C is the key result: auto is not inherently safe on hybrids.**
  llama.cpp's own auto-fit under-reserves for recurrent models — a spilling
  hybrid on auto crash-looped while the raw fit said "fits". A spilling hybrid
  on auto is therefore always flagged TIGHT.
- Hybrid/big-vocab models hold ~690 MiB/GPU beyond their logged buffers
  (vs ~250 MiB/GPU dense) — `calibrate` ratchets the per-host margin
  (`Profile.LoadMargin`) upward when it observes worse.
