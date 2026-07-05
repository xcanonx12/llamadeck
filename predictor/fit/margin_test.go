package fit

// Load-margin, TIGHT-state, and SafeMaxNGL tests. Ground truth: the dual-3080
// residual campaign (predictor/VALIDATION.md) — llama-server holds ~250 MiB/GPU
// (dense) to ~690 MiB/GPU (hybrid) beyond its logged buffers, and a hybrid AUTO
// launch (-ngl 999) crash-looped even though the raw fit said it fit.

import "testing"

func TestLoadMarginResolution(t *testing.T) {
	c := DefaultConfig()
	if got := loadMargin(c); got != DefaultLoadMargin {
		t.Fatalf("zero LoadMargin should resolve to the default, got %d", got)
	}
	c.LoadMargin = 100 * mib
	if got := loadMargin(c); got != 100*mib {
		t.Fatalf("explicit margin ignored: %d", got)
	}
	c.LoadMargin = -1
	if got := loadMargin(c); got != 0 {
		t.Fatalf("negative margin should disable (0), got %d", got)
	}
}

func TestAutoFitReservesMargin(t *testing.T) {
	m := dualGPUModel()
	hw := Hardware{FreeVRAM: 4 * gib, FreeRAM: 64 * gib}
	c := DefaultConfig()
	raw := c
	raw.LoadMargin = -1 // disabled
	rRaw, _ := Predict(m, hw, raw)
	rDef, _ := Predict(m, hw, c)
	if rDef.RecommendedNGL > rRaw.RecommendedNGL {
		t.Fatalf("margin-aware auto fit must not offload more than raw: %d vs %d",
			rDef.RecommendedNGL, rRaw.RecommendedNGL)
	}
	if rDef.Tight {
		t.Fatal("auto fit reserves the margin — it must not report Tight on a dense model")
	}
	// The reserve must actually bite when VRAM is exactly at the raw edge.
	edge := Hardware{FreeVRAM: rRaw.VRAMUsed + 100*mib, FreeRAM: 64 * gib}
	rEdge, _ := Predict(m, edge, c)
	if rEdge.RecommendedNGL >= rRaw.RecommendedNGL && rRaw.RecommendedNGL == m.NLayers {
		t.Fatalf("at the raw edge the margin should reduce the fit: %d vs %d",
			rEdge.RecommendedNGL, rRaw.RecommendedNGL)
	}
}

func TestExplicitNGLTightAndOOM(t *testing.T) {
	m := dualGPUModel()
	c := DefaultConfig()
	c.NGL = 16 // all layers, explicitly

	// Roomy: fits with margin to spare — neither tight nor OOM.
	roomy, _ := Predict(m, Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib}, c)
	if roomy.Tight || roomy.Mode == ModeOOM {
		t.Fatalf("roomy explicit fit misflagged: tight=%v mode=%s", roomy.Tight, roomy.Mode)
	}

	// Tight: fits on paper but inside the margin.
	tight, _ := Predict(m, Hardware{FreeVRAM: roomy.VRAMUsed + 100*mib, FreeRAM: 64 * gib}, c)
	if !tight.Tight {
		t.Fatalf("explicit fit with 100 MiB headroom must be Tight (margin %s)", HumanBytes(DefaultLoadMargin))
	}
	if tight.Mode == ModeOOM {
		t.Fatal("tight is a warning, not OOM")
	}

	// Hard over-subscription: OOM.
	oom, _ := Predict(m, Hardware{FreeVRAM: roomy.VRAMUsed - 500*mib, FreeRAM: 64 * gib}, c)
	if oom.Mode != ModeOOM {
		t.Fatalf("explicit over-subscription must be OOM, got %s", oom.Mode)
	}
}

// The measured auto-hybrid crash (campaign run C): a spilling hybrid on auto
// must carry the Tight warning — llama's own auto-fit under-reserves for
// recurrent models.
func TestAutoHybridSpillIsTight(t *testing.T) {
	m := qwen35ish()
	c := DefaultConfig()
	c.Ctx = 8192
	c.KVType = "q4_0"
	hw := Hardware{GPUsFree: []int64{3451 * mib, 2633 * mib}, FreeRAM: 26 * gib}
	dr, err := PredictDevices(m, hw, c)
	if err != nil {
		t.Fatal(err)
	}
	if !dr.Tight {
		t.Fatal("spilling hybrid on auto must be flagged Tight (real launch crash-looped)")
	}
	// A dense model spilling on auto stays un-flagged (dense auto launch ran fine).
	d, _ := Predict(dualGPUModel(), Hardware{FreeVRAM: 700 * mib, FreeRAM: 64 * gib}, DefaultConfig())
	if d.Tight {
		t.Fatal("spilling dense auto fit must not be Tight")
	}
	// A hybrid that fully fits on GPU isn't flagged either.
	full, _ := Predict(m, Hardware{FreeVRAM: 64 * gib, FreeRAM: 64 * gib}, c)
	if full.Tight {
		t.Fatal("fully-offloaded hybrid must not be Tight")
	}
}

// The per-device composition buckets must sum exactly to Used in both split
// regimes — the graph renders them as segments, so a drift would lie visually.
func TestDeviceBreakdownSumsToUsed(t *testing.T) {
	m := qwen35ish()
	hw := Hardware{GPUsFree: []int64{3451 * mib, 2633 * mib}, FreeRAM: 26 * gib}
	for _, ngl := range []int{0, 19} {
		c := DefaultConfig()
		c.Ctx = 8192
		c.KVType = "q4_0"
		c.NGL = ngl
		dr, err := PredictDevices(m, hw, c)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range dr.Devices {
			sum := d.WeightBytes + d.KVBytes + d.ComputeBytes + d.BaseBytes
			if sum != d.Used {
				t.Fatalf("ngl=%d GPU%d buckets sum %d != Used %d", ngl, d.Index, sum, d.Used)
			}
			if d.Layers > 0 && (d.WeightBytes == 0 || d.KVBytes == 0) {
				t.Fatalf("ngl=%d GPU%d holds %d layers but has empty buckets: %+v", ngl, d.Index, d.Layers, d)
			}
		}
	}
}

// Quantized KV must include the ggml per-block scale overhead (block = 32
// elems: q4_0 = 18 B, q8_0 = 34 B). Real launch: 36 MiB per attention layer at
// ctx 32768 q4_0 — the old 0.5 B/elem predicted 32.
func TestQuantizedKVIncludesBlockOverhead(t *testing.T) {
	m := qwen35ish() // 4 KV heads × head_dim 256
	c := DefaultConfig()
	c.Ctx = 32768
	c.KVType = "q4_0"
	if got := kvPerLayer(m, c); got != 36*mib {
		t.Fatalf("q4_0 KV/layer = %s, want 36 MiB (measured)", HumanBytes(got))
	}
	c.KVType = "q8_0"
	if got := kvPerLayer(m, c); got != 68*mib {
		t.Fatalf("q8_0 KV/layer = %s, want 68 MiB (34 B / 32-elem block)", HumanBytes(got))
	}
}

// RAM accounting pinned to a REAL launch (Qwen3.5-9B Q4_K_S, ctx 8192, q4_0,
// -ngl 12, default 4 seqs): CPU_Mapped model 3045.05 MiB + CPU KV 45 MiB +
// CPU RS 134 MiB = 3224 MiB host-side. Our model puts one block fewer on the
// CPU (llama counts the output head as an -ngl layer — deliberate), so allow
// one block (~126 MiB) of slack, no more.
func TestRAMAccountingMatchesRealLaunch(t *testing.T) {
	m := qwen35ish()
	c := DefaultConfig()
	c.Ctx = 8192
	c.KVType = "q4_0"
	c.NGL = 12
	r, err := Predict(m, Hardware{FreeVRAM: 6 * gib, FreeRAM: 26 * gib}, c)
	if err != nil {
		t.Fatal(err)
	}
	real := int64(3224 * mib) // 3045.05 model + 45 KV + 134 RS
	diff := real - r.RAMUsed
	if diff < 0 {
		diff = -diff
	}
	if diff > 150*mib {
		t.Fatalf("RAMUsed %s vs real host-side %s — off by %s (>1 block)",
			HumanBytes(r.RAMUsed), HumanBytes(real), HumanBytes(diff))
	}
}

// Accounting identities that must hold for ANY knob combination — every byte
// of the model lands exactly once (embedding twice, by design: llama keeps a
// host copy AND ships it to the GPU when offloading).
func TestAccountingIdentitiesAcrossKnobs(t *testing.T) {
	hw := Hardware{FreeVRAM: 6 * gib, FreeRAM: 26 * gib}
	for _, m := range []*Model{qwen35ish(), dualGPUModel(), moeModel()} {
		for _, ngl := range []int{0, 5, 12} {
			for _, seqs := range []int{0, 1, 8} {
				for _, moe := range []int{0, 4} {
					c := DefaultConfig()
					c.Ctx = 8192
					c.KVType = "q4_0"
					c.NGL = ngl
					c.NSeqs = seqs
					c.NCPUMoE = moe
					r, err := Predict(m, hw, c)
					if err != nil {
						t.Fatal(err)
					}
					if r.LayersOnGPU == 0 {
						continue // CPU path uses its own single-sided sums
					}
					ts := m.Tensors
					// Untied: embed host-only, output rides the GPU. Tied (no
					// output.weight): the embedding doubles as the GPU head,
					// so it counts on both sides.
					wantW := ts.Embedding + ts.Output
					if ts.Output == 0 {
						wantW += ts.Embedding
					}
					for _, w := range ts.PerLayer {
						wantW += w
					}
					if got := r.GPUWeightBytes + r.CPUWeightBytes; got != wantW {
						t.Fatalf("%s ngl=%d seqs=%d moe=%d: weights %d split to %d — bytes lost",
							m.Arch, ngl, seqs, moe, wantW, got)
					}
					if got := r.GPUKVBytes + r.CPUKVBytes; got != r.KVBytes {
						t.Fatalf("%s ngl=%d seqs=%d moe=%d: KV %d split to %d",
							m.Arch, ngl, seqs, moe, r.KVBytes, got)
					}
				}
			}
		}
	}
}

func TestSafeMaxNGL(t *testing.T) {
	m := qwen35ish()
	c := failingConfig() // the real crash config: ctx 32768, ubatch 1024, q4_0, -ngl 19
	hw := Hardware{GPUsFree: []int64{3451 * mib, 2633 * mib}, FreeRAM: 26 * gib}
	safe := SafeMaxNGL(m, hw, c)
	if safe >= 19 {
		t.Fatalf("crash config launched with -ngl 19; SafeMaxNGL must be lower, got %d", safe)
	}
	if safe > 0 {
		cc := c
		cc.NGL = safe
		dr, err := PredictDevices(m, hw, cc)
		if err != nil {
			t.Fatal(err)
		}
		if !dr.Fits || dr.Tight {
			t.Fatalf("SafeMaxNGL=%d must itself predict a comfortable fit: %+v", safe, dr)
		}
	}

	// Dense on a huge GPU: every layer is safe.
	if got := SafeMaxNGL(dualGPUModel(), Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib}, DefaultConfig()); got != 16 {
		t.Fatalf("roomy dense SafeMaxNGL = %d, want 16", got)
	}
	// No VRAM at all: nothing is safe.
	if got := SafeMaxNGL(dualGPUModel(), Hardware{FreeVRAM: 200 * mib, FreeRAM: 64 * gib}, DefaultConfig()); got != 0 {
		t.Fatalf("tiny VRAM SafeMaxNGL = %d, want 0", got)
	}
}
