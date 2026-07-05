package fit

import "testing"

// model mirroring Llama-3.2-1B (16 layers) with a tensor table so per-layer
// weights are exact. Sizes are illustrative, not byte-exact to the real GGUF.
func dualGPUModel() *Model {
	const n = 16
	per := make([]int64, n)
	for i := range per {
		per[i] = 40 * mib // ~640 MiB across layers
	}
	return &Model{
		Arch: "llama", NLayers: n, NHeads: 32, NKVHeads: 8,
		HeadDim: 64, EmbedLength: 2048, VocabSize: 128256, FileBytes: 770 * mib,
		Tensors: &TensorStats{
			Total: 763 * mib, Embedding: 205 * mib, Output: 100 * mib, PerLayer: per,
		},
	}
}

func TestPredictDevicesMainIsLastAndBalances(t *testing.T) {
	m := dualGPUModel()
	hw := Hardware{GPUsFree: []int64{9 * gib, 9 * gib}, FreeRAM: 32 * gib}
	r, err := PredictDevices(m, hw, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Devices) != 2 {
		t.Fatalf("got %d devices", len(r.Devices))
	}
	if !r.Devices[1].Main || r.Devices[0].Main {
		t.Fatalf("main should be the last device: %+v", r.Devices)
	}
	// The main device reserves its output+logits buffer first, so load-balancing
	// pushes layers toward the non-main device.
	if r.Devices[0].Layers < r.Devices[1].Layers {
		t.Fatalf("layers should pile on the non-main device: %d vs %d",
			r.Devices[0].Layers, r.Devices[1].Layers)
	}
	if r.Devices[0].Layers+r.Devices[1].Layers != m.NLayers {
		t.Fatalf("9+9 GiB should offload all %d layers, got %d+%d",
			m.NLayers, r.Devices[0].Layers, r.Devices[1].Layers)
	}
	if !r.Fits {
		t.Fatalf("9+9 GiB should fit a ~1 GiB model: %+v", r.Devices)
	}
}

func TestPredictDevicesPerDeviceOOM(t *testing.T) {
	m := dualGPUModel()
	// The pool fits, but the main (last) device is so tiny it can't even hold its
	// mandatory output+logits+base buffers → genuine per-device OOM the pooled
	// total hides. (A tiny NON-main device would just be left idle, not OOM.)
	hw := Hardware{GPUsFree: []int64{9 * gib, 300 * mib}, FreeRAM: 32 * gib}
	r, err := PredictDevices(m, hw, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if r.Fits {
		t.Fatalf("main device (300 MiB) can't hold its buffers → should OOM: %+v", r.Devices)
	}
	if r.Bottleneck != 1 {
		t.Fatalf("bottleneck = %d, want 1", r.Bottleneck)
	}
	if r.OverBy <= 0 {
		t.Fatalf("OverBy = %d, want >0", r.OverBy)
	}
}

func TestParseServerLogPerDevice(t *testing.T) {
	// Real dual-3080 snippet (from MULTIGPU-PLAN.md ground truth).
	log := `
llama_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 3080):   9875 total,    535 used,   2693 free vs. target of   1024
llama_params_fit_impl:   - CUDA1 (NVIDIA GeForce RTX 3080):   9875 total,    769 used,   1642 free vs. target of   1024
CUDA0 model buffer size = 343.94 MiB
CUDA1 model buffer size = 418.87 MiB
CUDA0 KV buffer size = 80.00 MiB
CUDA1 KV buffer size = 48.00 MiB
CUDA0 compute buffer size = 112.04 MiB
CUDA1 compute buffer size = 286.55 MiB
CPU_Mapped model buffer size = 205.49 MiB
offloaded 16/16 layers to GPU
`
	mm, err := ParseServerLog(log)
	if err != nil {
		t.Fatal(err)
	}
	if len(mm.Devices) != 2 {
		t.Fatalf("got %d devices, want 2: %+v", len(mm.Devices), mm.Devices)
	}
	d0, d1 := mm.Devices[0], mm.Devices[1]
	if d0.Name != "CUDA0" || d1.Name != "CUDA1" {
		t.Fatalf("device names/order wrong: %q %q", d0.Name, d1.Name)
	}
	// CUDA1 is the main device: bigger compute buffer.
	if d1.ComputeBytes <= d0.ComputeBytes {
		t.Fatalf("expected CUDA1 to carry the bigger compute buffer: %d vs %d", d1.ComputeBytes, d0.ComputeBytes)
	}
	// Capacity = projected used + projected remainder (the line's "free" alone
	// is the post-placement remainder, NOT what the device had available).
	if d0.FreeBytes != (535+2693)*mib || d1.FreeBytes != (769+1642)*mib {
		t.Fatalf("planning capacity parse: %d / %d", d0.FreeBytes, d1.FreeBytes)
	}
	// Host embedding copy stays out of the per-device (GPU) buckets.
	if mm.CPUModelBytes != 205*mib+unitToBytes(0.49, "mib") {
		t.Fatalf("CPU model bytes = %d", mm.CPUModelBytes)
	}
}

// moeModel: a small MoE where experts dominate each layer's weight.
func moeModel() *Model {
	const n = 8
	per := make([]int64, n)
	exp := make([]int64, n)
	for i := range per {
		exp[i] = 900 * mib // experts
		per[i] = exp[i] + 100*mib
	}
	return &Model{
		Arch: "qwen3moe", NLayers: n, NHeads: 32, NKVHeads: 8,
		HeadDim: 128, EmbedLength: 2048, VocabSize: 128256, FileBytes: 8 * gib,
		Tensors: &TensorStats{
			Total: 8 * gib, Embedding: 200 * mib, Output: 100 * mib,
			PerLayer: per, ExpertPerLayer: exp,
		},
	}
}

func TestPredictNCPUMoEShiftsExpertsToRAM(t *testing.T) {
	m := moeModel()
	hw := Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib}
	base, _ := Predict(m, hw, DefaultConfig())

	c := DefaultConfig()
	c.NCPUMoE = 4 // offload experts of the first 4 layers
	off, _ := Predict(m, hw, c)

	wantShift := int64(4 * 900 * mib)
	if got := base.VRAMUsed - off.VRAMUsed; got != wantShift {
		t.Fatalf("VRAM should drop by %s, dropped %s", HumanBytes(wantShift), HumanBytes(got))
	}
	if got := off.RAMUsed - base.RAMUsed; got != wantShift {
		t.Fatalf("RAM should rise by %s, rose %s", HumanBytes(wantShift), HumanBytes(got))
	}
}

func TestNCPUMoENoopOnDense(t *testing.T) {
	m := dualGPUModel() // no ExpertPerLayer set → dense
	hw := Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib}
	base, _ := Predict(m, hw, DefaultConfig())
	c := DefaultConfig()
	c.NCPUMoE = 8
	off, _ := Predict(m, hw, c)
	if base.VRAMUsed != off.VRAMUsed || base.RAMUsed != off.RAMUsed {
		t.Fatalf("n-cpu-moe must be a no-op on a dense model")
	}
}

func TestPredictHonorsExplicitNGL(t *testing.T) {
	m := dualGPUModel() // 16 layers, fits easily in 24 GiB
	hw := Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib}

	auto, _ := Predict(m, hw, DefaultConfig())
	if auto.LayersOnGPU != 16 {
		t.Fatalf("auto should offload all 16 layers, got %d", auto.LayersOnGPU)
	}

	c := DefaultConfig()
	c.NGL = 8 // force fewer layers than fit
	forced, _ := Predict(m, hw, c)
	if forced.LayersOnGPU != 8 {
		t.Fatalf("forced ngl=8, got %d layers", forced.LayersOnGPU)
	}
	if forced.VRAMUsed >= auto.VRAMUsed {
		t.Fatalf("fewer layers should use less VRAM: forced %d vs auto %d", forced.VRAMUsed, auto.VRAMUsed)
	}
	if forced.RAMUsed <= auto.RAMUsed {
		t.Fatalf("fewer GPU layers should spill more to RAM: forced %d vs auto %d", forced.RAMUsed, auto.RAMUsed)
	}
	if forced.RecommendedNGL != 16 {
		t.Fatalf("RecommendedNGL should stay the greedy best-fit (16), got %d", forced.RecommendedNGL)
	}
}

func TestIsMoE(t *testing.T) {
	if !moeModel().IsMoE() {
		t.Fatal("moeModel should report MoE")
	}
	if dualGPUModel().IsMoE() {
		t.Fatal("dense model should not report MoE")
	}
	if (&Model{}).IsMoE() {
		t.Fatal("model without a tensor table should not report MoE")
	}
}

// Regression for the "GPU all overestimate" bug: when only a few layers fit the
// pool, the per-device sim must offload only those, not force all NLayers on.
func TestPredictDevicesMatchesPooledLayerCount(t *testing.T) {
	m := dualGPUModel() // 16 layers, ~40 MiB each
	// Tiny per-device free: the pool fits only a handful of layers.
	hw := Hardware{GPUsFree: []int64{700 * mib, 600 * mib}, FreeRAM: 64 * gib}

	pooled, _ := Predict(m, Hardware{FreeVRAM: 1300 * mib, FreeRAM: 64 * gib, NumGPUs: 2}, DefaultConfig())
	dr, err := PredictDevices(m, hw, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, d := range dr.Devices {
		total += d.Layers
	}
	if total != pooled.LayersOnGPU {
		t.Fatalf("per-device layers %d must equal pooled fit %d (not %d full offload)",
			total, pooled.LayersOnGPU, m.NLayers)
	}
	if total >= m.NLayers {
		t.Fatalf("with only ~1.3 GiB free, must NOT offload all %d layers, got %d", m.NLayers, total)
	}
	// No device should be predicted to use wildly more than its free VRAM.
	for _, d := range dr.Devices {
		if d.Used > d.Free*3 {
			t.Fatalf("GPU%d used %s vs free %s — overestimate regression",
				d.Index, HumanBytes(d.Used), HumanBytes(d.Free))
		}
	}
}

// -ngl must change the per-device (--gpus all) prediction, not just single-GPU.
func TestPredictDevicesHonorsNGL(t *testing.T) {
	m := dualGPUModel()
	hw := Hardware{GPUsFree: []int64{12 * gib, 12 * gib}, FreeRAM: 64 * gib}
	auto, _ := PredictDevices(m, hw, DefaultConfig())
	autoTotal := auto.Devices[0].Layers + auto.Devices[1].Layers

	c := DefaultConfig()
	c.NGL = 4
	forced, _ := PredictDevices(m, hw, c)
	forcedTotal := forced.Devices[0].Layers + forced.Devices[1].Layers
	if forcedTotal != 4 {
		t.Fatalf("-ngl 4 should place 4 layers across devices, got %d", forcedTotal)
	}
	if forcedTotal >= autoTotal {
		t.Fatalf("-ngl 4 should offload fewer than auto (%d), got %d", autoTotal, forcedTotal)
	}
}

func TestTensorStatsScaledTo(t *testing.T) {
	ts := &TensorStats{Total: 1000, Embedding: 100, Output: 100,
		PerLayer: []int64{400, 400}, ExpertPerLayer: []int64{200, 200}}
	s := ts.ScaledTo(500) // half the size
	if s.Total != 500 || s.Embedding != 50 || s.PerLayer[0] != 200 || s.ExpertPerLayer[1] != 100 {
		t.Fatalf("scaled stats wrong: %+v", s)
	}
	if ts.Total != 1000 { // original untouched
		t.Fatal("ScaledTo must not mutate the original")
	}
}

func TestPredictDevicesNeedsTwo(t *testing.T) {
	m := dualGPUModel()
	if _, err := PredictDevices(m, Hardware{GPUsFree: []int64{9 * gib}}, DefaultConfig()); err == nil {
		t.Fatal("expected error with a single device")
	}
}
