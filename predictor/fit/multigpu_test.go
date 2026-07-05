package fit

import "testing"

// Pooling GPUs makes more VRAM available, so more layers offload — even though
// each GPU adds its own base + compute overhead.
func TestMultiGPUPoolsVRAM(t *testing.T) {
	m := llama3ish() // ~8B, 32 layers
	c := DefaultConfig()
	c.Ctx = 8192

	one := mustPredict(t, m, Hardware{FreeVRAM: 5 * gib, FreeRAM: 64 * gib, NumGPUs: 1}, c)
	two := mustPredict(t, m, Hardware{FreeVRAM: 10 * gib, FreeRAM: 64 * gib, NumGPUs: 2}, c)

	if one.LayersOnGPU >= m.NLayers {
		t.Fatalf("precondition: 8B should not fully fit one 5 GiB GPU, got %d/%d", one.LayersOnGPU, m.NLayers)
	}
	if two.LayersOnGPU <= one.LayersOnGPU {
		t.Errorf("pooling 2 GPUs should offload more layers: 1 GPU=%d, 2 GPU=%d", one.LayersOnGPU, two.LayersOnGPU)
	}
}

// Each GPU pays its own CUDA context + compute buffer, so per-device overhead
// scales with the GPU count.
func TestMultiGPUOverheadScales(t *testing.T) {
	m := llama3ish()
	c := DefaultConfig()
	c.Ctx = 8192
	hw1 := Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib, NumGPUs: 1}
	hw2 := Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib, NumGPUs: 2}

	one := mustPredict(t, m, hw1, c)
	two := mustPredict(t, m, hw2, c)

	if one.Mode != ModeGPU || two.Mode != ModeGPU {
		t.Fatalf("both should fit fully on 24 GiB: %s / %s", one.Mode, two.Mode)
	}
	if two.BaseBytes != 2*one.BaseBytes {
		t.Errorf("base overhead should be 2× with 2 GPUs: %d vs %d", two.BaseBytes, one.BaseBytes)
	}
	if two.ComputeBytes != 2*one.ComputeBytes {
		t.Errorf("compute buffer should be 2× with 2 GPUs: %d vs %d", two.ComputeBytes, one.ComputeBytes)
	}
	if two.VRAMUsed <= one.VRAMUsed {
		t.Errorf("2-GPU VRAM use should exceed 1-GPU (extra overhead): %d vs %d", two.VRAMUsed, one.VRAMUsed)
	}
}

func mustPredict(t *testing.T, m *Model, hw Hardware, c Config) *Result {
	t.Helper()
	r, err := Predict(m, hw, c)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
