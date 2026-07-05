package fit

import "testing"

// smallModel is a minimal structurally-valid model for engine tests.
func smallModel() *Model {
	return &Model{
		Arch: "llama", NLayers: 16, HeadDim: 64, NKVHeads: 8, NHeads: 8,
		VocabSize: 32000, EmbedLength: 2048, FileBytes: 800 << 20,
	}
}

func TestPredict_NoGPU_IsPureCPU(t *testing.T) {
	c := DefaultConfig()
	c.Ctx = 8192
	hw := Hardware{FreeVRAM: 0, FreeRAM: 64 << 30, NumGPUs: 0}

	r, err := Predict(smallModel(), hw, c)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if r.Mode != ModeCPU {
		t.Errorf("Mode = %q, want %q", r.Mode, ModeCPU)
	}
	if r.VRAMUsed != 0 {
		t.Errorf("VRAMUsed = %d, want 0", r.VRAMUsed)
	}
	if r.LayersOnGPU != 0 {
		t.Errorf("LayersOnGPU = %d, want 0", r.LayersOnGPU)
	}
	// In CPU mode RAM is exactly the three host buckets.
	if want := r.CPUWeightBytes + r.CPUKVBytes + r.ComputeBytes; r.RAMUsed != want {
		t.Errorf("RAMUsed = %d, want weights+kv+compute = %d", r.RAMUsed, want)
	}
}

func TestPredict_NoGPU_OOMWhenRAMTooSmall(t *testing.T) {
	c := DefaultConfig()
	c.Ctx = 8192
	c.Mlock = true // pins weights → hard RAM requirement (no mmap paging escape)
	hw := Hardware{FreeVRAM: 0, FreeRAM: 64 << 20, NumGPUs: 0}

	r, err := Predict(smallModel(), hw, c)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if r.Mode != ModeOOM {
		t.Errorf("Mode = %q, want %q", r.Mode, ModeOOM)
	}
}
