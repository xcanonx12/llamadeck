package fit

// Golden test against the REAL Qwen3.5-9B-GGUF:Q4_K_S file when it's in the
// local HF cache (skipped elsewhere). This is the exact model whose launch
// crash-looped while the graph said "fits with room" — parsing the real header
// end-to-end must reproduce the OOM verdict the synthetic twin asserts.

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRealQwen35HeaderPredictsOOM(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, ".cache/huggingface/hub",
		"models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_S.gguf")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("real GGUF not cached locally: %v", err)
	}
	defer f.Close()
	st, _ := f.Stat()
	m, err := ParseGGUF(f, st.Size())
	if err != nil {
		t.Fatal(err)
	}

	if !m.IsRecurrent() || m.FullAttnInterval != 4 || m.SSMStateSize != 128 {
		t.Fatalf("hybrid keys misparsed from real header: %+v", m)
	}
	if m.Tensors == nil {
		t.Fatal("tensor table must parse from the real header")
	}

	c := DefaultConfig()
	c.Ctx = 32768
	c.UBatch = 1024
	c.KVType = "q4_0"
	c.NGL = 19
	hw := Hardware{GPUsFree: []int64{3451 * mib, 2633 * mib}, FreeRAM: 32 * gib}
	r, err := PredictDevices(m, hw, c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Fits {
		t.Fatalf("the real crash-loop config must predict OOM: %+v", r.Devices)
	}
	if r.Bottleneck != 1 {
		t.Fatalf("bottleneck = GPU%d, want GPU1", r.Bottleneck)
	}
	realGPU1 := int64(2718 * mib)
	if got := r.Devices[1].Used; got < realGPU1 {
		t.Fatalf("GPU1 predicted %s < real %s — over-promising", HumanBytes(got), HumanBytes(realGPU1))
	}
	t.Logf("GPU0 %s/%s %dL · GPU1 %s/%s %dL (main, over by %s)",
		HumanBytes(r.Devices[0].Used), HumanBytes(r.Devices[0].Free), r.Devices[0].Layers,
		HumanBytes(r.Devices[1].Used), HumanBytes(r.Devices[1].Free), r.Devices[1].Layers,
		HumanBytes(r.OverBy))
}
