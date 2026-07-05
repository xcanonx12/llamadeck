package fit

import "testing"

func TestTensorBytes(t *testing.T) {
	cases := []struct {
		typ   uint32
		nelem int64
		want  int64
		ok    bool
	}{
		{1, 256, 512, true},  // F16: 2 B/elem
		{12, 256, 144, true}, // Q4_K: 144 B / 256-block
		{14, 256, 210, true}, // Q6_K
		{8, 32, 34, true},    // Q8_0
		{0, 10, 40, true},    // F32
		{999, 256, 0, false}, // unknown type
	}
	for _, c := range cases {
		got, ok := tensorBytes(c.nelem, c.typ)
		if ok != c.ok || got != c.want {
			t.Errorf("type %d nelem %d: got (%d,%v) want (%d,%v)", c.typ, c.nelem, got, ok, c.want, c.ok)
		}
	}
}

func TestClassifyTensor(t *testing.T) {
	ts := &TensorStats{PerLayer: make([]int64, 4), ExpertPerLayer: make([]int64, 4)}
	classifyTensor(ts, "token_embd.weight", 100, 4)
	classifyTensor(ts, "output.weight", 50, 4)
	classifyTensor(ts, "blk.1.attn_q.weight", 20, 4)
	classifyTensor(ts, "blk.2.ffn_gate_exps.weight", 30, 4)
	classifyTensor(ts, "blk.9.attn_q.weight", 99, 4) // out of range → ignored
	if ts.Embedding != 100 || ts.Output != 50 {
		t.Errorf("embed/output = %d/%d, want 100/50", ts.Embedding, ts.Output)
	}
	if ts.PerLayer[1] != 20 || ts.PerLayer[2] != 30 {
		t.Errorf("perLayer = %v", ts.PerLayer)
	}
	if ts.ExpertPerLayer[2] != 30 || ts.ExpertPerLayer[1] != 0 {
		t.Errorf("expertPerLayer = %v, want [_,0,30,_]", ts.ExpertPerLayer)
	}
}

// TestEngineUsesExactTensors checks the embedding is charged to host RAM (with a
// GPU copy in the weights) rather than reducing VRAM.
func TestEngineUsesExactTensors(t *testing.T) {
	m := &Model{
		Arch: "llama", NLayers: 4, NHeads: 8, NKVHeads: 8, HeadDim: 64,
		EmbedLength: 512, VocabSize: 32000, FileBytes: 1000 * mib,
		Tensors: &TensorStats{
			Total:     1000 * mib,
			Embedding: 200 * mib,
			Output:    0,
			PerLayer:  []int64{200 * mib, 200 * mib, 200 * mib, 200 * mib},
		},
	}
	hw := Hardware{FreeVRAM: 8 * gib, FreeRAM: 32 * gib, NumGPUs: 1}
	r, err := Predict(m, hw, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if r.Mode != ModeGPU || r.LayersOnGPU != 4 {
		t.Fatalf("expected full GPU, got %s ngl %d", r.Mode, r.LayersOnGPU)
	}
	// Embedding (200 MiB) should appear in host RAM even though it's fully on GPU.
	if r.RAMUsed < 200*mib {
		t.Errorf("RAMUsed %s should include the host embedding copy (>=200 MiB)", HumanBytes(r.RAMUsed))
	}
}
