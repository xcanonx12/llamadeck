package fit

import "testing"

// Golden accuracy table: real model architectures with KV-cache footprints
// computed BY HAND (independently of the engine) at a fixed context, so this is
// a true regression net, not a tautology. KV is the dynamic, OOM-driving bucket;
// pinning it across diverse GQA ratios and head dims guards the core math.
//
// KV total = 2 (K+V) * n_kv_heads * head_dim * ctx * 2 bytes (f16) * n_layers.
// Structural params are public (config.json / GGUF metadata of each model).
func TestKVAccuracyGolden(t *testing.T) {
	const ctx = 4096
	const mib = 1 << 20

	cases := []struct {
		name      string
		m         Model
		wantKVMiB int64 // hand-computed at ctx=4096, f16
	}{
		// 2*8*64*4096*2*16  = 134,217,728 = 128 MiB
		{"Llama-3.2-1B", Model{NLayers: 16, NHeads: 32, NKVHeads: 8, EmbedLength: 2048, HeadDim: 64, VocabSize: 128256}, 128},
		// 2*8*128*4096*2*32 = 536,870,912 = 512 MiB
		{"Llama-3.1-8B", Model{NLayers: 32, NHeads: 32, NKVHeads: 8, EmbedLength: 4096, HeadDim: 128, VocabSize: 128256}, 512},
		// 2*4*128*4096*2*28 = 234,881,024 = 224 MiB  (GQA 4 kv heads)
		{"Qwen2.5-7B", Model{NLayers: 28, NHeads: 28, NKVHeads: 4, EmbedLength: 3584, HeadDim: 128, VocabSize: 152064}, 224},
		// 2*8*128*4096*2*32 = 512 MiB  (vocab differs, KV identical to 8B)
		{"Mistral-7B-v0.3", Model{NLayers: 32, NHeads: 32, NKVHeads: 8, EmbedLength: 4096, HeadDim: 128, VocabSize: 32768}, 512},
		// 2*8*256*4096*2*42 = 1,409,286,144 = 1344 MiB  (Gemma-2 head_dim 256, not embed/heads)
		{"Gemma-2-9B", Model{NLayers: 42, NHeads: 16, NKVHeads: 8, EmbedLength: 3584, HeadDim: 256, VocabSize: 256000}, 1344},
		// 2*10*128*4096*2*40 = 838,860,800 = 800 MiB  (GQA 10 kv heads)
		{"Phi-4", Model{NLayers: 40, NHeads: 40, NKVHeads: 10, EmbedLength: 5120, HeadDim: 128, VocabSize: 100352}, 800},
	}

	cfg := DefaultConfig()
	cfg.Ctx = ctx
	hw := Hardware{FreeVRAM: 80 * gib, FreeRAM: 256 * gib} // ample; KV total is offload-independent

	for _, c := range cases {
		c.m.Arch = "test"
		c.m.FileBytes = 1 * gib // arbitrary; KV doesn't depend on weights
		r, err := Predict(&c.m, hw, cfg)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		want := c.wantKVMiB * mib
		if r.KVBytes != want {
			t.Errorf("%s: KV = %d (%s), want %d (%d MiB)",
				c.name, r.KVBytes, HumanBytes(r.KVBytes), want, c.wantKVMiB)
		}
	}
}

// KV must scale linearly with context — doubling ctx doubles the cache.
func TestKVScalesWithContext(t *testing.T) {
	m := &Model{Arch: "test", NLayers: 32, NHeads: 32, NKVHeads: 8,
		EmbedLength: 4096, HeadDim: 128, VocabSize: 128256, FileBytes: 1 * gib}
	hw := Hardware{FreeVRAM: 80 * gib, FreeRAM: 256 * gib}

	cfg := DefaultConfig()
	cfg.Ctx = 4096
	r1, _ := Predict(m, hw, cfg)
	cfg.Ctx = 8192
	r2, _ := Predict(m, hw, cfg)

	if r2.KVBytes != 2*r1.KVBytes {
		t.Errorf("KV not linear in ctx: 4096→%s, 8192→%s (want 2×)",
			HumanBytes(r1.KVBytes), HumanBytes(r2.KVBytes))
	}
}
