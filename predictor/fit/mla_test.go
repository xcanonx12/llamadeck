package fit

import (
	"bytes"
	"testing"
)

// buildDeepseekMLA builds a minimal deepseek2 GGUF with MLA metadata.
func buildDeepseekMLA() []byte {
	b := &ggufBuilder{}
	b.u32(ggufMagic)
	b.u32(3) // version
	b.u64(0) // tensor_count
	b.u64(8) // metadata_kv_count

	b.kvStr("general.architecture", "deepseek2")
	b.kvU32("deepseek2.block_count", 4)
	b.kvU32("deepseek2.attention.head_count", 16)
	b.kvU32("deepseek2.attention.head_count_kv", 16)
	b.kvU32("deepseek2.embedding_length", 2048)
	b.kvU32("deepseek2.attention.kv_lora_rank", 512)
	b.kvU32("deepseek2.attention.qk_rope_head_dim", 64)
	b.kvU32("deepseek2.vocab_size", 1000)
	return b.buf.Bytes()
}

func TestParseMLA(t *testing.T) {
	m, err := ParseGGUF(bytes.NewReader(buildDeepseekMLA()), 1*gib)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !m.IsMLA {
		t.Error("expected IsMLA=true for deepseek2")
	}
	if m.KVLoraRank != 512 || m.RopeDim != 64 {
		t.Errorf("MLA dims = lora %d, rope %d; want 512, 64", m.KVLoraRank, m.RopeDim)
	}
}

func TestMLAKVMath(t *testing.T) {
	m := &Model{
		Arch: "deepseek2", NLayers: 4, NHeads: 16, NKVHeads: 16,
		EmbedLength: 2048, HeadDim: 128, VocabSize: 1000, FileBytes: 1 * gib,
		IsMLA: true, KVLoraRank: 512, RopeDim: 64,
	}
	c := DefaultConfig()
	c.Ctx = 4096
	r, err := Predict(m, Hardware{FreeVRAM: 24 * gib, FreeRAM: 64 * gib}, c)
	if err != nil {
		t.Fatal(err)
	}
	// MLA: (kv_lora_rank + rope_dim) * ctx * 2 bytes * layers
	// = (512+64) * 4096 * 2 * 4 = 18,874,368 = 18 MiB
	want := int64((512 + 64) * 4096 * 2 * 4)
	if r.KVBytes != want {
		t.Errorf("MLA KV = %d (%s), want %d", r.KVBytes, HumanBytes(r.KVBytes), want)
	}
	if !r.MLA {
		t.Error("Result.MLA should be true")
	}

	// Sanity: MLA KV must be far smaller than the naive per-head formula would give.
	naive := int64(2 * 16 * 128 * 4096 * 2 * 4)
	if r.KVBytes >= naive {
		t.Errorf("MLA KV (%d) should be far below the naive per-head estimate (%d)", r.KVBytes, naive)
	}
}
