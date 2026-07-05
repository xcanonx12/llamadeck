package fit

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// --- GGUF fixture builder -------------------------------------------------

type ggufBuilder struct{ buf bytes.Buffer }

func (b *ggufBuilder) u32(v uint32) { binary.Write(&b.buf, binary.LittleEndian, v) }
func (b *ggufBuilder) u64(v uint64) { binary.Write(&b.buf, binary.LittleEndian, v) }
func (b *ggufBuilder) str(s string) {
	b.u64(uint64(len(s)))
	b.buf.WriteString(s)
}

func (b *ggufBuilder) kvU32(key string, v uint32) {
	b.str(key)
	b.u32(gtUint32)
	b.u32(v)
}
func (b *ggufBuilder) kvStr(key, v string) {
	b.str(key)
	b.u32(gtString)
	b.str(v)
}

// buildGGUF returns a minimal valid GGUF byte stream with the given KV pairs.
// It also writes one string array (a fake tokenizer) to exercise array skipping.
func buildGGUF() []byte {
	b := &ggufBuilder{}
	b.u32(ggufMagic)
	b.u32(3) // version
	b.u64(0) // tensor_count
	b.u64(8) // metadata_kv_count

	b.kvStr("general.architecture", "llama")
	b.kvU32("llama.block_count", 32)
	b.kvU32("llama.attention.head_count", 32)
	b.kvU32("llama.attention.head_count_kv", 8)
	b.kvU32("llama.embedding_length", 4096)
	b.kvU32("llama.feed_forward_length", 14336)
	b.kvU32("llama.context_length", 8192)
	// A string array we must skip correctly to keep parsing aligned.
	b.str("tokenizer.ggml.tokens")
	b.u32(gtArray)
	b.u32(gtString)
	b.u64(3)
	b.str("<s>")
	b.str("</s>")
	b.str("hello")

	return b.buf.Bytes()
}

func TestParseGGUF(t *testing.T) {
	data := buildGGUF()
	m, err := ParseGGUF(bytes.NewReader(data), 5*gib)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	checks := []struct {
		name string
		got  int
		want int
	}{
		{"NLayers", m.NLayers, 32},
		{"NHeads", m.NHeads, 32},
		{"NKVHeads", m.NKVHeads, 8},
		{"EmbedLength", m.EmbedLength, 4096},
		{"FFNLength", m.FFNLength, 14336},
		{"CtxTrain", m.CtxTrain, 8192},
		{"HeadDim", m.HeadDim, 128},   // 4096 / 32
		{"VocabSize", m.VocabSize, 3}, // from token array length
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
	if m.Arch != "llama" {
		t.Errorf("Arch = %q, want llama", m.Arch)
	}
}

// llama3ish is a Llama-3-8B-shaped model used for the math asserts.
func llama3ish() *Model {
	return &Model{
		Arch: "llama", NLayers: 32, NHeads: 32, NKVHeads: 8,
		EmbedLength: 4096, HeadDim: 128, FFNLength: 14336,
		CtxTrain: 8192, VocabSize: 128256,
		FileBytes: 4900 * mib, // ~Q4_K_M
	}
}

func TestKVMathExact(t *testing.T) {
	m := llama3ish()
	c := DefaultConfig()
	c.Ctx = 8192
	// 2 * NKVHeads(8) * HeadDim(128) * Ctx(8192) * 2 bytes * NLayers(32)
	// = 33,554,432 per layer * 32 = 1 GiB exactly.
	want := int64(1 * gib)
	r, err := Predict(m, Hardware{FreeVRAM: 24 * gib, FreeRAM: 32 * gib}, c)
	if err != nil {
		t.Fatal(err)
	}
	if r.KVBytes != want {
		t.Errorf("KVBytes = %d (%s), want %d (1 GiB)", r.KVBytes, HumanBytes(r.KVBytes), want)
	}
}

func TestModes(t *testing.T) {
	m := llama3ish()
	c := DefaultConfig()
	c.Ctx = 8192

	cases := []struct {
		name   string
		hw     Hardware
		noMmap bool
		want   Mode
		paged  bool
	}{
		{"big GPU → all on GPU", Hardware{FreeVRAM: 24 * gib, FreeRAM: 32 * gib}, false, ModeGPU, false},
		{"small GPU + big RAM → hybrid", Hardware{FreeVRAM: 4 * gib, FreeRAM: 64 * gib}, false, ModeHybrid, false},
		{"no GPU + big RAM → CPU", Hardware{FreeVRAM: 0, FreeRAM: 64 * gib}, false, ModeCPU, false},
		// mmap (default): weights beyond free RAM are reclaimable page cache —
		// llama runs, paging from disk. Only --no-mmap/--mlock make it a hard OOM.
		{"no GPU + tiny RAM + mmap → paged CPU", Hardware{FreeVRAM: 0, FreeRAM: 2 * gib}, false, ModeCPU, true},
		{"no GPU + tiny RAM + no-mmap → OOM", Hardware{FreeVRAM: 0, FreeRAM: 2 * gib}, true, ModeOOM, false},
	}
	for _, tc := range cases {
		cc := c
		cc.NoMmap = tc.noMmap
		r, err := Predict(m, tc.hw, cc)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if r.Mode != tc.want || r.Paged != tc.paged {
			t.Errorf("%s: Mode = %q paged=%v, want %q paged=%v (ngl=%d/%d, vram=%s ram=%s)",
				tc.name, r.Mode, r.Paged, tc.want, tc.paged, r.LayersOnGPU, r.NLayers,
				HumanBytes(r.VRAMUsed), HumanBytes(r.RAMUsed))
		}
	}
}

func TestMlockMakesWeightsHard(t *testing.T) {
	m := llama3ish()
	c := DefaultConfig()
	c.Ctx = 8192
	c.Mlock = true // pinned pages can't be reclaimed → weights are a hard requirement
	r, _ := Predict(m, Hardware{FreeVRAM: 0, FreeRAM: 2 * gib}, c)
	if r.Mode != ModeOOM || r.Paged {
		t.Fatalf("mlock'd 4.9 GiB model on 2 GiB RAM must be OOM, got %s paged=%v", r.Mode, r.Paged)
	}
}

func TestHybridNGLInRange(t *testing.T) {
	m := llama3ish()
	c := DefaultConfig()
	c.Ctx = 8192
	r, _ := Predict(m, Hardware{FreeVRAM: 4 * gib, FreeRAM: 64 * gib}, c)
	if r.LayersOnGPU <= 0 || r.LayersOnGPU >= r.NLayers {
		t.Errorf("expected partial offload, got ngl=%d/%d", r.LayersOnGPU, r.NLayers)
	}
}
