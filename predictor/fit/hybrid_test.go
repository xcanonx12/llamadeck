package fit

// Hybrid (attention + SSM / linear-attention) architecture tests, pinned to a
// REAL failing launch: unsloth/Qwen3.5-9B-GGUF:Q4_K_S on a dual RTX 3080 with
// ctx 32768, ubatch 1024, KV q4_0, --gpus all, -ngl 19. The old pure-transformer
// engine said "fits with room"; the container crash-looped (exit 139) on a
// 986 MiB compute-buffer cudaMalloc on device 1. These tests assert the engine
// now predicts that OOM before launch.
//
// Ground truth (docker logs of the crash + exact tensor table of the GGUF):
//   qwen35: 32 blocks, every 4th is full attention (full_attention_interval=4),
//   ssm conv_kernel=4 state=128 groups=16 inner=4096, vocab 248320.
//   llama_memory_recurrent: 8.375 MiB per recurrent layer (4 seqs).
//   Split: CPU blocks 0..13 (1676.75 MiB), CUDA0 blocks 14..24 (1293.48 MiB),
//   CUDA1 blocks 25..31 + output head (1617.90 MiB) + 986 MiB compute → OOM.

import (
	"bytes"
	"testing"
)

// qwen35ish mirrors unsloth/Qwen3.5-9B-GGUF:Q4_K_S with its exact per-block
// tensor sizes (attention blocks ~119-125 MiB, recurrent blocks ~112 MiB).
func qwen35ish() *Model {
	per := []int64{ // MiB per block, from the real Q4_K_S tensor table
		125, 125, 125, 119, 119, 119, 119, 113, 119, 119, 119, 113, 119, 119, 119, 113,
		119, 119, 119, 112, 119, 119, 119, 112, 119, 119, 119, 112, 119, 119, 119, 112,
	}
	perB := make([]int64, len(per))
	var total int64
	for i, v := range per {
		perB[i] = v * mib
		total += perB[i]
	}
	embed, output := int64(545*mib), int64(795*mib)
	return &Model{
		Arch: "qwen35", NLayers: 32, NHeads: 16, NKVHeads: 4,
		HeadDim: 256, EmbedLength: 4096, VocabSize: 248320,
		CtxTrain: 262144, FileBytes: total + embed + output,
		SSMConvKernel: 4, SSMStateSize: 128, SSMGroupCount: 16, SSMInnerSize: 4096,
		FullAttnInterval: 4,
		Tensors: &TensorStats{
			Total: total + embed + output, Embedding: embed, Output: output,
			PerLayer: perB, ExpertPerLayer: make([]int64, 32),
		},
	}
}

func failingConfig() Config {
	c := DefaultConfig()
	c.Ctx = 32768
	c.UBatch = 1024
	c.KVType = "q4_0"
	c.NGL = 19
	return c
}

func TestRSPerLayerMatchesRealLaunch(t *testing.T) {
	m := qwen35ish()
	// llama_memory_recurrent measured exactly 8.375 MiB per recurrent layer:
	// ((conv_kernel-1)·(inner + 2·groups·state) + state·inner) · 4 B · 4 seqs.
	want := int64((3*(4096+2*16*128) + 128*4096) * 4 * 4)
	if got := rsPerLayer(m, DefaultConfig()); got != want {
		t.Fatalf("rsPerLayer = %d, want %d (8.375 MiB)", got, want)
	}
	if want != 8781824 {
		t.Fatalf("ground-truth constant drifted: %d", want)
	}
	// RS is per sequence: --parallel 8 doubles it, --parallel 1 quarters it.
	c8 := DefaultConfig()
	c8.NSeqs = 8
	if got := rsPerLayer(m, c8); got != 2*want {
		t.Fatalf("8 seqs should double RS: %d vs %d", got, 2*want)
	}
	c1 := DefaultConfig()
	c1.NSeqs = 1
	if got := rsPerLayer(m, c1); got != want/4 {
		t.Fatalf("1 seq should quarter RS: %d vs %d", got, want/4)
	}
}

func TestAttnLayerPattern(t *testing.T) {
	m := qwen35ish()
	attn := 0
	for i := 0; i < m.NLayers; i++ {
		if m.AttnLayer(i) {
			attn++
			if (i+1)%4 != 0 {
				t.Fatalf("layer %d flagged attention; want every 4th only", i)
			}
		}
	}
	if attn != 8 {
		t.Fatalf("attention layers = %d, want 8 (real KV buffers covered exactly 8)", attn)
	}

	// Pure SSM (no interval key): no attention layers at all.
	pure := qwen35ish()
	pure.FullAttnInterval = 0
	for i := 0; i < pure.NLayers; i++ {
		if pure.AttnLayer(i) {
			t.Fatalf("pure recurrent model must have no attention layers (layer %d)", i)
		}
	}

	// Dense transformer: every layer is attention.
	if d := llama3ish(); !d.AttnLayer(0) || !d.AttnLayer(31) {
		t.Fatal("dense model layers must all be attention")
	}
}

func TestHybridKVTotalIsPerLayerType(t *testing.T) {
	m := qwen35ish()
	c := failingConfig()
	r, err := Predict(m, Hardware{FreeVRAM: 24 * gib, FreeRAM: 32 * gib}, c)
	if err != nil {
		t.Fatal(err)
	}
	// 8 attention layers × KV + 24 recurrent layers × RS — NOT 32 × KV, which
	// over-counted by ~700 MiB on the real launch (real total: 288 KV + 201 RS).
	want := 8*kvPerLayer(m, c) + 24*rsPerLayer(m, c)
	if r.KVBytes != want {
		t.Fatalf("KVBytes = %s, want %s", HumanBytes(r.KVBytes), HumanBytes(want))
	}
	if !r.Recurrent {
		t.Fatal("hybrid model must be flagged Recurrent (approximate)")
	}
}

// The headline regression: the exact config that crash-looped must be predicted
// as a per-device OOM on the main GPU (device 1), not "fits with room".
func TestHybridExplicitNGLPredictsRealOOM(t *testing.T) {
	m := qwen35ish()
	c := failingConfig()
	// Free VRAM at launch time (nvidia-smi during the repro).
	hw := Hardware{GPUsFree: []int64{3451 * mib, 2633 * mib}, FreeRAM: 32 * gib}
	r, err := PredictDevices(m, hw, c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Fits || r.Mode != ModeOOM {
		t.Fatalf("real launch OOM'd on device 1; prediction says fits=%v mode=%s: %+v",
			r.Fits, r.Mode, r.Devices)
	}
	if r.Bottleneck != 1 {
		t.Fatalf("bottleneck = GPU%d, want GPU1 (the main device that cudaMalloc-failed)", r.Bottleneck)
	}
	// Explicit -ngl → proportional split: the roomier GPU0 gets more blocks
	// (real: 11 on CUDA0, 7+output on CUDA1).
	if r.Devices[0].Layers <= r.Devices[1].Layers {
		t.Fatalf("proportional split should favor GPU0: %d vs %d",
			r.Devices[0].Layers, r.Devices[1].Layers)
	}
	// Prediction must be conservative: predicted main-device usage ≥ the real
	// footprint that failed (1617.9 model + 72 KV + 41.88 RS + 986 compute).
	realGPU1 := int64(2718 * mib) // 1617.9 model + 72 KV + 41.88 RS + 986 compute
	if r.Devices[1].Used < realGPU1 {
		t.Fatalf("GPU1 predicted %s < real %s — still over-promising",
			HumanBytes(r.Devices[1].Used), HumanBytes(realGPU1))
	}
}

// Auto -ngl keeps the validated load-balanced split (13/1 on the dual-3080) —
// the explicit-NGL proportional branch must not regress it.
func TestAutoNGLStillLoadBalances(t *testing.T) {
	m := dualGPUModel()
	hw := Hardware{GPUsFree: []int64{9 * gib, 9 * gib}, FreeRAM: 32 * gib}
	r, err := PredictDevices(m, hw, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if r.Devices[0].Layers < r.Devices[1].Layers {
		t.Fatalf("auto -ngl must still push layers off the main device: %d vs %d",
			r.Devices[0].Layers, r.Devices[1].Layers)
	}
	if !r.Fits {
		t.Fatalf("auto case should fit: %+v", r.Devices)
	}
}

func TestParseGGUFHybridKeys(t *testing.T) {
	b := &ggufBuilder{}
	b.u32(ggufMagic)
	b.u32(3)
	b.u64(0)  // tensor_count
	b.u64(12) // kv_count
	b.kvStr("general.architecture", "qwen35")
	b.kvU32("qwen35.block_count", 32)
	b.kvU32("qwen35.attention.head_count", 16)
	b.kvU32("qwen35.attention.head_count_kv", 4)
	b.kvU32("qwen35.attention.key_length", 256)
	b.kvU32("qwen35.embedding_length", 4096)
	b.kvU32("qwen35.vocab_size", 248320)
	b.kvU32("qwen35.ssm.conv_kernel", 4)
	b.kvU32("qwen35.ssm.state_size", 128)
	b.kvU32("qwen35.ssm.group_count", 16)
	b.kvU32("qwen35.ssm.inner_size", 4096)
	b.kvU32("qwen35.full_attention_interval", 4)

	m, err := ParseGGUF(bytes.NewReader(b.buf.Bytes()), 5*gib)
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsRecurrent() {
		t.Fatal("qwen35 with ssm keys must be recurrent")
	}
	if m.SSMConvKernel != 4 || m.SSMStateSize != 128 || m.SSMGroupCount != 16 ||
		m.SSMInnerSize != 4096 || m.FullAttnInterval != 4 {
		t.Fatalf("ssm keys misparsed: %+v", m)
	}
	if llamaish, _ := ParseGGUF(bytes.NewReader(buildGGUF()), 5*gib); llamaish.IsRecurrent() {
		t.Fatal("plain llama must not be recurrent")
	}
}
