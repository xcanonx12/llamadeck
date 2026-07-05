package fit

// The allocation engine. Splits a model into weights + KV cache + compute
// buffer + base overhead, then fills VRAM layer-by-layer (mirroring llama.cpp's
// -ngl semantics) and spills the remainder to system RAM. Conservative by
// design: we'd rather predict OOM slightly early than promise a fit that crashes.

import "fmt"

const (
	mib = 1 << 20
	gib = 1 << 30
)

// KV cache element sizes in bytes, INCLUDING the quantized types' per-block
// scale overhead (ggml block = 32 elems: q4_0 = 18 B, q8_0 = 34 B). Validated
// against a real launch: q4_0 KV = 36 MiB per attention layer at ctx 32768
// (0.5 B/elem would predict 32).
var kvBytesPerElem = map[string]float64{
	"f16":  2.0,
	"q8_0": 34.0 / 32,
	"q4_0": 18.0 / 32,
}

// Config holds the runtime knobs that affect the memory footprint.
type Config struct {
	Ctx    int    // context size (tokens)
	UBatch int    // physical batch — drives the compute/logits buffer
	KVType string // f16 | q8_0 | q4_0
	// ponytail: BaseOverhead is the per-GPU CUDA context + cuBLAS scratch, a
	// fixed empirical constant. This is THE calibration knob — once we can launch
	// a model and read its real footprint, tune this (and ComputeFixed) per host.
	BaseOverhead int64
	ComputeFixed int64 // fixed graph scratch on top of the logits/activation terms
	// ComputeScale corrects the compute-buffer estimate from observed launches
	// (see calib.go). 1.0 = uncalibrated raw heuristic.
	ComputeScale float64
	// NCPUMoE mirrors --n-cpu-moe: keep the expert (FFN) tensors of the first N
	// layers on CPU. 0 = off (no-op). Only affects MoE models (ExpertPerLayer>0).
	NCPUMoE int
	// NGL mirrors -ngl: place exactly this many layers on the GPU. 0 = auto (the
	// greedy best fit, and the value reported as RecommendedNGL).
	NGL int
	// LoadMargin is the per-device VRAM headroom reserved for llama.cpp's
	// load-time extras the buffer log doesn't fully cover (CUDA context growth,
	// alloc_compute_meta, fragmentation). The auto fit RESERVES it (mirroring
	// llama_params_fit's own free-VRAM target); explicit -ngl configs inside it
	// are flagged Tight ("may crash at load"). 0 ⇒ DefaultLoadMargin.
	LoadMargin int64
	// NSeqs mirrors --parallel: server request slots. Recurrent state is per
	// sequence, so it scales the RS buffer of hybrid/SSM models (KV is a shared
	// ctx-token pool — unaffected). 0 ⇒ defaultSeqs (llama-server's default).
	NSeqs int
	// NoMmap / Mlock mirror --no-mmap / --mlock. With mmap (both false, the
	// default) CPU-side WEIGHTS are file-backed page cache — reclaimable, so
	// exceeding free RAM means paging (slow), not a crash. Either flag makes
	// them anonymous/pinned: a hard requirement. KV, recurrent state, and
	// compute are anonymous either way.
	NoMmap bool
	Mlock  bool
}

// DefaultLoadMargin is the measured load-time gap default. From the dual-3080
// residual campaign (predictor/VALIDATION.md): llama-server's real process VRAM
// exceeded its logged buffers by ~250 MiB/GPU (dense) to ~690 MiB/GPU
// (hybrid/big-vocab), and per-device placement noise added up to ~500 MiB near
// the edge. 512 MiB covers the observed tail without crying wolf on roomy fits.
const DefaultLoadMargin = 512 * mib

// loadMargin resolves the effective margin for a config: 0 ⇒ the measured
// default, negative ⇒ disabled (raw fit, for tests and comparisons).
func loadMargin(c Config) int64 {
	switch {
	case c.LoadMargin > 0:
		return c.LoadMargin
	case c.LoadMargin < 0:
		return 0
	}
	return DefaultLoadMargin
}

// DefaultConfig returns sane, deliberately conservative defaults.
func DefaultConfig() Config {
	return Config{
		Ctx:          4096,
		UBatch:       512,
		KVType:       "f16",
		BaseOverhead: 500 * mib,
		ComputeFixed: 64 * mib,
		ComputeScale: 1.0,
	}
}

// Hardware is the available (free, not total) memory on the host.
type Hardware struct {
	FreeVRAM int64 // bytes, summed across the GPUs in use; 0 ⇒ no usable GPU
	FreeRAM  int64 // bytes
	NumGPUs  int   // GPUs to model; 0 ⇒ treated as 1 when FreeVRAM>0 (back-compat)
	// GPUsFree is per-device free VRAM (bytes). Additive: when len>1 the caller
	// can run PredictDevices for a per-device split; pooled Predict ignores it.
	GPUsFree []int64
}

// gpuCount returns the effective GPU count for the prediction.
func (h Hardware) gpuCount() int {
	if h.NumGPUs > 0 {
		return h.NumGPUs
	}
	if h.FreeVRAM > 0 {
		return 1
	}
	return 0
}

// Mode is the predicted execution mode.
type Mode string

const (
	ModeGPU    Mode = "100% GPU"
	ModeHybrid Mode = "HYBRID"
	ModeCPU    Mode = "100% CPU"
	ModeOOM    Mode = "OOM"
)

// Result is the full prediction breakdown.
type Result struct {
	WeightsBytes   int64
	KVBytes        int64 // full KV cache + recurrent state across all layers
	ComputeBytes   int64
	BaseBytes      int64
	PerLayerWeight int64
	PerLayerKV     int64
	NLayers        int
	LayersOnGPU    int
	VRAMUsed       int64
	RAMUsed        int64
	Mode           Mode
	RecommendedNGL int
	MLA            bool // model uses Multi-head Latent Attention; KV estimate is approximate
	Recurrent      bool // hybrid/SSM model; layer split + state estimate is approximate
	// Tight: the config fits on paper but leaves less free VRAM than the
	// load-time margin — it may crash at load. Only explicit -ngl configs can
	// be tight; the auto fit reserves the margin up front.
	Tight bool
	// Paged: total RAM exceeds free RAM but the overflow is mmap'd weights
	// (reclaimable page cache) — it runs, paging from disk (slow), not OOM.
	// Only possible with mmap (NoMmap and Mlock both false).
	Paged bool

	// Exact side-of-the-fence totals (weights include embedding/output where they
	// actually live; KV includes recurrent state). The graph renders these instead
	// of layers×per-layer products, which lie for hybrids' non-uniform layers.
	GPUWeightBytes int64
	GPUKVBytes     int64
	CPUWeightBytes int64
	CPUKVBytes     int64
}

// kvPerLayer returns the KV-cache bytes for a single transformer layer.
func kvPerLayer(m *Model, c Config) int64 {
	bpe := kvBytesPerElem[c.KVType]
	if bpe == 0 {
		bpe = 2.0 // unknown type → assume f16
	}
	if m.IsMLA && m.KVLoraRank > 0 {
		// MLA (DeepSeek-V2/V3): a single compressed KV latent (kv_lora_rank) plus
		// the decoupled RoPE key per token — not per-head K and V. Approximate;
		// the prediction is flagged so callers can warn.
		return int64(float64(m.KVLoraRank+m.RopeDim) * float64(c.Ctx) * bpe)
	}
	// 2 = separate K and V tensors.
	return int64(2 * float64(m.NKVHeads) * float64(m.HeadDim) * float64(c.Ctx) * bpe)
}

// defaultSeqs is llama-server's default number of parallel sequences. Recurrent
// state is PER SEQUENCE (unlike KV, which is a shared ctx-token pool), so it
// multiplies the RS buffer. Verified: a real Qwen3.5-9B launch with no
// --parallel flag allocated exactly 4× the single-seq state per layer.
const defaultSeqs = 4

// rsPerLayer returns the recurrent-state bytes for ONE recurrent (SSM /
// linear-attention) layer: the f32 conv window plus the f32 ssm state, per
// sequence, times the sequence count (--parallel). Formula validated byte-exact
// against llama_memory_recurrent on a real Qwen3.5-9B launch (8.375 MiB/layer
// at the default 4 seqs).
func rsPerLayer(m *Model, c Config) int64 {
	if !m.IsRecurrent() {
		return 0
	}
	seqs := int64(c.NSeqs)
	if seqs <= 0 {
		seqs = defaultSeqs
	}
	conv := int64(m.SSMConvKernel-1) * int64(m.SSMInnerSize+2*m.SSMGroupCount*m.SSMStateSize)
	if conv < 0 {
		conv = 0
	}
	state := int64(m.SSMStateSize) * int64(m.SSMInnerSize)
	return (conv + state) * 4 * seqs
}

// stateAtLayer returns layer i's per-layer memory-state cost: KV cache for
// attention layers, recurrent state for SSM layers.
func stateAtLayer(m *Model, c Config, i int) int64 {
	if m.AttnLayer(i) {
		return kvPerLayer(m, c)
	}
	return rsPerLayer(m, c)
}

// computeBuffer estimates the transient compute/activation buffer. Dominated by
// the f32 logits buffer (ubatch × vocab); the activation term is a rough scratch
// estimate. ponytail: heuristic — refine against measured footprints later.
func computeBuffer(m *Model, c Config) int64 {
	logits := int64(c.UBatch) * int64(m.VocabSize) * 4
	activations := int64(c.UBatch) * int64(m.EmbedLength) * 8
	raw := logits + activations + c.ComputeFixed
	scale := c.ComputeScale
	if scale <= 0 {
		scale = 1.0 // zero-value Config ⇒ treat as uncalibrated
	}
	return int64(float64(raw) * scale)
}

// Predict runs the allocation and returns the breakdown. Returns an error only
// for structurally invalid models (so the caller can surface a clear message).
func Predict(m *Model, hw Hardware, c Config) (*Result, error) {
	if m.NLayers <= 0 || m.HeadDim <= 0 || m.VocabSize <= 0 {
		return nil, fmt.Errorf("incomplete model metadata (layers=%d head_dim=%d vocab=%d)",
			m.NLayers, m.HeadDim, m.VocabSize)
	}

	perLayerKV := kvPerLayer(m, c)
	// Per-layer memory-state cost: KV for attention layers, recurrent state for
	// SSM layers (identical for pure transformers). Hybrids are NOT uniform.
	stateAt := make([]int64, m.NLayers)
	kvTotal := int64(0)
	for i := range stateAt {
		stateAt[i] = stateAtLayer(m, c, i)
		kvTotal += stateAt[i]
	}
	compute := computeBuffer(m, c)

	// Per-layer weights: exact from the tensor table when available, else the
	// uniform FileBytes/NLayers approximation. The token embedding stays host
	// (CPU_Mapped) and the output head rides a GPU when anything is offloaded.
	perLayerW := make([]int64, m.NLayers)
	var embedBytes, outputBytes int64
	if ts := m.Tensors; ts != nil && len(ts.PerLayer) == m.NLayers {
		embedBytes, outputBytes = ts.Embedding, ts.Output
		copy(perLayerW, ts.PerLayer)
	} else {
		pw := m.FileBytes / int64(m.NLayers)
		for i := range perLayerW {
			perLayerW[i] = pw
		}
	}

	// --n-cpu-moe: expert weight of the first N layers rides host RAM, not the GPU.
	moeToRAM := offloadExperts(m, perLayerW, c)

	// Multi-GPU: llama.cpp pools VRAM (weights + KV split across devices) but each
	// device pays its OWN CUDA context AND its own compute buffer. So fixed GPU
	// overhead scales with the GPU count, while weights/KV fill the pooled budget.
	// (Conservative: per-device compute is approximated as N× the single estimate.)
	gpus := hw.gpuCount()
	baseTotal := int64(gpus) * c.BaseOverhead
	computeTotal := int64(gpus) * compute

	r := &Result{
		WeightsBytes:   m.FileBytes,
		KVBytes:        kvTotal,
		ComputeBytes:   compute, // overridden below to the on-device total in GPU mode
		BaseBytes:      baseTotal,
		PerLayerWeight: perLayerW[0],
		PerLayerKV:     perLayerKV,
		NLayers:        m.NLayers,
		MLA:            m.IsMLA,
		Recurrent:      m.IsRecurrent(),
	}

	// When any layer is offloaded, the OUTPUT HEAD rides the GPU. The token
	// embedding stays host-side (the input lookup runs on CPU) — EXCEPT for
	// tied-embedding models (no separate output.weight, e.g. Llama-3.2), where
	// the embedding tensor doubles as the output head and ships to the GPU too,
	// with the host keeping its own copy. Measured on real launches: Qwen3.5
	// (untied) GPU model buffer = blocks + output only; Llama (tied) ≈ whole file.
	// llama.cpp offloads the LAST -ngl blocks (verified against a real launch:
	// -ngl 19 on 32 layers put blocks 14..31 on GPU, 0..13 on CPU), so the greedy
	// fill walks from the top. Uniform models are unaffected; hybrids aren't
	// uniform (attention KV vs recurrent state, different weights per block).
	// The auto fit reserves the load-time margin per GPU (llama_params_fit
	// itself targets leaving VRAM free), so "auto" recommendations are safe by
	// construction; explicit -ngl bypasses the greedy fill and is judged below.
	gpuHead := outputBytes
	if gpuHead == 0 {
		gpuHead = embedBytes // tied embeddings serve as the output head
	}
	avail := hw.FreeVRAM - baseTotal - computeTotal - gpuHead - int64(gpus)*loadMargin(c)
	greedyMax := 0
	if avail >= 0 {
		for greedyMax < m.NLayers {
			i := m.NLayers - 1 - greedyMax
			if avail < perLayerW[i]+stateAt[i] {
				break
			}
			avail -= perLayerW[i] + stateAt[i]
			greedyMax++
		}
	}
	// Honor an explicit -ngl (c.NGL>0): place exactly that many layers (capped at
	// the layer count), even if it over- or under-subscribes VRAM, so the graph
	// reflects the user's choice live. 0 = auto = the greedy best fit.
	// llama.cpp actually counts the output head as one of the N (-ngl 12 loads 11
	// blocks + output, "offloaded 12/33") — we model N blocks + output, one block
	// conservative on the main device. Deliberate: over-predicting beats a crash.
	layersOnGPU := greedyMax
	if c.NGL > 0 {
		layersOnGPU = c.NGL
		if layersOnGPU > m.NLayers {
			layersOnGPU = m.NLayers
		}
	}
	r.LayersOnGPU = layersOnGPU
	r.RecommendedNGL = greedyMax

	// Split weight/state by what ended up on the GPU (the last layersOnGPU
	// blocks) vs host.
	gpuStart := m.NLayers - layersOnGPU
	var gpuLayerW, cpuWeight, gpuKV int64
	for i := 0; i < m.NLayers; i++ {
		if i >= gpuStart {
			gpuLayerW += perLayerW[i]
			gpuKV += stateAt[i]
		} else {
			cpuWeight += perLayerW[i]
		}
	}
	cpuKV := kvTotal - gpuKV

	if layersOnGPU == 0 {
		// Pure CPU: the whole model + KV + compute lives in host RAM (embedding once).
		r.VRAMUsed = 0
		r.RAMUsed = embedBytes + outputBytes + cpuWeight + kvTotal + compute + moeToRAM
		r.Mode = ModeCPU
		r.CPUWeightBytes = embedBytes + outputBytes + cpuWeight + moeToRAM
		r.CPUKVBytes = kvTotal
	} else {
		r.ComputeBytes = computeTotal
		r.VRAMUsed = baseTotal + computeTotal + gpuHead + gpuLayerW + gpuKV
		// Host RAM: embedding copy + CPU-side layer weights + their KV/RS +
		// offloaded experts. Validated against a real launch: predicted host
		// weights matched CPU_Mapped to <1% and CPU KV/RS exactly. llama also
		// keeps small host compute buffers (CPU + CUDA_Host, ~65 MiB at ubatch
		// 512) — left unmodeled, noise at RAM scale.
		r.RAMUsed = embedBytes + cpuWeight + cpuKV + moeToRAM
		r.GPUWeightBytes = gpuHead + gpuLayerW
		r.GPUKVBytes = gpuKV
		r.CPUWeightBytes = embedBytes + cpuWeight + moeToRAM
		r.CPUKVBytes = cpuKV
		if layersOnGPU == m.NLayers {
			r.Mode = ModeGPU
		} else {
			r.Mode = ModeHybrid
		}
	}
	// Whatever must live in RAM has to fit there, or it's an OOM — except
	// mmap'd weights (the default): those are reclaimable page cache, so only
	// the anonymous part (KV/RS, compute, and pinned weights) is a hard
	// requirement; weight overflow beyond free RAM means paging, not a crash.
	hardRAM := r.RAMUsed
	if !c.NoMmap && !c.Mlock {
		hardRAM -= r.CPUWeightBytes
	}
	if hardRAM > hw.FreeRAM {
		r.Mode = ModeOOM
	} else if r.RAMUsed > hw.FreeRAM {
		r.Paged = true
	}
	// An explicit -ngl can over-subscribe VRAM outright (hard OOM) or land
	// inside the load-time margin (tight — may crash at load). The auto fill
	// can't reach either: it reserved the margin above.
	if r.VRAMUsed > hw.FreeVRAM {
		r.Mode = ModeOOM
	} else if r.VRAMUsed > 0 && hw.FreeVRAM-r.VRAMUsed < loadMargin(c) {
		r.Tight = true
	}
	// Auto on a spilling HYBRID is tight too: llama's own auto-fit packs GPUs
	// to its reserve target, which under-covers recurrent models' load-time
	// extras (measured: a hybrid auto launch crash-looped, exit 139, while a
	// spilling dense auto launch was fine — see VALIDATION.md campaign run C).
	if c.NGL == 0 && m.IsRecurrent() && layersOnGPU > 0 && layersOnGPU < m.NLayers {
		r.Tight = true
	}
	return r, nil
}

// HumanBytes formats a byte count as a GiB/MiB string.
func HumanBytes(b int64) string {
	switch {
	case b >= gib:
		return fmt.Sprintf("%.2f GiB", float64(b)/gib)
	case b >= mib:
		return fmt.Sprintf("%.0f MiB", float64(b)/mib)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
