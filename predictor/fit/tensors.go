package fit

// GGUF tensor-info parsing. The tensor table (name, dims, ggml type, offset)
// sits right after the metadata KV block in the same header window we already
// fetch, so we can size every tensor exactly — token embedding, output head,
// per-layer weights, and MoE expert tensors — without reading any weight data.
// This replaces the FileBytes/NLayers uniform approximation and underpins the
// per-device placement and n-cpu-moe models.

import "strings"

// ggmlType maps a ggml type tag to its block size (elements) and bytes-per-block.
// Bytes for a tensor = nelements / blk * size. Values mirror ggml's type traits;
// an unknown type aborts tensor sizing so the caller falls back cleanly.
var ggmlType = map[uint32]struct{ blk, size int }{
	0:  {1, 4},     // F32
	1:  {1, 2},     // F16
	2:  {32, 18},   // Q4_0
	3:  {32, 20},   // Q4_1
	6:  {32, 22},   // Q5_0
	7:  {32, 24},   // Q5_1
	8:  {32, 34},   // Q8_0
	9:  {32, 36},   // Q8_1
	10: {256, 84},  // Q2_K
	11: {256, 110}, // Q3_K
	12: {256, 144}, // Q4_K
	13: {256, 176}, // Q5_K
	14: {256, 210}, // Q6_K
	15: {256, 292}, // Q8_K
	16: {256, 66},  // IQ2_XXS
	17: {256, 74},  // IQ2_XS
	18: {256, 98},  // IQ3_XXS
	19: {256, 50},  // IQ1_S
	20: {32, 18},   // IQ4_NL
	21: {256, 110}, // IQ3_S
	22: {256, 82},  // IQ2_S
	23: {256, 136}, // IQ4_XS
	24: {1, 1},     // I8
	25: {1, 2},     // I16
	26: {1, 4},     // I32
	27: {1, 8},     // I64
	28: {1, 8},     // F64
	29: {256, 56},  // IQ1_M
	30: {1, 2},     // BF16
}

// TensorStats holds exact, tensor-derived weight sizes for a model. Nil on the
// Model when the header window didn't contain the full tensor table.
type TensorStats struct {
	Total          int64   // Σ all tensors (≈ FileBytes)
	Embedding      int64   // token_embd.* — usually CPU_Mapped, not GPU weight
	Output         int64   // output.weight (0 when tied to the embedding)
	PerLayer       []int64 // total weight of blk.{i}.* per repeating block
	ExpertPerLayer []int64 // expert (ffn_*_exps) bytes per block (0 = dense)
}

// ScaledTo returns a copy of the stats rescaled so Total == newTotal, preserving
// the per-layer / embedding / output / expert distribution. Used when the user
// picks a different quant of the same model: the tensor SHAPE is identical, only
// the byte sizes scale. ponytail: assumes uniform quant scaling — good enough for
// the graph; exact per-quant sizing would need re-parsing that quant's GGUF.
func (ts *TensorStats) ScaledTo(newTotal int64) *TensorStats {
	if ts == nil || ts.Total <= 0 || newTotal <= 0 {
		return ts
	}
	r := float64(newTotal) / float64(ts.Total)
	scale := func(v int64) int64 { return int64(float64(v) * r) }
	out := &TensorStats{
		Total:          newTotal,
		Embedding:      scale(ts.Embedding),
		Output:         scale(ts.Output),
		PerLayer:       make([]int64, len(ts.PerLayer)),
		ExpertPerLayer: make([]int64, len(ts.ExpertPerLayer)),
	}
	for i, v := range ts.PerLayer {
		out.PerLayer[i] = scale(v)
	}
	for i, v := range ts.ExpertPerLayer {
		out.ExpertPerLayer[i] = scale(v)
	}
	return out
}

// tensorBytes returns a tensor's on-disk byte size, false for an unknown type.
func tensorBytes(nelem int64, typ uint32) (int64, bool) {
	t, ok := ggmlType[typ]
	if !ok || t.blk == 0 {
		return 0, false
	}
	return nelem / int64(t.blk) * int64(t.size), true
}

// parseTensors reads the tensor-info table at the cursor's current position
// (immediately after the KV block) and aggregates per-tensor sizes. Returns nil
// if any tensor uses an unknown type or the window is truncated, so the caller
// can fall back to the FileBytes approximation rather than trust a partial sum.
func parseTensors(c *cursor, tensorCount uint64, nLayers int) *TensorStats {
	if nLayers <= 0 {
		return nil
	}
	ts := &TensorStats{
		PerLayer:       make([]int64, nLayers),
		ExpertPerLayer: make([]int64, nLayers),
	}
	for i := uint64(0); i < tensorCount; i++ {
		name, err := c.str()
		if err != nil {
			return nil
		}
		ndim, err := c.u32()
		if err != nil {
			return nil
		}
		nelem := int64(1)
		for d := uint32(0); d < ndim; d++ {
			dim, err := c.u64()
			if err != nil {
				return nil
			}
			nelem *= int64(dim)
		}
		typ, err := c.u32()
		if err != nil {
			return nil
		}
		if _, err := c.u64(); err != nil { // offset
			return nil
		}
		b, ok := tensorBytes(nelem, typ)
		if !ok {
			return nil // unknown type → don't trust a partial total
		}
		ts.Total += b
		classifyTensor(ts, name, b, nLayers)
	}
	return ts
}

// classifyTensor buckets a tensor by name into embedding / output / per-layer
// (and the expert sub-total within a layer).
func classifyTensor(ts *TensorStats, name string, b int64, nLayers int) {
	switch {
	case strings.HasPrefix(name, "token_embd"):
		ts.Embedding += b
	case strings.HasPrefix(name, "output.") || name == "output.weight":
		ts.Output += b
	}
	if !strings.HasPrefix(name, "blk.") {
		return
	}
	// blk.<i>.<rest>
	rest := name[len("blk."):]
	dot := strings.IndexByte(rest, '.')
	if dot < 0 {
		return
	}
	idx := atoiSafe(rest[:dot])
	if idx < 0 || idx >= nLayers {
		return
	}
	ts.PerLayer[idx] += b
	if strings.Contains(name, "_exps") { // MoE expert tensors: ffn_{gate,up,down}_exps
		ts.ExpertPerLayer[idx] += b
	}
}

func atoiSafe(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return -1
		}
		n = n*10 + int(s[i]-'0')
	}
	if s == "" {
		return -1
	}
	return n
}
