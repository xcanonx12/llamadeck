package fit

// Minimal GGUF header parser. We only read the metadata KV block — enough to
// pull the structural keys the memory engine needs — never the tensor data.
// Works over any io.ReaderAt, so the same code parses a local file or a byte
// window fetched via an HTTP Range request.

import (
	"encoding/binary"
	"fmt"
	"io"
)

const ggufMagic = 0x46554747 // "GGUF" little-endian

// GGUF metadata value type tags.
const (
	gtUint8 uint32 = iota
	gtInt8
	gtUint16
	gtInt16
	gtUint32
	gtInt32
	gtFloat32
	gtBool
	gtString
	gtArray
	gtUint64
	gtInt64
	gtFloat64
)

// Model holds the structural keys extracted from a GGUF header plus the total
// weight size (≈ file size). All counts are exact; FileBytes is the on-disk size.
type Model struct {
	Arch        string
	NLayers     int
	NHeads      int
	NKVHeads    int
	EmbedLength int
	HeadDim     int // key_length if present, else EmbedLength/NHeads
	FFNLength   int
	CtxTrain    int
	VocabSize   int
	FileBytes   int64

	// Multi-head Latent Attention (DeepSeek-V2/V3): KV is compressed into a latent
	// vector, so the standard per-head KV formula does NOT apply.
	IsMLA      bool
	KVLoraRank int // compressed KV latent dimension
	RopeDim    int // decoupled RoPE key dimension

	// Recurrent / hybrid architectures (Mamba, Jamba, Qwen3-Next/3.5 "Gated Delta
	// Net"): some or ALL layers keep a fixed-size recurrent state instead of a
	// growing KV cache. Parsed from the {arch}.ssm.* GGUF keys.
	SSMConvKernel    int
	SSMStateSize     int
	SSMGroupCount    int
	SSMInnerSize     int
	FullAttnInterval int // every Nth layer is full attention (0 = none are)

	// Tensors holds exact per-tensor sizes from the GGUF tensor table; nil when
	// the header window didn't include the full table (engine falls back to the
	// FileBytes/NLayers approximation).
	Tensors *TensorStats
}

// IsRecurrent reports whether the model has recurrent (SSM / linear-attention)
// layers — pure Mamba or a hybrid like Qwen3.5's Gated Delta Net. Those layers
// keep a per-sequence recurrent state instead of a per-token KV cache, and the
// prediction is flagged approximate (like MLA).
func (m *Model) IsRecurrent() bool {
	return m.SSMStateSize > 0 && m.SSMInnerSize > 0
}

// AttnLayer reports whether layer i is a full-attention layer (KV cache) as
// opposed to a recurrent one (fixed-size state). Hybrids declare every Nth
// layer attention via full_attention_interval; pure SSM models have none.
func (m *Model) AttnLayer(i int) bool {
	if !m.IsRecurrent() {
		return true
	}
	if m.FullAttnInterval <= 0 {
		return false
	}
	return (i+1)%m.FullAttnInterval == 0
}

// IsMoE reports whether the model has expert (MoE) tensors, i.e. whether
// --n-cpu-moe can offload anything. False for dense models and when the tensor
// table wasn't parsed.
func (m *Model) IsMoE() bool {
	if m.Tensors == nil {
		return false
	}
	for _, e := range m.Tensors.ExpertPerLayer {
		if e > 0 {
			return true
		}
	}
	return false
}

// cursor reads little-endian primitives sequentially from an io.ReaderAt.
type cursor struct {
	r   io.ReaderAt
	off int64
}

func (c *cursor) read(n int) ([]byte, error) {
	b := make([]byte, n)
	if _, err := c.r.ReadAt(b, c.off); err != nil {
		return nil, fmt.Errorf("read at %d (+%d): %w", c.off, n, err)
	}
	c.off += int64(n)
	return b, nil
}

func (c *cursor) u32() (uint32, error) {
	b, err := c.read(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (c *cursor) u64() (uint64, error) {
	b, err := c.read(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (c *cursor) str() (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	b, err := c.read(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// sizeOf returns the byte width of a fixed-size scalar type, or -1 if variable.
func sizeOf(t uint32) int {
	switch t {
	case gtUint8, gtInt8, gtBool:
		return 1
	case gtUint16, gtInt16:
		return 2
	case gtUint32, gtInt32, gtFloat32:
		return 4
	case gtUint64, gtInt64, gtFloat64:
		return 8
	}
	return -1
}

// readScalar reads one scalar value, returning it as uint64 when numeric (we
// only need integer counts), the string for strings, or nil for floats/bools.
func (c *cursor) readScalar(t uint32) (any, error) {
	switch t {
	case gtUint8, gtInt8, gtBool:
		b, err := c.read(1)
		if err != nil {
			return nil, err
		}
		return uint64(b[0]), nil
	case gtUint16, gtInt16:
		b, err := c.read(2)
		if err != nil {
			return nil, err
		}
		return uint64(binary.LittleEndian.Uint16(b)), nil
	case gtUint32, gtInt32, gtFloat32:
		v, err := c.u32()
		return uint64(v), err
	case gtUint64, gtInt64, gtFloat64:
		v, err := c.u64()
		return v, err
	case gtString:
		return c.str()
	}
	return nil, fmt.Errorf("unknown scalar type %d", t)
}

// skipArray consumes an array value, returning its element count (we use the
// count of tokenizer.ggml.tokens as a vocab-size fallback).
func (c *cursor) skipArray() (int, error) {
	elemType, err := c.u32()
	if err != nil {
		return 0, err
	}
	count, err := c.u64()
	if err != nil {
		return 0, err
	}
	if w := sizeOf(elemType); w > 0 {
		c.off += int64(count) * int64(w) // fixed-width: jump past in one move
		return int(count), nil
	}
	// Variable-width elements (strings, nested arrays): walk each one.
	for i := uint64(0); i < count; i++ {
		if elemType == gtString {
			if _, err := c.str(); err != nil {
				return 0, err
			}
		} else if elemType == gtArray {
			if _, err := c.skipArray(); err != nil {
				return 0, err
			}
		} else {
			return 0, fmt.Errorf("unsupported array elem type %d", elemType)
		}
	}
	return int(count), nil
}

// ParseGGUF reads the metadata block from r and maps it to a Model. fileBytes
// is the total file size (weights ≈ file size); pass it from stat or HTTP HEAD.
func ParseGGUF(r io.ReaderAt, fileBytes int64) (*Model, error) {
	c := &cursor{r: r}
	magic, err := c.u32()
	if err != nil {
		return nil, err
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file (magic %#x)", magic)
	}
	if _, err := c.u32(); err != nil { // version
		return nil, err
	}
	tensorCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	kvCount, err := c.u64()
	if err != nil {
		return nil, err
	}

	kv := make(map[string]uint64)
	var arch string
	var tokenCount int
	for i := uint64(0); i < kvCount; i++ {
		key, err := c.str()
		if err != nil {
			return nil, err
		}
		vtype, err := c.u32()
		if err != nil {
			return nil, err
		}
		if vtype == gtArray {
			n, err := c.skipArray()
			if err != nil {
				return nil, err
			}
			if key == "tokenizer.ggml.tokens" {
				tokenCount = n
			}
			continue
		}
		val, err := c.readScalar(vtype)
		if err != nil {
			return nil, err
		}
		switch v := val.(type) {
		case string:
			if key == "general.architecture" {
				arch = v
			}
		case uint64:
			kv[key] = v
		}
	}
	if arch == "" {
		return nil, fmt.Errorf("missing general.architecture")
	}

	get := func(suffix string) int { return int(kv[arch+"."+suffix]) }
	m := &Model{
		Arch:        arch,
		NLayers:     get("block_count"),
		NHeads:      get("attention.head_count"),
		NKVHeads:    get("attention.head_count_kv"),
		EmbedLength: get("embedding_length"),
		FFNLength:   get("feed_forward_length"),
		CtxTrain:    get("context_length"),
		VocabSize:   get("vocab_size"),
		FileBytes:   fileBytes,
	}
	if m.NKVHeads == 0 {
		m.NKVHeads = m.NHeads // no GQA → KV heads == attention heads
	}
	if kl := get("attention.key_length"); kl > 0 {
		m.HeadDim = kl
	} else if m.NHeads > 0 {
		m.HeadDim = m.EmbedLength / m.NHeads
	}
	if m.VocabSize == 0 {
		m.VocabSize = tokenCount // fallback to tokenizer token count
	}

	// MLA detection (DeepSeek-V2/V3): a compressed KV latent rather than per-head
	// K/V. Detected by the kv_lora_rank key or the deepseek2 architecture.
	m.KVLoraRank = get("attention.kv_lora_rank")
	m.RopeDim = get("attention.qk_rope_head_dim")
	if m.RopeDim == 0 {
		m.RopeDim = get("rope.dimension_count")
	}
	m.IsMLA = m.KVLoraRank > 0 || arch == "deepseek2"

	// Recurrent / hybrid detection: the {arch}.ssm.* keys carried by Mamba,
	// Jamba, and Qwen3-Next/3.5 (verified against a real Qwen3.5-9B header:
	// ssm.conv_kernel=4, state_size=128, group_count=16, inner_size=4096,
	// full_attention_interval=4 → every 4th layer attention, rest recurrent).
	m.SSMConvKernel = get("ssm.conv_kernel")
	m.SSMStateSize = get("ssm.state_size")
	m.SSMGroupCount = get("ssm.group_count")
	m.SSMInnerSize = get("ssm.inner_size")
	m.FullAttnInterval = get("full_attention_interval")

	// Exact per-tensor sizes (cursor is now at the tensor-info table). Best-effort:
	// nil if the window was truncated or a type is unknown — engine falls back.
	m.Tensors = parseTensors(c, tensorCount, m.NLayers)
	return m, nil
}
