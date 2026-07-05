package app

import (
	"strings"
	"testing"

	"llamadeck/fit"
)

func TestSplitWidths(t *testing.T) {
	cases := []struct {
		width, leftW, rightW int
	}{
		{200, 64, 133}, // cap holds: 200*2/5=80 → 64; 200-64-3
		{120, 48, 69},  // 120*2/5=48 in range; 120-48-3
		{80, 32, 45},   // 80*2/5=32; 80-32-3
		{45, 18, 24},   // 45*2/5=18→28, rightW 14<24 → leftW=45-24-3=18, rightW=24
	}
	for _, c := range cases {
		gotL, gotR := splitWidths(c.width)
		if gotL != c.leftW || gotR != c.rightW {
			t.Errorf("splitWidths(%d) = (%d, %d), want (%d, %d)",
				c.width, gotL, gotR, c.leftW, c.rightW)
		}
		// Divider is always 3 columns; the three widths must sum to the total.
		if gotL+gotR+3 != c.width {
			t.Errorf("splitWidths(%d): %d+%d+3 != %d", c.width, gotL, gotR, c.width)
		}
	}
}

func TestPreviewPanel_FullQuantList(t *testing.T) {
	sh := &shared{
		hw:  fit.Hardware{FreeVRAM: 8 << 30, FreeRAM: 32 << 30, NumGPUs: 1},
		cfg: fit.DefaultConfig(),
	}
	quants := []string{"IQ2_XS", "IQ3_XS", "IQ4_XS", "Q2_K", "Q3_K_M", "Q4_K_M",
		"Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "BF16", "F16"}
	mt := &modelsTab{
		sh:      sh,
		preview: &fit.Result{Mode: fit.ModeGPU, LayersOnGPU: 16, NLayers: 16,
			VRAMUsed: 1 << 30, RAMUsed: 1 << 30, WeightsBytes: 1 << 30,
			KVBytes: 1 << 28, ComputeBytes: 1 << 27},
		pModel:  &fit.Model{Arch: "llama", NLayers: 16, VocabSize: 32000},
		pRepo:   "org/model",
		pQuant:  "Q4_K_M",
		pQuants: quants,
	}
	// Narrow panel forces wrapping across multiple lines.
	out := mt.previewPanel(30)

	for _, q := range quants {
		if !strings.Contains(out, q) {
			t.Errorf("quant %q missing from preview (list was truncated):\n%s", q, out)
		}
	}
	if strings.Contains(out, "…") {
		t.Errorf("preview must not truncate the quant list with an ellipsis:\n%s", out)
	}
}

func TestPreviewPanel_NameAdapts(t *testing.T) {
	sh := &shared{
		hw:  fit.Hardware{FreeVRAM: 8 << 30, FreeRAM: 32 << 30, NumGPUs: 1},
		cfg: fit.DefaultConfig(),
	}
	mt := &modelsTab{
		sh:      sh,
		preview: &fit.Result{Mode: fit.ModeGPU, LayersOnGPU: 80, NLayers: 80},
		pModel:  &fit.Model{Arch: "llama", NLayers: 80, VocabSize: 128000},
		pRepo:   "unsloth/Meta-Llama-3.1-70B-Instruct-GGUF",
		pQuant:  "Q4_K_M",
	}
	// Narrow panel: the long name + quant must wrap, not truncate.
	out := mt.previewPanel(30)

	// The repo tail ("GGUF") and the recommended quant both survive — a
	// truncate(name, w) would have cut them and appended an ellipsis.
	if !strings.Contains(out, "GGUF") {
		t.Errorf("model name tail truncated (should wrap, not cut):\n%s", out)
	}
	if !strings.Contains(out, "Q4_K_M") {
		t.Errorf("recommended quant missing next to the name:\n%s", out)
	}
	if strings.Contains(out, "…") {
		t.Errorf("name line must not truncate with an ellipsis:\n%s", out)
	}
}
