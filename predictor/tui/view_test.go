package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
)

func init() {
	// Deterministic plain output for substring asserts (no ANSI).
	lipgloss.SetColorProfile(0) // termenv.Ascii
}

func newState(vram, ram int64, ctx int) State {
	m := &fit.Model{
		Arch: "llama", NLayers: 32, NHeads: 32, NKVHeads: 8,
		EmbedLength: 4096, HeadDim: 128, FFNLength: 14336,
		CtxTrain: 8192, VocabSize: 128256, FileBytes: 4900 << 20,
	}
	cfg := fit.DefaultConfig()
	cfg.Ctx = ctx
	st := State{Source: "meta/Llama-3-8B-GGUF", Quant: "Q4_K_M", Model: m,
		HW: fit.Hardware{FreeVRAM: vram, FreeRAM: ram}, Cfg: cfg, Width: 72}
	st.Recompute()
	return st
}

func TestRenderViewGPU(t *testing.T) {
	out := RenderView(newState(24<<30, 32<<30, 8192))
	for _, want := range []string{"llamadeck", "Q4_K_M", "100% GPU", "32/32", "-ngl 32", "VRAM", "RAM"} {
		if !strings.Contains(out, want) {
			t.Errorf("view missing %q\n---\n%s", want, out)
		}
	}
}

func TestRenderViewOOM(t *testing.T) {
	out := RenderView(newState(0, 1<<30, 131072))
	if !strings.Contains(out, "OOM") {
		t.Errorf("expected OOM verdict\n---\n%s", out)
	}
	if !strings.Contains(out, "⚠") {
		t.Errorf("expected RAM overflow marker\n---\n%s", out)
	}
}

func TestRenderViewHybrid(t *testing.T) {
	out := RenderView(newState(4<<30, 64<<30, 8192))
	if !strings.Contains(out, "HYBRID") {
		t.Errorf("expected HYBRID verdict\n---\n%s", out)
	}
}

// Multi-GPU all path: per-device bars + fit/OOM verdict replace the pooled bar.
func TestRenderViewPerDevice(t *testing.T) {
	st := newState(20<<30, 32<<30, 8192) // pooled FreeVRAM kept for fallback
	st.HW.GPUsFree = []int64{10 << 30, 10 << 30}
	out := RenderView(st)
	for _, want := range []string{"GPU0", "GPU1", "main", "per-device split"} {
		if !strings.Contains(out, want) {
			t.Errorf("per-device view missing %q\n---\n%s", want, out)
		}
	}

	// The pool fits all layers, but the main (last) device is too small to hold
	// even its mandatory output+logits+compute buffer — the per-device OOM a
	// pooled total hides. This is the real failure per-device prediction catches.
	st.HW.GPUsFree = []int64{7 << 30, 700 << 20}
	out = RenderView(st)
	for _, want := range []string{"WON'T FIT", "GPU1 over by", "pin a single GPU"} {
		if !strings.Contains(out, want) {
			t.Errorf("per-device OOM view missing %q\n---\n%s", want, out)
		}
	}
}

// An explicit -ngl that fits on paper but inside the load-time margin must
// carry the TIGHT warning and the safe -ngl remedy; the OOM remedy on an
// explicit -ngl must name the safe value too.
func TestRenderViewTightAndSafeNGL(t *testing.T) {
	st := newState(20<<30, 32<<30, 8192)
	st.Cfg.NGL = 32
	// Size the GPU so the explicit full offload fits with < margin to spare.
	pooled, _ := fit.Predict(st.Model, fit.Hardware{FreeVRAM: 1 << 62, FreeRAM: 1 << 62}, st.Cfg)
	st.HW.FreeVRAM = pooled.VRAMUsed + (100 << 20)
	st.Recompute()
	out := RenderView(st)
	if !strings.Contains(out, "TIGHT") && !strings.Contains(out, "WON'T FIT") {
		t.Errorf("near-exact explicit fit must warn (TIGHT or WON'T FIT)\n---\n%s", out)
	}
	if !strings.Contains(out, "-ngl ≤") && !strings.Contains(out, "no explicit -ngl is safe") {
		t.Errorf("warning must include the safe -ngl remedy\n---\n%s", out)
	}
}
