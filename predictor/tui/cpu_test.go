package tui

import (
	"strings"
	"testing"

	"llamadeck/fit"
)

func cpuState(freeRAM int64) State {
	m := &fit.Model{Arch: "llama", NLayers: 16, HeadDim: 64, NKVHeads: 8,
		NHeads: 8, VocabSize: 32000, EmbedLength: 2048, FileBytes: 800 << 20}
	c := fit.DefaultConfig()
	c.Ctx = 8192
	s := State{Source: "org/model", Model: m, Width: 80,
		HW: fit.Hardware{FreeVRAM: 0, FreeRAM: freeRAM, NumGPUs: 0}, Cfg: c, HideHint: true}
	s.Recompute()
	return s
}

func TestRenderView_CPU_RAMOnly(t *testing.T) {
	out := RenderView(cpuState(64 << 30))
	if strings.Contains(out, "VRAM") {
		t.Errorf("CPU view must not mention VRAM:\n%s", out)
	}
	if strings.Contains(out, "GPU layers") {
		t.Errorf("CPU view must not show the GPU-layer gauge:\n%s", out)
	}
	if !strings.Contains(out, "RAM") {
		t.Errorf("CPU view should show a RAM line:\n%s", out)
	}
	if !strings.Contains(out, "RUN") {
		t.Errorf("CPU fit should read RUN:\n%s", out)
	}
}

func TestRenderView_CPU_OOMWhenTight(t *testing.T) {
	s := cpuState(64 << 20)
	s.Cfg.Mlock = true
	s.Recompute()
	out := RenderView(s)
	if !strings.Contains(out, "OOM") {
		t.Errorf("CPU view should read OOM when RAM too small:\n%s", out)
	}
}
