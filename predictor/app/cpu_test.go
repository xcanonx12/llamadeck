package app

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"llamadeck/fit"
)

func TestFitSettings_HW_CPUModeZerosVRAM(t *testing.T) {
	sh := &shared{
		hw:      fit.Hardware{FreeVRAM: 24 << 30, FreeRAM: 64 << 30, NumGPUs: 1},
		gpuFree: []int64{24 << 30},
		cpuMode: true,
	}
	s := newFitSettings(sh, 0)
	hw := s.hw()
	if hw.FreeVRAM != 0 {
		t.Errorf("cpuMode hw.FreeVRAM = %d, want 0", hw.FreeVRAM)
	}
	if hw.NumGPUs != 0 {
		t.Errorf("cpuMode hw.NumGPUs = %d, want 0", hw.NumGPUs)
	}
	if hw.GPUsFree != nil {
		t.Errorf("cpuMode hw.GPUsFree = %v, want nil", hw.GPUsFree)
	}
	if hw.FreeRAM != 64<<30 {
		t.Errorf("cpuMode hw.FreeRAM = %d, want RAM preserved", hw.FreeRAM)
	}
}

func TestToSpec_CPUMode(t *testing.T) {
	sh := &shared{
		hw:      fit.Hardware{FreeRAM: 64 << 30},
		cfg:     fit.DefaultConfig(),
		cpuMode: true,
	}
	sh.cfg.Ctx = 8192
	s := newFitSettings(sh, 8080)
	s.flash = true // would be on normally; CPU must clear it

	spec := s.toSpec("org/model:Q4_K_M", "/models")
	if !spec.CPU {
		t.Error("toSpec in cpuMode must set CPU=true")
	}
	if spec.NGL != 0 {
		t.Errorf("NGL = %d, want 0", spec.NGL)
	}
	if spec.Device != "" {
		t.Errorf("Device = %q, want empty", spec.Device)
	}
	if spec.NCPUMoE != 0 {
		t.Errorf("NCPUMoE = %d, want 0", spec.NCPUMoE)
	}
	if spec.FlashAttn {
		t.Error("FlashAttn must be off in CPU mode")
	}
	if spec.CacheTypeK != "" || spec.CacheTypeV != "" {
		t.Errorf("KV cache type must be default (f16) in CPU mode; got K=%q V=%q",
			spec.CacheTypeK, spec.CacheTypeV)
	}
}

func TestAdjust_LockedRowsNoOpInCPU(t *testing.T) {
	sh := &shared{hw: fit.Hardware{FreeRAM: 64 << 30}, cfg: fit.DefaultConfig(), cpuMode: true}
	s := newFitSettings(sh, 0)
	s.sel = rNGL
	s.adjust("right") // would set ngl in GPU mode
	if !s.nglAuto || s.ngl != 0 {
		t.Errorf("rNGL must not change in cpuMode: nglAuto=%v ngl=%d", s.nglAuto, s.ngl)
	}
}

func TestFitTab_MTogglesCPUMode(t *testing.T) {
	sh := &shared{hw: fit.Hardware{FreeVRAM: 24 << 30, FreeRAM: 64 << 30, NumGPUs: 1},
		cfg: fit.DefaultConfig(), cpuMode: false}
	sh.cfg.KVType = "q8_0"
	sh.selected = &selection{src: "org/model"}
	ft := newFitTab(sh).(*fitTab)

	ft.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("m")})
	if !sh.cpuMode {
		t.Error("m should enable cpuMode")
	}
	if sh.cfg.KVType != "f16" {
		t.Errorf("enabling cpuMode should force KVType to f16, got %q", sh.cfg.KVType)
	}
	ft.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("m")})
	if sh.cpuMode {
		t.Error("m should disable cpuMode on second press")
	}
}

func TestFitTab_MNoOpWithoutGPU(t *testing.T) {
	sh := &shared{hw: fit.Hardware{FreeVRAM: 0, FreeRAM: 64 << 30, NumGPUs: 0},
		cfg: fit.DefaultConfig(), cpuMode: true}
	sh.selected = &selection{src: "org/model"}
	ft := newFitTab(sh).(*fitTab)
	ft.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("m")})
	if !sh.cpuMode {
		t.Error("m must be a no-op on a GPU-less host; cpuMode should stay true")
	}
}

func TestConfirmPanel_CPUOOMWordsRAM(t *testing.T) {
	m := &fit.Model{Arch: "llama", NLayers: 16, HeadDim: 64, NKVHeads: 8,
		NHeads: 8, VocabSize: 32000, EmbedLength: 2048, FileBytes: 800 << 20}
	cfg := fit.DefaultConfig()
	cfg.Ctx = 8192
	cfg.Mlock = true // hard RAM requirement → OOM path
	sh := &shared{hw: fit.Hardware{FreeVRAM: 0, FreeRAM: 64 << 20, NumGPUs: 0},
		cfg: cfg, cpuMode: true, imageOK: true}
	sh.selected = &selection{src: "org/model", model: m}
	ft := newFitTab(sh).(*fitTab)
	out := ft.confirmPanel()
	if !strings.Contains(out, "RAM") {
		t.Errorf("CPU OOM warning should mention RAM:\n%s", out)
	}
	if strings.Contains(out, "VRAM") {
		t.Errorf("CPU OOM warning must not mention VRAM:\n%s", out)
	}
}
