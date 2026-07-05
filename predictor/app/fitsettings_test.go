package app

import (
	"testing"

	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
)

func newTestSettings() *fitSettings {
	sh := &shared{cfg: fit.DefaultConfig()}
	sh.cfg.Ctx = 8192
	sh.cfg.UBatch = 512
	return newFitSettings(sh, 8080)
}

func TestFitSettingsNGLAuto(t *testing.T) {
	s := newTestSettings()
	// Auto must launch -ngl 999 (llama auto-fits/load-balances — what the auto
	// graph models), NOT the suggested count: an explicit count triggers llama's
	// proportional split, a different placement that can OOM the main GPU.
	if !s.nglAuto || s.effNGL() != 999 {
		t.Fatalf("default should be auto → -ngl 999, got auto=%v eff=%d", s.nglAuto, s.effNGL())
	}
	s.sel = rNGL
	s.adjust("right") // auto → 0 (manual)
	s.adjust("right") // 0 → 1
	if s.nglAuto || s.effNGL() != 1 {
		t.Fatalf("after edits expected manual ngl=1, got auto=%v eff=%d", s.nglAuto, s.effNGL())
	}
	s.adjust("left") // 1 → 0
	s.adjust("left") // 0 → auto
	if !s.nglAuto {
		t.Error("stepping below 0 should return to auto")
	}
}

func TestNGLLabelWarnsAboveSafeMax(t *testing.T) {
	if got := nglLabel(true, 0, 24, 18); got != "auto (24)" {
		t.Errorf("auto label = %q", got)
	}
	if got := nglLabel(false, 24, 24, 18); got != "24 ⚠ (max safe 18)" {
		t.Errorf("above-safe label = %q", got)
	}
	if got := nglLabel(false, 12, 24, 18); got != "12" {
		t.Errorf("safe explicit label = %q", got)
	}
	if got := nglLabel(false, 24, 24, 0); got != "24" {
		t.Errorf("unknown safe max must not warn: %q", got)
	}
}

func TestFitSettingsKVDrivesGraphAndLaunch(t *testing.T) {
	s := newTestSettings()
	s.sel = rKV
	s.adjust("right") // f16 → q8_0
	if s.sh.cfg.KVType != "q8_0" {
		t.Fatalf("KV cycle should write cfg.KVType=q8_0, got %s", s.sh.cfg.KVType)
	}
	// flash is off, but quantized KV must force it on in the spec
	s.flash = false
	spec := s.toSpec("o/r", "/c")
	if spec.CacheTypeK != "q8_0" || spec.CacheTypeV != "q8_0" {
		t.Errorf("expected q8_0 cache in spec, got K=%q V=%q", spec.CacheTypeK, spec.CacheTypeV)
	}
	if !spec.FlashAttn {
		t.Error("quantized KV must force flash-attn on")
	}
}

// The mmap/mlock toggles must drive BOTH the live graph (shared cfg → the RAM
// verdict flips between paged and hard-OOM) and the launch spec.
func TestNoMmapMlockDriveGraphAndSpec(t *testing.T) {
	s := newTestSettings()
	s.sel = rNoMmap
	s.adjust(" ")
	if !s.sh.cfg.NoMmap {
		t.Fatal("No-mmap toggle must write cfg.NoMmap (live graph)")
	}
	s.sel = rMlock
	s.adjust(" ")
	if !s.sh.cfg.Mlock {
		t.Fatal("mlock toggle must write cfg.Mlock (live graph)")
	}
	spec := s.toSpec("o/r", "/c")
	if !spec.NoMmap || !spec.Mlock {
		t.Fatalf("spec must carry the toggles: nommap=%v mlock=%v", spec.NoMmap, spec.Mlock)
	}
}

func TestFitSettingsEditPort(t *testing.T) {
	s := newTestSettings()
	s.port = 0
	s.sel = rPort
	s.startEdit()
	if !s.editing {
		t.Fatal("e on a numeric row should enter edit mode")
	}
	for _, k := range []string{"8", "0", "9", "0"} {
		s.editKey(k)
	}
	s.editKey("enter")
	if s.editing {
		t.Fatal("enter should exit edit mode")
	}
	if s.port != 8090 {
		t.Fatalf("typed port should be 8090, got %d", s.port)
	}
}

func TestFitSettingsEditRejectsNonNumericAndEsc(t *testing.T) {
	s := newTestSettings()
	s.sel = rKV // not numeric
	s.startEdit()
	if s.editing {
		t.Fatal("e on the KV picker must not enter edit mode")
	}
	s.sel = rThreads
	s.startEdit()
	s.editKey("4")
	s.editKey("esc")
	if s.editing || s.threads != 0 {
		t.Fatalf("esc should cancel with no change, got editing=%v threads=%d", s.editing, s.threads)
	}
}

func TestFitSettingsMoEGatedOnDense(t *testing.T) {
	s := newTestSettings()
	s.sh.selected = &selection{model: &fit.Model{NLayers: 16}} // dense: no tensor table
	s.sel = rMoE
	s.adjust("right")
	if s.moe != 0 {
		t.Fatalf("dense model MoE must stay 0, got %d", s.moe)
	}
	s.startEdit()
	if s.editing {
		t.Fatal("dense model MoE must not be editable")
	}
}

// The Fit tab must never render more lines than its height budget, or the app
// header scrolls off the top on short terminals.
func TestFitTabRespectsHeight(t *testing.T) {
	lipgloss.SetColorProfile(0)
	sh := &shared{cfg: fit.DefaultConfig(), imageOK: true}
	sh.hw = fit.Hardware{FreeVRAM: 12 << 30, FreeRAM: 32 << 30, NumGPUs: 1}
	sh.selected = &selection{src: "meta/Llama-3-8B", model: &fit.Model{
		Arch: "llama", NLayers: 32, NHeads: 32, NKVHeads: 8,
		EmbedLength: 4096, HeadDim: 128, VocabSize: 128256, FileBytes: 4900 << 20,
	}}
	tab := newFitTab(sh)
	for _, h := range []int{20, 24, 30, 40} {
		out := tab.View(90, h)
		if got := lipgloss.Height(out); got > h {
			t.Errorf("height %d: view rendered %d lines (overflow)", h, got)
		}
	}
}

func TestFitSettingsCtxClampAndF16Omitted(t *testing.T) {
	s := newTestSettings()
	s.sel = rCtx
	s.adjust("right") // 8192 → 16384
	if s.sh.cfg.Ctx != 16384 {
		t.Errorf("ctx right should double to 16384, got %d", s.sh.cfg.Ctx)
	}
	// default f16 KV → omitted (server default), flash stays as set
	s2 := newTestSettings()
	spec := s2.toSpec("o/r", "/c")
	if spec.CacheTypeK != "" || spec.CacheTypeV != "" {
		t.Errorf("f16 KV should be omitted, got K=%q", spec.CacheTypeK)
	}
	if spec.NGL != 999 {
		t.Errorf("auto ngl must launch as 999 (llama auto-fit), got %d", spec.NGL)
	}
}
