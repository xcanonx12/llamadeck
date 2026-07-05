package fit

import "testing"

func TestMaxCtxMonotonic(t *testing.T) {
	m := llama3ish()
	cfg := DefaultConfig()
	hw := Hardware{FreeVRAM: 12 * gib, FreeRAM: 64 * gib}

	gpu := MaxCtxForMode(m, hw, cfg, ModeGPU)
	hybrid := MaxCtxForMode(m, hw, cfg, ModeHybrid)
	notOOM := MaxCtxForMode(m, hw, cfg, ModeCPU)

	if gpu <= 0 {
		t.Fatalf("expected some GPU-fitting ctx, got %d", gpu)
	}
	// Looser targets must allow at least as much context.
	if hybrid < gpu || notOOM < hybrid {
		t.Errorf("not monotonic: gpu=%d hybrid=%d notOOM=%d", gpu, hybrid, notOOM)
	}
	// The found GPU ctx must actually verify as GPU; the next doubling must not.
	c := cfg
	c.Ctx = gpu
	if r, _ := Predict(m, hw, c); r.Mode != ModeGPU {
		t.Errorf("ctx %d predicted %s, expected GPU", gpu, r.Mode)
	}
	c.Ctx = gpu * 2
	if r, _ := Predict(m, hw, c); r.Mode == ModeGPU && gpu*2 <= adviseMaxCtx {
		t.Errorf("ctx %d still fits GPU; boundary wrong", gpu*2)
	}
}

func TestMaxCtxNoneFits(t *testing.T) {
	m := llama3ish()
	cfg := DefaultConfig()
	// Can't fit even the smallest context fully on this GPU.
	hw := Hardware{FreeVRAM: 1 * gib, FreeRAM: 64 * gib}
	if got := MaxCtxForMode(m, hw, cfg, ModeGPU); got != 0 {
		t.Errorf("expected 0 (no GPU fit), got %d", got)
	}
}
