package fit

// Advisory helpers: turn the one-shot predictor into a recommender. "What's the
// largest context I can fit?" and "which quant should I pick?" are the questions
// an engineer actually has before downloading anything.

const (
	adviseMinCtx = 512
	adviseMaxCtx = 1 << 20
)

// ModeRank orders execution modes best→worst so they can be compared.
func ModeRank(m Mode) int {
	switch m {
	case ModeGPU:
		return 3
	case ModeHybrid:
		return 2
	case ModeCPU:
		return 1
	default: // OOM
		return 0
	}
}

// SafeMaxNGL returns the largest explicit -ngl that fits every device with the
// load-time margin to spare (and doesn't overflow RAM) — the number to show a
// user whose hand-picked -ngl is predicted to crash. Scans downward with the
// explicit-split model (proportional per-device when multi-GPU free VRAM is
// known, pooled otherwise). Returns 0 when no explicit -ngl is safe.
func SafeMaxNGL(m *Model, hw Hardware, c Config) int {
	for n := m.NLayers; n >= 1; n-- {
		cc := c
		cc.NGL = n
		if len(hw.GPUsFree) > 1 {
			dr, err := PredictDevices(m, hw, cc)
			if err == nil {
				if dr.Fits && !dr.Tight && dr.Mode != ModeOOM {
					return n
				}
				continue
			}
			// fall through to pooled when per-device isn't available
		}
		r, err := Predict(m, hw, cc)
		if err != nil {
			return 0
		}
		if r.Mode != ModeOOM && !r.Tight {
			return n
		}
	}
	return 0
}

// MaxCtxForMode returns the largest power-of-two context whose predicted mode is
// at least as good as target. Memory grows monotonically with context, so mode
// only degrades as ctx rises — a simple doubling scan finds the boundary.
// Returns 0 if even the smallest context can't meet the target.
func MaxCtxForMode(model *Model, hw Hardware, cfg Config, target Mode) int {
	want := ModeRank(target)
	best := 0
	for ctx := adviseMinCtx; ctx <= adviseMaxCtx; ctx *= 2 {
		c := cfg
		c.Ctx = ctx
		r, err := Predict(model, hw, c)
		if err != nil || ModeRank(r.Mode) < want {
			break
		}
		best = ctx
	}
	return best
}
