package fit

// Bucket-level audit of the predictor against a REAL llama-server launch log.
// This is the "where exactly does the math start to lie?" mechanism: each
// memory bucket (weights, kv+rs, compute, per-device placement, verdict) is
// checked independently with its own tolerance, so a drift — a new llama.cpp
// buffer, a new architecture, a changed split — is pinned to the bucket that
// moved, not smeared into one aggregate error.
//
// Used two ways, same checks:
//   - `go test`: replays the saved corpus under fit/testdata/audit (real logs +
//     manifests) — instant, no GPU, catches regressions in the engine itself.
//   - `llamadeck audit`: runs them against a live container — catches drift in
//     llama.cpp or a new model family.

import "fmt"

// Check is one bucket's verdict.
type Check struct {
	Name   string
	Pred   int64
	Real   int64
	OK     bool
	Detail string // why it failed (empty when OK)
}

// auditSlack returns the placement slack for a model: our explicit -ngl model
// is deliberately one block conservative (llama counts the output head as an
// -ngl layer), so side-of-the-fence sums may differ by the largest block.
func auditSlack(m *Model) int64 {
	slack := int64(200 * mib)
	if ts := m.Tensors; ts != nil {
		for _, w := range ts.PerLayer {
			if w > slack {
				slack = w
			}
		}
		slack += slack / 4 // headroom over the biggest block
	}
	return slack
}

// within reports |a-b| <= tol.
func within(a, b, tol int64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// AuditLaunch compares a prediction to the measured footprint of a real launch
// and returns one Check per bucket. outcome is "ran" or "crashed" (empty =
// unknown, skips the verdict check).
func AuditLaunch(m *Model, hw Hardware, c Config, mm *Measured, outcome string) []Check {
	var checks []Check
	add := func(name string, pred, real int64, ok bool, detail string) {
		if ok {
			detail = ""
		}
		checks = append(checks, Check{Name: name, Pred: pred, Real: real, OK: ok, Detail: detail})
	}

	pooled, err := Predict(m, hw, c)
	if err != nil {
		return []Check{{Name: "predict", OK: false, Detail: err.Error()}}
	}
	slack := auditSlack(m)
	pct := func(v int64, p int64) int64 { return v * p / 100 }

	// GPU-side weights: exact tensor math, so ±2% plus one block of placement slack.
	realGPUW := mm.GPUModelBytes
	add("gpu-weights", pooled.GPUWeightBytes, realGPUW,
		within(pooled.GPUWeightBytes, realGPUW, slack+pct(realGPUW, 2)),
		"exact per-tensor sizing drifted beyond one block")

	// Host-side weights: same math, other side of the fence.
	realCPUW := mm.CPUModelBytes
	add("host-weights", pooled.CPUWeightBytes, realCPUW,
		within(pooled.CPUWeightBytes, realCPUW, slack+pct(realCPUW, 2)),
		"host (embedding + spilled layers) sizing drifted beyond one block")

	// KV + recurrent state, both sides combined: structural math, ±2% + one
	// layer-state of placement slack.
	realKV := mm.TotalKV()
	kvSlack := kvPerLayer(m, c) + rsPerLayer(m, c) + pct(realKV, 2)
	add("kv+rs", pooled.KVBytes, realKV,
		within(pooled.KVBytes, realKV, kvSlack),
		"KV/recurrent-state formula drifted (new arch? changed cache layout?)")

	// Compute: the one estimate. Must be conservative (>= real) but not absurd.
	realCmp := mm.TotalCompute()
	if realCmp > 0 {
		ok := pooled.ComputeBytes >= realCmp && pooled.ComputeBytes <= 4*realCmp
		add("compute", pooled.ComputeBytes, realCmp, ok,
			"compute estimate no longer conservative (or >4x wasteful) — recalibrate or fix the heuristic")
	}

	// Per-device placement: with >=2 known devices, each device's prediction
	// plus the load margin must cover its real buffers. The margin is exactly
	// what the verdict banks on — placement drift beyond it is what produces an
	// unpredicted per-device OOM (the original -ngl 19 crash fails this check).
	if len(hw.GPUsFree) > 1 {
		if dr, err := PredictDevices(m, hw, c); err == nil {
			for i, d := range dr.Devices {
				name := fmt.Sprintf("CUDA%d", i)
				var real int64
				for _, md := range mm.Devices {
					if md.Name == name {
						real = md.Total()
					}
				}
				if real == 0 {
					continue
				}
				add("placement-"+name, d.Used, real, d.Used+loadMargin(c) >= real,
					"device predicted below its real buffers by more than the load margin — split model drifted")
			}
		}
	}

	// Verdict vs reality: a crashed launch must have been flagged (OOM or
	// Tight); a healthy one must not have been called a hard OOM.
	if outcome != "" {
		var flagged bool
		var mode Mode
		if len(hw.GPUsFree) > 1 {
			if dr, err := PredictDevices(m, hw, c); err == nil {
				flagged, mode = !dr.Fits || dr.Tight, dr.Mode
			}
		} else {
			flagged, mode = pooled.Mode == ModeOOM || pooled.Tight, pooled.Mode
		}
		if outcome == "crashed" {
			add("verdict", 0, 0, flagged,
				fmt.Sprintf("launch CRASHED but prediction said %s without a tight/OOM flag", mode))
		} else {
			add("verdict", 0, 0, mode != ModeOOM,
				"launch RAN but prediction was a hard OOM — too pessimistic")
		}
	}
	return checks
}
