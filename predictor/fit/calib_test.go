package fit

import "testing"

// A realistic recent llama-server load log (hybrid offload).
const sampleLog = `
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors
load_tensors: offloading 28 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size =  4156.00 MiB
load_tensors:   CPU_Mapped model buffer size =   281.81 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   896.00 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =    41.88 MiB
llama_memory_recurrent:        CPU RS buffer size =   134.00 MiB
llama_memory_recurrent: size =  201.00 MiB (     4 cells,  32 layers,  4 seqs), R (f32):    9.00 MiB, S (f32):  192.00 MiB
llama_context:      CUDA0 compute buffer size =   304.00 MiB
llama_context:        CPU compute buffer size =    16.01 MiB
`

func TestParseServerLog(t *testing.T) {
	m, err := ParseServerLog(sampleLog)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	checks := []struct {
		name string
		got  int64
		want int64
	}{
		{"GPUModel", m.GPUModelBytes, unitToBytes(4156, "MiB")},
		{"CPUModel", m.CPUModelBytes, unitToBytes(281.81, "MiB")},
		// KV bucket includes recurrent state (896 KV + 41.88 GPU RS); the
		// engine's KVBytes prediction covers both, so measured must too. The
		// "llama_memory_recurrent: size = 201.00 MiB" summary line must NOT match.
		{"GPUKV", m.GPUKVBytes, unitToBytes(896, "MiB") + unitToBytes(41.88, "MiB")},
		{"CPUKV", m.CPUKVBytes, unitToBytes(134, "MiB")},
		{"GPUCompute", m.GPUComputeBytes, unitToBytes(304, "MiB")},
		{"CPUCompute", m.CPUComputeBytes, unitToBytes(16.01, "MiB")},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %d (%s), want %d (%s)", c.name, c.got, HumanBytes(c.got), c.want, HumanBytes(c.want))
		}
	}
	if m.LayersOffloaded != 29 || m.LayersTotal != 29 {
		t.Errorf("layers = %d/%d, want 29/29", m.LayersOffloaded, m.LayersTotal)
	}
}

func TestParseServerLogRejectsGarbage(t *testing.T) {
	if _, err := ParseServerLog("this is not a llama-server log\n"); err == nil {
		t.Error("expected error on non-server log")
	}
}

func TestComputeScale(t *testing.T) {
	p := &Profile{}
	if got := p.ComputeScale(); got != 1.0 {
		t.Errorf("empty profile scale = %v, want 1.0", got)
	}
	p.AddSample(Sample{Model: "a", Ctx: 4096, PredCompute: 100, RealCompute: 150}) // 1.5
	p.AddSample(Sample{Model: "b", Ctx: 4096, PredCompute: 100, RealCompute: 120}) // 1.2
	if got := p.ComputeScale(); got != 1.5 {
		t.Errorf("scale = %v, want 1.5 (max ratio)", got)
	}
}

func TestObserveLoadMargin(t *testing.T) {
	p := &Profile{}
	p.ObserveLoadMargin(100 * mib) // below the default → ignored
	if p.LoadMargin != 0 {
		t.Fatalf("small residual must not set a margin below the default, got %d", p.LoadMargin)
	}
	p.ObserveLoadMargin(700 * mib)
	if p.LoadMargin != 700*mib {
		t.Fatalf("margin should ratchet to 700 MiB, got %d", p.LoadMargin)
	}
	p.ObserveLoadMargin(600 * mib) // smaller later observation → keep worst
	if p.LoadMargin != 700*mib {
		t.Fatalf("margin must never shrink, got %d", p.LoadMargin)
	}
	c := DefaultConfig()
	p.Apply(&c)
	if c.LoadMargin != 700*mib {
		t.Fatalf("Apply should carry the calibrated margin, got %d", c.LoadMargin)
	}
}

func TestAddSampleReplaces(t *testing.T) {
	p := &Profile{}
	p.AddSample(Sample{Model: "a", Ctx: 4096, PredCompute: 100, RealCompute: 150})
	p.AddSample(Sample{Model: "a", Ctx: 4096, PredCompute: 100, RealCompute: 110}) // same key → replace
	if len(p.Samples) != 1 {
		t.Fatalf("samples = %d, want 1 (replaced)", len(p.Samples))
	}
	if p.Samples[0].RealCompute != 110 {
		t.Errorf("RealCompute = %d, want 110 (latest)", p.Samples[0].RealCompute)
	}
}

// Calibration must be able to flip a too-optimistic prediction. A model that
// "fits" under the raw heuristic should OOM once a large compute correction
// from a real launch is applied.
func TestCalibrationFlipsVerdict(t *testing.T) {
	m := llama3ish()
	c := DefaultConfig()
	c.Ctx = 8192
	// VRAM sized so the raw prediction just fits all layers on GPU (incl. the
	// reserved load margin).
	hw := Hardware{FreeVRAM: 7*gib + 512*mib, FreeRAM: 1 * gib}

	raw, _ := Predict(m, hw, c)
	if raw.Mode != ModeGPU {
		t.Fatalf("precondition: raw should fit on GPU, got %s (vram used %s)", raw.Mode, HumanBytes(raw.VRAMUsed))
	}

	// A real launch revealed the compute buffer is ~4× our estimate.
	c.ComputeScale = 4.0
	cal, _ := Predict(m, hw, c)
	if cal.Mode == ModeGPU {
		t.Errorf("calibrated prediction should no longer fit fully on GPU, got %s", cal.Mode)
	}
}
