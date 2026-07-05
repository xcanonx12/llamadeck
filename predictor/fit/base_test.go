package fit

import "testing"

const twoGPULog = `
load_tensors:        CUDA0 model buffer size =   311.30 MiB
load_tensors:        CUDA1 model buffer size =   451.51 MiB
llama_kv_cache:      CUDA0 KV buffer size =   144.00 MiB
llama_kv_cache:      CUDA1 KV buffer size =   112.00 MiB
sched_reserve:      CUDA0 compute buffer size =   144.04 MiB
sched_reserve:      CUDA1 compute buffer size =   302.55 MiB
sched_reserve:  CUDA_Host compute buffer size =    72.05 MiB
`

func TestParseServerLogGPUDevices(t *testing.T) {
	m, err := ParseServerLog(twoGPULog)
	if err != nil {
		t.Fatal(err)
	}
	if m.GPUDevices != 2 {
		t.Errorf("GPUDevices = %d, want 2 (CUDA0, CUDA1; CUDA_Host is host)", m.GPUDevices)
	}
	// CUDA_Host compute must NOT count as GPU compute.
	wantGPUCompute := unitToBytes(144.04, "MiB") + unitToBytes(302.55, "MiB")
	if m.GPUComputeBytes != wantGPUCompute {
		t.Errorf("GPUComputeBytes = %d, want %d (excludes CUDA_Host)", m.GPUComputeBytes, wantGPUCompute)
	}
}

func TestCalibratedBasePerGPU(t *testing.T) {
	mm := &Measured{
		GPUModelBytes: 700 * mib, GPUKVBytes: 200 * mib, GPUComputeBytes: 300 * mib,
		GPUDevices: 2,
	}
	// buffers = 1200 MiB; process holds 2200 MiB ⇒ base = 1000 MiB over 2 GPUs = 500/GPU.
	got := CalibratedBasePerGPU(2200*mib, mm)
	if got != 500*mib {
		t.Errorf("base/GPU = %s, want 500 MiB", HumanBytes(got))
	}
	// Never negative if the process reports less than the buffers (measurement noise).
	if CalibratedBasePerGPU(500*mib, mm) != 0 {
		t.Error("expected 0 when procVRAM < buffers")
	}
}

func TestProfileApply(t *testing.T) {
	p := &Profile{BaseOverhead: 600 * mib}
	p.AddSample(Sample{Model: "m", Ctx: 4096, PredCompute: 100, RealCompute: 150}) // 1.5×
	c := DefaultConfig()
	p.Apply(&c)
	if c.ComputeScale != 1.5 {
		t.Errorf("ComputeScale = %v, want 1.5", c.ComputeScale)
	}
	if c.BaseOverhead != 600*mib {
		t.Errorf("BaseOverhead = %s, want 600 MiB", HumanBytes(c.BaseOverhead))
	}
}
