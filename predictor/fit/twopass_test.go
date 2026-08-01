package fit

import "testing"

// Abridged from a real DGX Spark launch (Qwen3-0.6B Q4_K_M, ctx 8192, 4 slots).
// Current llama.cpp logs a dry fit probe (load_mode = none) before the real load
// (load_mode = mmap); the probe's compute buffers carry REAL values, so summing
// the whole log counted them twice.
const twoPassLog = `
0.00.530.845 I llama_model_loader: loaded meta data with 32 key-value pairs and 310 tensors
0.00.655.738 I load_tensors: loading model tensors, this can take a while... (load_mode = none)
0.00.657.478 I load_tensors:        CUDA0 model buffer size =     0.00 MiB
0.00.657.479 I load_tensors:    CUDA_Host model buffer size =     0.00 MiB
0.00.660.612 I llama_kv_cache:      CUDA0 KV buffer size =     0.00 MiB
0.00.662.340 I sched_reserve:      CUDA0 compute buffer size =    34.01 MiB
0.00.662.342 I sched_reserve:  CUDA_Host compute buffer size =    12.01 MiB
0.00.821.181 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
0.00.837.841 I load_tensors:   CPU_Mapped model buffer size =   121.71 MiB
0.00.837.841 I load_tensors:        CUDA0 model buffer size =   372.65 MiB
0.01.084.293 I llama_kv_cache:      CUDA0 KV buffer size =   896.00 MiB
0.01.399.398 I sched_reserve:      CUDA0 compute buffer size =    34.01 MiB
0.01.399.399 I sched_reserve:  CUDA_Host compute buffer size =    12.01 MiB
`

// mibf mirrors unitToBytes' truncation for fractional MiB values.
func mibf(v float64) int64 { return int64(v * 1024 * 1024) }

func TestParseServerLogIgnoresFitProbePass(t *testing.T) {
	m, err := ParseServerLog(twoPassLog)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		got  int64
		want int64
	}{
		{"gpu compute", m.GPUComputeBytes, mibf(34.01)},
		{"cpu compute", m.CPUComputeBytes, mibf(12.01)},
		{"gpu model", m.GPUModelBytes, mibf(372.65)},
		{"cpu model", m.CPUModelBytes, mibf(121.71)},
		{"gpu kv", m.GPUKVBytes, int64(896 * mib)},
	} {
		if c.got != c.want {
			t.Errorf("%s = %s, want %s", c.name, HumanBytes(c.got), HumanBytes(c.want))
		}
	}
}

// A model with SWA logs two KV buffer lines for one device in a SINGLE pass;
// those must still sum, which is why the fix is pass-scoped, not a dedupe.
func TestParseServerLogSumsTwoKVCachesInOnePass(t *testing.T) {
	log := `
load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
load_tensors:        CUDA0 model buffer size =  1000.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   400.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   200.00 MiB
sched_reserve:      CUDA0 compute buffer size =    50.00 MiB
`
	m, err := ParseServerLog(log)
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(600 * mib); m.GPUKVBytes != want {
		t.Errorf("gpu kv = %s, want %s", HumanBytes(m.GPUKVBytes), HumanBytes(want))
	}
}

// The fit-planner capacity lines precede the final pass — trimming to that pass
// must not lose them.
func TestParseServerLogKeepsFitPlannerCapacity(t *testing.T) {
	log := `
llama_params_fit: - CUDA0 (NVIDIA GB10): 119000 total, 1400 used, 117000 free vs. target
load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
load_tensors:        CUDA0 model buffer size =   372.65 MiB
sched_reserve:      CUDA0 compute buffer size =    34.01 MiB
`
	m, err := ParseServerLog(log)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Devices) != 1 {
		t.Fatalf("devices = %d, want 1", len(m.Devices))
	}
	if want := int64((1400 + 117000) * mib); m.Devices[0].FreeBytes != want {
		t.Errorf("CUDA0 capacity = %s, want %s",
			HumanBytes(m.Devices[0].FreeBytes), HumanBytes(want))
	}
}
