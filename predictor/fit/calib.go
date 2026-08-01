package fit

// Calibration closes the loop: launch a model, read the REAL buffer sizes that
// llama-server prints at load time, compare them to what the engine predicted,
// and persist a correction for the fuzzy compute-buffer estimate. Ground truth
// comes from the server's own log — no nvidia-smi guesswork.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Measured holds the real footprint extracted from a llama-server log.
type Measured struct {
	GPUModelBytes   int64
	CPUModelBytes   int64
	GPUKVBytes      int64
	CPUKVBytes      int64
	GPUComputeBytes int64
	CPUComputeBytes int64
	LayersOffloaded int
	LayersTotal     int
	GPUDevices      int              // distinct accelerator devices seen (for per-GPU base calc)
	Devices         []DeviceMeasured // per-accelerator buffers, in first-seen order (for per-device verify)
}

// DeviceMeasured is the real per-accelerator footprint (one CUDA<i>/ROCm/etc).
type DeviceMeasured struct {
	Name         string
	ModelBytes   int64
	KVBytes      int64
	ComputeBytes int64
	FreeBytes    int64 // VRAM capacity at fit-planning time (projected used+free from llama_params_fit_impl)
}

func (d DeviceMeasured) Total() int64 { return d.ModelBytes + d.KVBytes + d.ComputeBytes }

func (m Measured) TotalCompute() int64 { return m.GPUComputeBytes + m.CPUComputeBytes }
func (m Measured) TotalKV() int64      { return m.GPUKVBytes + m.CPUKVBytes }
func (m Measured) TotalModel() int64   { return m.GPUModelBytes + m.CPUModelBytes }

var (
	// e.g. "CUDA0 model buffer size =  4156.00 MiB", "CPU_Mapped model buffer size = 281.81 MiB",
	// "llama_memory_recurrent: CUDA0 RS buffer size = 41.88 MiB" (hybrid/SSM models).
	// RS (recurrent state) counts into the KV bucket — the engine's KVBytes
	// prediction includes recurrent state, so measured must too.
	bufRe = regexp.MustCompile(`(?i)(\w+)\s+(model|KV|compute|RS)\s+buffer size\s*=\s*([0-9.]+)\s*([KMG]i?B)`)
	// e.g. "offloaded 29/29 layers to GPU"
	offloadRe = regexp.MustCompile(`(?i)offloaded\s+(\d+)\s*/\s*(\d+)\s+layers`)
	// e.g. "- CUDA0 (NVIDIA GeForce RTX 3080):  9875 total,  535 used,  2693 free vs. target"
	// (values are MiB). The "(model name)" and trailing ":" are optional.
	// IMPORTANT semantics: "used" is the PROJECTED launch usage and "free" the
	// projected REMAINDER — capacity at fit-planning is used+free (both captured).
	paramsFitRe = regexp.MustCompile(`(?i)(\w+)\s*(?:\([^)]*\))?\s*:?\s*\d+\s+total,\s*(\d+)\s+used,\s*(\d+)\s+free`)
)

func unitToBytes(n float64, unit string) int64 {
	switch strings.ToLower(unit) {
	case "b":
		return int64(n)
	case "kib", "kb":
		return int64(n * 1024)
	case "mib", "mb":
		return int64(n * 1024 * 1024)
	case "gib", "gb":
		return int64(n * 1024 * 1024 * 1024)
	}
	return int64(n)
}

func isGPUDevice(dev string) bool {
	d := strings.ToUpper(dev)
	// "CPU", and host-side accelerator buffers like "CUDA_Host", are host memory.
	if strings.Contains(d, "CPU") || strings.Contains(d, "HOST") {
		return false
	}
	// CUDA, ROCm/HIP, Vulkan, SYCL, Metal — anything else is an accelerator.
	return true
}

// lastLoadPass trims the log to the final "loading model tensors" pass.
//
// Current llama.cpp probes the fit with a dry pass (load_mode = none) before the
// real load (load_mode = mmap), and BOTH passes log buffer sizes. The probe's
// model/KV lines read 0.00 MiB (harmless when summed) but its compute lines
// carry the real values, so summing every match in the log double-counted the
// compute buffers — measured on a DGX Spark launch: 92 MiB reported for a
// 34+12 MiB reality, which would teach calibrate a ~2x ComputeScale.
//
// Splitting by pass rather than deduping by device+kind is deliberate: within
// ONE pass a model may legitimately log two buffer lines for the same device
// (SWA models keep a second KV cache), and those must still sum.
// Logs from builds without the marker (older llama.cpp, the testdata fixtures)
// are returned whole.
func lastLoadPass(log string) string {
	const marker = "loading model tensors"
	i := strings.LastIndex(log, marker)
	if i <= 0 {
		return log
	}
	if j := strings.LastIndexByte(log[:i], '\n'); j >= 0 {
		return log[j+1:]
	}
	return log[i:]
}

// ParseServerLog extracts the real buffer sizes from llama-server stderr.
func ParseServerLog(log string) (*Measured, error) {
	// Only the buffer sizes are pass-scoped: the fit-planner capacity lines
	// (paramsFitRe) are emitted BEFORE the final load pass, so they must still be
	// searched across the whole log.
	bufLog := lastLoadPass(log)
	m := &Measured{}
	found := false
	devices := map[string]bool{}
	devIdx := map[string]int{} // GPU name → index into m.Devices (first-seen order)
	perDev := func(name string) *DeviceMeasured {
		key := strings.ToUpper(name)
		if i, ok := devIdx[key]; ok {
			return &m.Devices[i]
		}
		devIdx[key] = len(m.Devices)
		m.Devices = append(m.Devices, DeviceMeasured{Name: key})
		return &m.Devices[len(m.Devices)-1]
	}
	for _, mt := range bufRe.FindAllStringSubmatch(bufLog, -1) {
		dev, kind, numStr, unit := mt[1], strings.ToLower(mt[2]), mt[3], mt[4]
		n, err := strconv.ParseFloat(numStr, 64)
		if err != nil {
			continue
		}
		bytes := unitToBytes(n, unit)
		gpu := isGPUDevice(dev)
		if gpu {
			devices[strings.ToUpper(dev)] = true
		}
		found = true
		var d *DeviceMeasured
		if gpu {
			d = perDev(dev)
		}
		switch kind {
		case "model":
			if gpu {
				m.GPUModelBytes += bytes
				d.ModelBytes += bytes
			} else {
				m.CPUModelBytes += bytes
			}
		case "kv", "rs":
			if gpu {
				m.GPUKVBytes += bytes
				d.KVBytes += bytes
			} else {
				m.CPUKVBytes += bytes
			}
		case "compute":
			if gpu {
				m.GPUComputeBytes += bytes
				d.ComputeBytes += bytes
			} else {
				m.CPUComputeBytes += bytes
			}
		}
	}
	// Per-device VRAM capacity at fit-planning (MiB), merged by device name.
	// The planner line reports projected used + projected remainder; capacity is
	// their sum (slightly below nvidia-smi free — the CUDA context is already
	// allocated by then, which our BaseOverhead models — so it's conservative).
	for _, mt := range paramsFitRe.FindAllStringSubmatch(log, -1) {
		if !isGPUDevice(mt[1]) {
			continue
		}
		usedMiB, err1 := strconv.ParseFloat(mt[2], 64)
		freeMiB, err2 := strconv.ParseFloat(mt[3], 64)
		if err1 != nil || err2 != nil {
			continue
		}
		perDev(mt[1]).FreeBytes = unitToBytes(usedMiB+freeMiB, "mib")
	}
	if ofl := offloadRe.FindStringSubmatch(log); ofl != nil {
		m.LayersOffloaded, _ = strconv.Atoi(ofl[1])
		m.LayersTotal, _ = strconv.Atoi(ofl[2])
	}
	if !found {
		return nil, fmt.Errorf("no buffer-size lines found in log (is this llama-server output?)")
	}
	m.GPUDevices = len(devices)
	return m, nil
}

// --- calibration profile (persisted) --------------------------------------

// Sample records one observed launch: what we predicted vs what really happened.
type Sample struct {
	Model       string `json:"model"`
	Ctx         int    `json:"ctx"`
	PredCompute int64  `json:"pred_compute"`
	RealCompute int64  `json:"real_compute"`
	PredKV      int64  `json:"pred_kv"`
	RealKV      int64  `json:"real_kv"`
	PredWeights int64  `json:"pred_weights"`
	RealWeights int64  `json:"real_weights"`
}

// Profile is the persisted calibration state for this host.
type Profile struct {
	Samples      []Sample `json:"samples"`
	BaseOverhead int64    `json:"base_overhead,omitempty"` // per-GPU CUDA context, calibrated
	// LoadMargin is the per-GPU load-time headroom observed on this host: the
	// worst measured under-prediction across calibrated launches. Only ever
	// raised above DefaultLoadMargin (worst-case-wins, like ComputeScale).
	LoadMargin int64  `json:"load_margin,omitempty"`
	HFToken    string `json:"hf_token,omitempty"` // Hugging Face access token for gated/auth repos
}

// CalibratedBasePerGPU derives the per-GPU base/context overhead from a launch:
// the process's total GPU memory minus the model + KV + compute buffers it
// reported, divided by the number of GPUs it used.
func CalibratedBasePerGPU(procVRAM int64, mm *Measured) int64 {
	buffers := mm.GPUModelBytes + mm.GPUKVBytes + mm.GPUComputeBytes
	base := procVRAM - buffers
	if base < 0 {
		base = 0
	}
	n := int64(mm.GPUDevices)
	if n < 1 {
		n = 1
	}
	return base / n
}

// Apply sets the calibrated corrections (compute scale, base overhead, load
// margin) on a Config.
func (p *Profile) Apply(c *Config) {
	c.ComputeScale = p.ComputeScale()
	if p.BaseOverhead > 0 {
		c.BaseOverhead = p.BaseOverhead
	}
	if p.LoadMargin > DefaultLoadMargin {
		c.LoadMargin = p.LoadMargin
	}
}

// ObserveLoadMargin records a measured per-device under-prediction (real −
// predicted, bytes). The persisted margin only ratchets upward, and only past
// the measured default — a good run never shrinks the safety net.
func (p *Profile) ObserveLoadMargin(underBy int64) {
	if underBy > p.LoadMargin && underBy > DefaultLoadMargin {
		p.LoadMargin = underBy
	}
}

// ComputeScale is the correction applied to the compute-buffer estimate. We take
// the largest real/predicted ratio across all samples, so the calibrated
// prediction stays safe across every model we've actually observed.
func (p *Profile) ComputeScale() float64 {
	scale := 1.0
	for _, s := range p.Samples {
		if s.PredCompute > 0 {
			if r := float64(s.RealCompute) / float64(s.PredCompute); r > scale {
				scale = r
			}
		}
	}
	return scale
}

// AddSample appends a sample, replacing any prior one for the same model+ctx.
func (p *Profile) AddSample(s Sample) {
	for i, old := range p.Samples {
		if old.Model == s.Model && old.Ctx == s.Ctx {
			p.Samples[i] = s
			return
		}
	}
	p.Samples = append(p.Samples, s)
}

// ProfilePath returns the on-disk location of the calibration profile.
func ProfilePath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "llamadeck", "profile.json"), nil
}

// LoadProfile reads the saved profile, returning an empty one if none exists.
func LoadProfile() (*Profile, error) {
	path, err := ProfilePath()
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &Profile{}, nil
	}
	if err != nil {
		return nil, err
	}
	var p Profile
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// Save writes the profile to disk, creating the config directory as needed.
func (p *Profile) Save() error {
	path, err := ProfilePath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600) // may hold an HF token
}
