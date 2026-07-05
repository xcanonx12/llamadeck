package fit

// Per-device VRAM simulation for `--gpus all`. The pooled Predict tells you the
// TOTAL fits; it can't tell you that ONE device OOMs while the pool has room.
// llama.cpp splits layers across devices, KV follows each layer to its device,
// and ONE device (the one holding the last layers) additionally carries the
// output head plus the big logits compute buffer. That asymmetry is what makes a
// device overflow on its own — the failure pooled prediction misses.
//
// This is ADDITIVE: Predict is untouched and remains the single-GPU path and the
// fallback. PredictDevices runs only when per-device free VRAM is known (>1 dev).

import "fmt"

// DevicePrediction is the simulated footprint on one GPU.
type DevicePrediction struct {
	Index  int
	Layers int   // transformer layers placed on this device
	Used   int64 // predicted bytes used
	Free   int64 // bytes free on this device
	Main   bool  // carries the output head + the big logits buffer
	OOM    bool  // Used > Free
	OverBy int64 // Used-Free when OOM, else 0
	Tight  bool  // fits, but inside the load-time margin — may crash at load

	// Composition (sums to Used) — the graph renders these as colored segments,
	// same buckets as the pooled VRAM bar.
	WeightBytes  int64 // layer weights (+ output head on the main device)
	KVBytes      int64 // KV cache + recurrent state of the layers placed here
	ComputeBytes int64 // activation term (+ logits buffer on the main device)
	BaseBytes    int64 // CUDA context overhead (paid once the device is active)
}

// DeviceResult is the per-device breakdown for an `--gpus all` launch attempt
// (full offload). Bottleneck/OverBy point at the device that overflows first.
type DeviceResult struct {
	Devices    []DevicePrediction
	Bottleneck int   // index into Devices of the worst device; -1 if all fit
	OverBy     int64 // bytes the bottleneck device is over by (0 if it fits)
	Fits       bool  // every device fits
	Tight      bool  // fits, but some device is inside the load-time margin
	TightBy    int64 // worst shortfall vs the margin across tight devices
	Paged      bool  // mmap'd weights exceed free RAM — runs, paging from disk
	RAMUsed    int64 // host RAM (token-embedding copy) regardless of split
	Mode       Mode  // ModeGPU when it fits, ModeOOM otherwise
}

// computeParts splits the compute buffer into the logits term (present ONLY on
// the device holding the output head) and the activation+fixed term (present on
// every participating device). Sum == computeBuffer, so the pooled path is
// unaffected. ComputeScale (calibration) applies to both, same as computeBuffer.
func computeParts(m *Model, c Config) (logits, perDevice int64) {
	scale := c.ComputeScale
	if scale <= 0 {
		scale = 1.0
	}
	logits = int64(float64(int64(c.UBatch)*int64(m.VocabSize)*4) * scale)
	perDevice = int64(float64(int64(c.UBatch)*int64(m.EmbedLength)*8+c.ComputeFixed) * scale)
	return
}

// offloadExperts implements --n-cpu-moe: it moves the expert (FFN) weight of the
// first c.NCPUMoE layers off the GPU, subtracting each layer's expert bytes from
// perLayerW in place and returning the total shifted to host RAM. No-op unless
// the model is MoE with a parsed expert table (dense models have zero experts).
func offloadExperts(m *Model, perLayerW []int64, c Config) int64 {
	if c.NCPUMoE <= 0 || m.Tensors == nil || len(m.Tensors.ExpertPerLayer) != m.NLayers {
		return 0
	}
	n := c.NCPUMoE
	if n > m.NLayers {
		n = m.NLayers
	}
	var moved int64
	for i := 0; i < n; i++ {
		e := m.Tensors.ExpertPerLayer[i]
		if e > perLayerW[i] {
			e = perLayerW[i]
		}
		perLayerW[i] -= e
		moved += e
	}
	return moved
}

// PredictDevices simulates an `--gpus all` launch across the devices whose free
// VRAM is given in hw.GPUsFree. Crucially it offloads only the layers that
// ACTUALLY fit (or the explicit -ngl), taken from the pooled prediction so the
// per-device bars agree with the VRAM/RAM gauge — it does NOT force a full
// offload. It then splits those layers across devices, adds the main device's
// extra output-head + logits buffer, and flags any device that overflows on its
// own (the failure a pooled total hides). Returns an error (caller falls back to
// pooled Predict) when fewer than 2 devices have known free VRAM.
func PredictDevices(m *Model, hw Hardware, c Config) (*DeviceResult, error) {
	if m.NLayers <= 0 || m.HeadDim <= 0 || m.VocabSize <= 0 {
		return nil, fmt.Errorf("incomplete model metadata (layers=%d head_dim=%d vocab=%d)",
			m.NLayers, m.HeadDim, m.VocabSize)
	}
	if len(hw.GPUsFree) < 2 {
		return nil, fmt.Errorf("per-device prediction needs ≥2 devices' free VRAM")
	}

	// Layer count + host RAM come from the pooled prediction (auto-fit or explicit
	// -ngl, N× per-GPU overhead) so this view can't contradict the gauge.
	var pooledFree int64
	for _, f := range hw.GPUsFree {
		if f > 0 {
			pooledFree += f
		}
	}
	pooled, err := Predict(m, Hardware{FreeVRAM: pooledFree, FreeRAM: hw.FreeRAM, NumGPUs: len(hw.GPUsFree)}, c)
	if err != nil {
		return nil, err
	}
	onGPU := pooled.LayersOnGPU

	// Per-layer memory-state cost (KV for attention layers, recurrent state for
	// SSM layers) — hybrids are not uniform, and llama.cpp offloads the LAST
	// onGPU blocks, so which blocks land on the GPU matters.
	stateAt := make([]int64, m.NLayers)
	for i := range stateAt {
		stateAt[i] = stateAtLayer(m, c, i)
	}
	gpuStart := m.NLayers - onGPU
	logits, computePerDev := computeParts(m, c)

	// Per-layer weights: exact from the tensor table when available, else uniform.
	perLayerW := make([]int64, m.NLayers)
	var outputBytes int64
	if ts := m.Tensors; ts != nil && len(ts.PerLayer) == m.NLayers {
		outputBytes = ts.Output
		copy(perLayerW, ts.PerLayer)
	} else {
		pw := m.FileBytes / int64(m.NLayers)
		for i := range perLayerW {
			perLayerW[i] = pw
		}
	}
	offloadExperts(m, perLayerW, c) // reduce GPU weight for MoE (RAM already in pooled)
	if outputBytes == 0 && m.Tensors != nil {
		outputBytes = m.Tensors.Embedding // tied embeddings serve as the output head
	}

	// Two very different split regimes, matching llama.cpp:
	//
	//   auto -ngl (c.NGL==0): llama_params_fit load-balances — the main device
	//   (last) reserves its output-head + logits buffer up front, then each layer
	//   lands on whichever device has the most room left. This pushes layers
	//   AWAY from the main device (validated: predicted 13/1 == real auto-fit).
	//
	//   explicit -ngl: NO auto-fit. llama.cpp splits the offloaded blocks
	//   proportionally to each device's free VRAM, sequentially (device 0 gets
	//   the first share, the last device the final share + output + logits) —
	//   blind to the compute buffer, so the main device CAN overload. That's the
	//   OOM this simulation exists to catch (validated on a real -ngl 19 launch:
	//   the same proportional formula reproduced the measured 11/7 block split).
	n := len(hw.GPUsFree)
	main := n - 1
	used := make([]int64, n)
	counts := make([]int, n)
	active := make([]bool, n) // a device only pays base+compute once it holds a layer
	// Per-device composition, so the graph can render the same colored buckets
	// as the pooled bar (weights / kv+rs / compute / base).
	wts := make([]int64, n)
	kvs := make([]int64, n)
	cmp := make([]int64, n)
	base := make([]int64, n)
	activation := c.BaseOverhead + computePerDev
	activate := func(i int) {
		if !active[i] {
			used[i] += activation
			base[i] = c.BaseOverhead
			cmp[i] += computePerDev
			active[i] = true
		}
	}
	if onGPU > 0 {
		activate(main)
		used[main] += outputBytes + logits // output must ride a GPU
		wts[main] += outputBytes
		cmp[main] += logits
	}
	if c.NGL > 0 {
		// Proportional-to-free sequential split (llama.cpp's default tensor
		// split): offloaded block b of onGPU goes to the first device whose
		// cumulative free-VRAM fraction exceeds b/onGPU.
		var totalFree int64
		for _, f := range hw.GPUsFree {
			if f > 0 {
				totalFree += f
			}
		}
		for b := 0; b < onGPU; b++ {
			l := gpuStart + b
			if l < 0 || l >= m.NLayers {
				continue
			}
			dev, cum := main, int64(0)
			for i := 0; i < n; i++ {
				if hw.GPUsFree[i] > 0 {
					cum += hw.GPUsFree[i]
				}
				if int64(b)*totalFree < int64(onGPU)*cum {
					dev = i
					break
				}
			}
			activate(dev)
			used[dev] += perLayerW[l] + stateAt[l]
			wts[dev] += perLayerW[l]
			kvs[dev] += stateAt[l]
			counts[dev]++
		}
	} else {
		for b := 0; b < onGPU; b++ {
			l := gpuStart + b
			if l < 0 || l >= m.NLayers {
				continue
			}
			cost := perLayerW[l] + stateAt[l]
			best, bestRem := -1, int64(0)
			for i := 0; i < n; i++ {
				rem := hw.GPUsFree[i] - used[i]
				if !active[i] {
					rem -= activation // pay to switch this device on
				}
				if best < 0 || rem > bestRem {
					best, bestRem = i, rem
				}
			}
			activate(best)
			used[best] += cost
			wts[best] += perLayerW[l]
			kvs[best] += stateAt[l]
			counts[best]++
		}
	}

	margin := loadMargin(c)
	devs := make([]DevicePrediction, n)
	for i := 0; i < n; i++ {
		free := hw.GPUsFree[i]
		d := DevicePrediction{Index: i, Layers: counts[i], Free: free, Used: used[i], Main: active[i] && i == main,
			WeightBytes: wts[i], KVBytes: kvs[i], ComputeBytes: cmp[i], BaseBytes: base[i]}
		if d.Used > free {
			d.OOM, d.OverBy = true, d.Used-free
		} else if active[i] && free-d.Used < margin {
			d.Tight = true
		}
		devs[i] = d
	}

	res := &DeviceResult{
		Devices:    devs,
		Bottleneck: -1,
		Fits:       true,
		Tight:      pooled.Tight, // inherits the auto-hybrid "may crash" signal
		Paged:      pooled.Paged,
		RAMUsed:    pooled.RAMUsed, // consistent with the RAM gauge
		Mode:       pooled.Mode,
	}
	for i, d := range devs {
		if d.OverBy > res.OverBy {
			res.Bottleneck, res.OverBy = i, d.OverBy
		}
		if d.OOM {
			res.Fits = false
			res.Mode = ModeOOM
		}
		if d.Tight {
			res.Tight = true
			if short := margin - (d.Free - d.Used); short > res.TightBy {
				res.TightBy = short
				if res.Bottleneck < 0 {
					res.Bottleneck = i
				}
			}
		}
	}
	return res, nil
}
