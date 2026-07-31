// llamadeck predicts whether a GGUF model fits the local hardware, and in what
// mode (GPU / hybrid / CPU / OOM), without downloading the model.
// Subcommands: predict (default), tui, recommend, calibrate.
package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"llamadeck/app"
	"llamadeck/fit"
	"llamadeck/hub"
	"llamadeck/infra"
	"llamadeck/tui"
)

// version is set at release time via -ldflags "-X main.version=...".
var version = "dev"

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  llamadeck                                                          launch the full TUI (Models/Monitor/Fit/Update)
  llamadeck <owner/repo[:QUANT] | model.gguf | https://...> [flags]   predict fit
  llamadeck tui <same source forms> [flags]                           interactive view
  llamadeck recommend <owner/repo> [flags]                            scan quants, pick the best
  llamadeck calibrate <same source forms> [flags]                     learn from a real launch
  llamadeck verify <same source forms> [flags]                        report prediction accuracy vs a real launch
  llamadeck audit <same source forms> [flags]                         per-bucket pass/fail vs a real launch (pinpoints drift)
  llamadeck plug [agent] [flags]                                      add a running server to your coding agent (see plug --help)

flags:
  --ctx N        context size (default 4096)
  --ubatch N     physical batch, drives compute buffer (default 512)
  --kv TYPE      kv cache type: f16|q8_0|q4_0 (default f16)
  --ngl N        explicit GPU layer count to model (0 = auto best fit)
  --parallel N   server request slots; scales hybrid recurrent state (default 4)
  --no-mmap      model weights as anonymous RAM (hard requirement, no paging)
  --mlock        pin weights in RAM (also a hard requirement)
  --vram-mb N    free VRAM override (total across GPUs); 0 = probe nvidia-smi
  --ram-mb N     free RAM override; 0 = probe /proc/meminfo (default 0)
  --gpus N       number of GPUs to model; 0 = probe (default 0)
  --cpu          model a CPU-only host (no GPU offload; RAM-only fit)

calibrate-only:
  --log FILE     llama-server log to read ('-' for stdin)
  --container N  read logs via 'docker logs <N>' instead of --log`)
	os.Exit(2)
}

// opts holds parsed command-line state shared by both subcommands.
type opts struct {
	src           string
	cfg           fit.Config
	vramMB, ramMB int64
	gpus          int
	logFile       string
	container     string
	cpu           bool
}

// hardware builds the Hardware to predict against: probed GPU count + pooled free
// VRAM and free RAM, with --vram-mb / --ram-mb / --gpus overrides.
func (o opts) hardware() fit.Hardware {
	count, total, unified := infra.GPUSummary()
	hw := fit.Hardware{FreeVRAM: total, FreeRAM: infra.FreeRAM(), NumGPUs: count, Unified: unified}
	if o.gpus > 0 {
		// Model a specific GPU count; scale pooled VRAM proportionally (assuming
		// uniform GPUs) unless an explicit --vram-mb override is given.
		if o.vramMB == 0 && count > 0 {
			hw.FreeVRAM = (total / int64(count)) * int64(o.gpus)
		}
		hw.NumGPUs = o.gpus
	}
	if o.vramMB != 0 {
		hw.FreeVRAM = o.vramMB << 20
	}
	if o.ramMB != 0 {
		hw.FreeRAM = o.ramMB << 20
	}
	if o.cpu {
		hw.FreeVRAM, hw.NumGPUs, hw.GPUsFree = 0, 0, nil
	}
	return hw
}

func parseArgs(args []string) opts {
	if len(args) == 0 {
		usage()
	}
	o := opts{src: args[0], cfg: fit.DefaultConfig()}
	rest := args[1:]
	for i := 0; i < len(rest); i++ {
		next := func() string {
			i++
			if i >= len(rest) {
				die("flag %s needs a value", rest[i-1])
			}
			return rest[i]
		}
		switch rest[i] {
		case "--ctx":
			o.cfg.Ctx = atoi(next())
		case "--ubatch":
			o.cfg.UBatch = atoi(next())
		case "--kv":
			o.cfg.KVType = next()
		case "--n-cpu-moe":
			o.cfg.NCPUMoE = atoi(next())
		case "--ngl":
			o.cfg.NGL = atoi(next())
		case "--parallel":
			o.cfg.NSeqs = atoi(next())
		case "--no-mmap":
			o.cfg.NoMmap = true
		case "--mlock":
			o.cfg.Mlock = true
		case "--cpu":
			o.cpu = true
		case "--vram-mb":
			o.vramMB = int64(atoi(next()))
		case "--ram-mb":
			o.ramMB = int64(atoi(next()))
		case "--gpus":
			o.gpus = atoi(next())
		case "--log":
			o.logFile = next()
		case "--container":
			o.container = next()
		case "-h", "--help":
			usage()
		default:
			die("unknown flag: %s", rest[i])
		}
	}
	return o
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		// Bare invocation on a terminal launches the full operational TUI.
		if isTTY(os.Stdout) {
			if err := app.Run(); err != nil {
				die("%v", err)
			}
			return
		}
		usage()
	}
	switch args[0] {
	case "version", "-v", "--version":
		fmt.Println("llamadeck", version)
	case "app":
		if err := app.Run(); err != nil {
			die("%v", err)
		}
	case "calibrate":
		runCalibrate(parseArgs(args[1:]))
	case "tui":
		runTUI(parseArgs(args[1:]))
	case "recommend":
		runRecommend(parseArgs(args[1:]))
	case "verify":
		runVerify(parseArgs(args[1:]))
	case "audit":
		runAudit(parseArgs(args[1:]))
	case "plug":
		runPlug(args[1:])
	default:
		runPredict(parseArgs(args))
	}
}

// runAudit judges every memory bucket of the prediction against a real launch
// (log or container) independently, with per-bucket tolerances — pinpointing
// WHICH part of the math drifted (weights / kv+rs / compute / per-device
// placement / verdict) instead of one aggregate error.
func runAudit(o opts) {
	logText, err := readLog(o)
	if err != nil {
		die("%v", err)
	}
	mm, err := fit.ParseServerLog(logText)
	if err != nil {
		die("%v", err)
	}
	m, err := loadModel(o.src)
	if err != nil {
		die("%v", err)
	}
	hw := o.hardware()
	// Prefer the free-VRAM the server itself saw at load (params_fit lines).
	var free []int64
	for _, d := range mm.Devices {
		if d.FreeBytes > 0 {
			free = append(free, d.FreeBytes)
		}
	}
	if len(free) > 1 && len(free) == len(mm.Devices) {
		hw.GPUsFree = free
	}
	if p, err := fit.LoadProfile(); err == nil {
		p.Apply(&o.cfg)
	}
	outcome := "ran"
	if strings.Contains(logText, "cudaMalloc failed") || strings.Contains(logText, "failed to allocate") {
		outcome = "crashed"
	}

	fmt.Printf("audit: %s  (outcome: %s)\n\n", o.src, outcome)
	fmt.Printf("  %-18s %12s %12s   %s\n", "bucket", "PRED", "REAL", "verdict")
	bad := 0
	for _, ch := range fit.AuditLaunch(m, hw, o.cfg, mm, outcome) {
		mark := "✓"
		if !ch.OK {
			mark, bad = "✗ "+ch.Detail, bad+1
		}
		fmt.Printf("  %-18s %12s %12s   %s\n", ch.Name, fit.HumanBytes(ch.Pred), fit.HumanBytes(ch.Real), mark)
	}
	if bad > 0 {
		fmt.Printf("\n%d bucket(s) drifted — that's where the math starts to lie.\n", bad)
		os.Exit(1)
	}
	fmt.Println("\nall buckets within tolerance.")
}

// runVerify compares the prediction to a real llama-server launch log and reports
// per-bucket error, without persisting anything (unlike calibrate). The manual
// "can I trust this on my box?" check.
func runVerify(o opts) {
	logText, err := readLog(o)
	if err != nil {
		die("%v", err)
	}
	measured, err := fit.ParseServerLog(logText)
	if err != nil {
		die("%v", err)
	}
	m, err := loadModel(o.src)
	if err != nil {
		die("%v", err)
	}
	o.cfg.ComputeScale = 1.0 // report the raw heuristic's accuracy
	pred, err := fit.Predict(m, fit.Hardware{FreeVRAM: 1 << 62, FreeRAM: 1 << 62}, o.cfg)
	if err != nil {
		die("%v", err)
	}

	fmt.Printf("accuracy for %s @ ctx %d\n\n", o.src, o.cfg.Ctx)
	fmt.Printf("  %-14s %12s %12s %10s\n", "bucket", "PREDICTED", "REAL", "ERROR")
	worst := 0.0
	row := func(label string, p, r int64) {
		var errPct float64
		if r > 0 {
			errPct = (float64(p) - float64(r)) / float64(r) * 100
		}
		abs := errPct
		if abs < 0 {
			abs = -abs
		}
		if abs > worst {
			worst = abs
		}
		flag := "ok"
		if abs > 15 {
			flag = "WARN"
		}
		fmt.Printf("  %-14s %12s %12s %+9.1f%%  %s\n",
			label, fit.HumanBytes(p), fit.HumanBytes(r), errPct, flag)
	}
	// Weights prediction targets VRAM-resident weights, so compare against the
	// GPU model buffer. llama.cpp also keeps some tensors (token embeddings /
	// output) on the host — shown separately, as it lands in RAM not VRAM.
	realGPUWeights := measured.GPUModelBytes
	if realGPUWeights == 0 {
		realGPUWeights = measured.TotalModel() // CPU-only run
	}
	row("weights (VRAM)", pred.WeightsBytes, realGPUWeights)
	row("kv cache", pred.KVBytes, measured.TotalKV())
	row("compute", pred.ComputeBytes, measured.TotalCompute())
	fmt.Printf("\nworst-bucket error: %.1f%%  (run 'calibrate' to correct the compute estimate)\n", worst)
	if measured.CPUModelBytes > 0 {
		fmt.Printf("note: + %s model buffer on host (embeddings/output) — counts toward RAM, not VRAM\n",
			fit.HumanBytes(measured.CPUModelBytes))
	}
	verifyPerDevice(m, measured, o.cfg)
}

// verifyPerDevice compares the per-device split prediction (PredictDevices) to
// the real per-accelerator buffers, when the log came from a multi-GPU launch
// that reported each device's free VRAM. Skipped silently otherwise.
func verifyPerDevice(m *fit.Model, measured *fit.Measured, cfg fit.Config) {
	var free []int64
	for _, d := range measured.Devices {
		if d.FreeBytes <= 0 {
			return // no params_fit free info → can't drive the split prediction
		}
		free = append(free, d.FreeBytes)
	}
	if len(free) < 2 {
		return
	}
	dr, err := fit.PredictDevices(m, fit.Hardware{GPUsFree: free}, cfg)
	if err != nil {
		return
	}
	fmt.Printf("\nper-device split (--gpus all)\n\n")
	fmt.Printf("  %-8s %12s %12s %10s %12s\n", "device", "PRED USED", "REAL USED", "ERROR", "RESIDUAL")
	for i, d := range dr.Devices {
		real := measured.Devices[i].Total()
		var errPct float64
		if real > 0 {
			errPct = (float64(d.Used) - float64(real)) / float64(real) * 100
		}
		tag := ""
		if d.Main {
			tag = " main"
		}
		// Residual = real − predicted. Positive means we UNDER-predicted — the
		// load-time gap the safety margin must cover.
		resid := real - d.Used
		sign := "+"
		if resid < 0 {
			sign, resid = "-", -resid
		}
		fmt.Printf("  %-8s %12s %12s %+9.1f%% %11s%s\n",
			measured.Devices[i].Name, fit.HumanBytes(d.Used), fit.HumanBytes(real), errPct,
			sign+fit.HumanBytes(resid), tag)
	}
}

// runRecommend scans every quant in a repo and reports which fit, the largest
// all-GPU context each allows, and the verdict at the requested context — then
// recommends the best download. Structural keys are shared across quants, so we
// parse one header and only HEAD each quant for its size.
func runRecommend(o opts) {
	ref, ok := fit.ParseRef(o.src)
	if !ok {
		die("recommend needs a Hugging Face repo (owner/repo), not a URL or file")
	}
	hw := o.hardware()
	if p, err := fit.LoadProfile(); err == nil {
		p.Apply(&o.cfg)
	}

	fmt.Fprintf(os.Stderr, "scanning %s (free: %s VRAM, %s RAM, @ctx %d)...\n",
		ref.Repo, fit.HumanBytes(hw.FreeVRAM), fit.HumanBytes(hw.FreeRAM), o.cfg.Ctx)

	rows, pick, err := hub.QuantReport(ref.Repo, hw, o.cfg)
	if err != nil {
		die("%v", err)
	}

	fmt.Printf("\n%-10s %10s  %12s  %s\n", "QUANT", "SIZE", "MAX GPU CTX", "AT CTX "+itoa(o.cfg.Ctx))
	for i, r := range rows {
		marker := "  "
		if i == pick {
			marker = "» "
		}
		maxctx := "—"
		if r.MaxGPU > 0 {
			maxctx = itoa(r.MaxGPU)
		}
		fmt.Printf("%s%-8s %10s  %12s  %s\n", marker, r.Quant, fit.HumanBytes(r.Size), maxctx, colorMode(r.Mode))
	}
	if pick >= 0 {
		why := "largest quant fully on GPU"
		if rows[pick].Mode != fit.ModeGPU {
			why = "largest quant that avoids OOM (no quant fits fully on GPU)"
		}
		fmt.Printf("\n» recommend: %s — %s at ctx %d\n", rows[pick].Quant, why, o.cfg.Ctx)
	} else {
		fmt.Printf("\nNo quant avoids OOM at ctx %d on this hardware.\n", o.cfg.Ctx)
	}
}

func itoa(n int) string { return strconv.Itoa(n) }

// runTUI launches the interactive view (static render when stdout isn't a TTY).
func runTUI(o opts) {
	m, err := loadModel(o.src)
	if err != nil {
		die("%v", err)
	}
	hw := o.hardware()
	if p, err := fit.LoadProfile(); err == nil {
		p.Apply(&o.cfg)
	}
	source, quant := o.src, ""
	if ref, ok := fit.ParseRef(o.src); ok {
		source, quant = ref.Repo, ref.Quant
	}
	st := tui.State{Source: source, Quant: quant, Model: m, HW: hw, Cfg: o.cfg}
	st.Recompute()

	if isTTY(os.Stdout) {
		if err := tui.Run(st); err != nil {
			die("%v", err)
		}
		return
	}
	fmt.Print(tui.RenderView(st)) // non-TTY: print one static frame
}

func isTTY(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func runPredict(o opts) {
	m, err := loadModel(o.src)
	if err != nil {
		die("%v", err)
	}

	hw := o.hardware()

	// Apply any learned calibration to the fuzzy compute estimate.
	if p, err := fit.LoadProfile(); err == nil {
		p.Apply(&o.cfg)
	}

	r, err := fit.Predict(m, hw, o.cfg)
	if err != nil {
		die("%v", err)
	}
	report(m, hw, o.cfg, r)
}

// runCalibrate parses a real llama-server log, compares the measured footprint
// to the raw (uncalibrated) prediction, and persists the correction.
func runCalibrate(o opts) {
	logText, err := readLog(o)
	if err != nil {
		die("%v", err)
	}
	measured, err := fit.ParseServerLog(logText)
	if err != nil {
		die("%v", err)
	}
	m, err := loadModel(o.src)
	if err != nil {
		die("%v", err)
	}

	// Raw prediction (ComputeScale = 1) so the ratio reflects heuristic → real.
	o.cfg.ComputeScale = 1.0
	pred, err := fit.Predict(m, fit.Hardware{FreeVRAM: 1 << 62, FreeRAM: 1 << 62}, o.cfg)
	if err != nil {
		die("%v", err)
	}

	realWeights := measured.GPUModelBytes // VRAM-resident weights (see verify)
	if realWeights == 0 {
		realWeights = measured.TotalModel()
	}
	sample := fit.Sample{
		Model:       o.src,
		Ctx:         o.cfg.Ctx,
		PredCompute: pred.ComputeBytes,
		RealCompute: measured.TotalCompute(),
		PredKV:      pred.KVBytes,
		RealKV:      measured.TotalKV(),
		PredWeights: pred.WeightsBytes,
		RealWeights: realWeights,
	}
	p, err := fit.LoadProfile()
	if err != nil {
		die("loading profile: %v", err)
	}
	p.AddSample(sample)

	// If the launch came from a running container, calibrate the per-GPU base
	// (CUDA context) overhead from the process's total VRAM minus its buffers.
	if o.container != "" {
		if pid := infra.ContainerPID(o.container); pid > 0 {
			if procVRAM := infra.GPUProcessMem(pid); procVRAM > 0 {
				p.BaseOverhead = fit.CalibratedBasePerGPU(procVRAM, measured)
			}
		}
	}

	// Ratchet the load-time margin from per-device residuals: any device whose
	// real buffers exceeded the per-device prediction is an observed gap the
	// margin must cover on this host (worst-case-wins, never below the default).
	observeMargin(p, m, measured, o.cfg)

	if err := p.Save(); err != nil {
		die("saving profile: %v", err)
	}

	fmt.Println("calibration sample recorded:")
	cmp := func(label string, pred, real int64) {
		ratio := 0.0
		if pred > 0 {
			ratio = float64(real) / float64(pred)
		}
		fmt.Printf("  %-14s predicted %10s   real %10s   (%.2f×)\n",
			label, fit.HumanBytes(pred), fit.HumanBytes(real), ratio)
	}
	cmp("weights", sample.PredWeights, sample.RealWeights)
	cmp("kv cache", sample.PredKV, sample.RealKV)
	cmp("compute", sample.PredCompute, sample.RealCompute)
	path, _ := fit.ProfilePath()
	fmt.Printf("\nnew compute scale: %.2f×  (applied to future predictions)\n", p.ComputeScale())
	if p.BaseOverhead > 0 {
		fmt.Printf("base overhead:     %s / GPU (calibrated)\n", fit.HumanBytes(p.BaseOverhead))
	}
	if p.LoadMargin > 0 {
		fmt.Printf("load margin:       %s (worst observed under-prediction)\n", fit.HumanBytes(p.LoadMargin))
	}
	fmt.Printf("saved: %s\n", path)
}

// observeMargin feeds per-device under-predictions into the profile's load
// margin. Needs the per-device split prediction, which needs each device's free
// VRAM from the log (params_fit lines); skipped silently when absent.
func observeMargin(p *fit.Profile, m *fit.Model, measured *fit.Measured, cfg fit.Config) {
	var free []int64
	for _, d := range measured.Devices {
		if d.FreeBytes <= 0 {
			return
		}
		free = append(free, d.FreeBytes)
	}
	if len(free) < 2 {
		return
	}
	dr, err := fit.PredictDevices(m, fit.Hardware{GPUsFree: free, FreeRAM: 1 << 62}, cfg)
	if err != nil {
		return
	}
	for i, d := range dr.Devices {
		if i < len(measured.Devices) {
			p.ObserveLoadMargin(measured.Devices[i].Total() - d.Used)
		}
	}
}

func readLog(o opts) (string, error) {
	switch {
	case o.container != "":
		out, err := runCombined("docker", "logs", o.container)
		if err != nil {
			// out holds docker's stderr (no such container / socket permission
			// denied) — dropping it left users with a bare "exit status 1".
			return "", fmt.Errorf("docker logs %s: %v: %s", o.container, err,
				strings.TrimSpace(firstLine(out)))
		}
		return out, nil
	case o.logFile == "-":
		b, err := io.ReadAll(os.Stdin)
		return string(b), err
	case o.logFile != "":
		b, err := os.ReadFile(o.logFile)
		return string(b), err
	}
	return "", fmt.Errorf("calibrate needs --log FILE, --log -, or --container NAME")
}

// loadModel parses a GGUF header from an "owner/repo[:quant]" ref, an HTTP(S)
// URL, or a local path, via the shared hub loader.
func loadModel(src string) (*fit.Model, error) {
	m, _, quant, err := hub.Load(src)
	if err != nil {
		return nil, err
	}
	if ref, ok := fit.ParseRef(src); ok && ref.Quant == "" && quant != "" {
		fmt.Fprintf(os.Stderr, "no quant given; using %s (specify :QUANT to override)\n", quant)
	}
	return m, nil
}

func report(m *fit.Model, hw fit.Hardware, c fit.Config, r *fit.Result) {
	fmt.Printf("model:   %s  (%d layers, %d/%d heads, vocab %d)\n",
		m.Arch, m.NLayers, m.NKVHeads, m.NHeads, m.VocabSize)
	fmt.Printf("config:  ctx %d, ubatch %d, kv %s\n", c.Ctx, c.UBatch, c.KVType)
	gpuNote := ""
	if hw.NumGPUs > 1 {
		gpuNote = fmt.Sprintf(" across %d GPUs", hw.NumGPUs)
	}
	fmt.Printf("host:    %s free VRAM%s, %s free RAM\n",
		fit.HumanBytes(hw.FreeVRAM), gpuNote, fit.HumanBytes(hw.FreeRAM))
	if hw.FreeVRAM == 0 {
		fmt.Printf("         %s\n", "no GPU detected — predicting CPU/RAM only (override with --vram-mb)")
	}
	fmt.Println()

	row := func(label string, b int64) { fmt.Printf("  %-18s %12s\n", label, fit.HumanBytes(b)) }
	fmt.Println("memory breakdown:")
	row("weights", r.WeightsBytes)
	row("kv cache", r.KVBytes)
	row("compute buffer", r.ComputeBytes)
	row("base overhead", r.BaseBytes)
	fmt.Println()
	row("→ VRAM used", r.VRAMUsed)
	row("→ RAM used", r.RAMUsed)
	fmt.Printf("\noffload: %d / %d layers on GPU  (suggested -ngl %d)\n",
		r.LayersOnGPU, r.NLayers, r.RecommendedNGL)
	fmt.Printf("verdict: %s\n", colorMode(r.Mode))
	if r.Mode == fit.ModeOOM || r.Tight {
		note := "warning: tight fit — may crash at load (llama.cpp needs extra VRAM while loading)"
		if r.Mode == fit.ModeOOM {
			note = "warning: predicted to crash at load"
		}
		if safe := fit.SafeMaxNGL(m, hw, c); safe > 0 {
			note += fmt.Sprintf("; largest safe explicit -ngl: %d", safe)
		}
		fmt.Println(note)
	}
	if c.NCPUMoE > 0 {
		fmt.Printf("note: --n-cpu-moe %d — 'weights' is the full model; offloaded experts ride RAM, so VRAM excludes them\n", c.NCPUMoE)
	}
	if r.MLA {
		fmt.Println("note: MLA model (DeepSeek-V2/V3) — KV cache is compressed; this estimate is approximate, verify against a real launch")
	}
	if r.Recurrent {
		fmt.Println("note: hybrid/SSM model (recurrent-state layers) — layer split and state size are approximate, verify against a real launch")
	}
	if r.Paged {
		fmt.Println("note: mmap'd weights exceed free RAM — it will run, paging from disk (slow); --no-mmap/--mlock would make this an OOM")
	}
}

func colorMode(mode fit.Mode) string {
	if os.Getenv("NO_COLOR") != "" {
		return string(mode)
	}
	var code string
	switch mode {
	case fit.ModeGPU:
		code = "32" // green
	case fit.ModeHybrid:
		code = "36" // cyan
	case fit.ModeCPU:
		code = "33" // yellow
	case fit.ModeOOM:
		code = "31" // red
	}
	return fmt.Sprintf("\033[1;%sm%s\033[0m", code, mode)
}

// runCombined captures both stdout and stderr (llama-server logs to stderr).
// firstLine is the useful part of a failed command's output.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func runCombined(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).CombinedOutput()
	return string(out), err
}

func atoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		die("expected a number, got %q", s)
	}
	return n
}

func die(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "llamadeck: "+format+"\n", a...)
	os.Exit(1)
}
