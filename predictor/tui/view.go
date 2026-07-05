// Package tui renders the fit prediction as a striking terminal view. RenderView
// is a pure State→string function (unit-testable, printable without a TTY); app.go
// wraps it in a Bubble Tea loop for live context/batch exploration.
package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
)

// State is everything the view needs. Cfg.Ctx / Cfg.UBatch are the live knobs.
type State struct {
	Source string
	Quant  string
	Model  *fit.Model
	HW     fit.Hardware
	Cfg    fit.Config
	Result *fit.Result
	Width  int
	// HideHint drops the "←/→ context" key-hint line — the app embeds this view
	// under its own settings panel where those keys mean something else.
	HideHint bool
}

// Recompute re-runs the prediction after a knob changes.
func (s *State) Recompute() {
	r, err := fit.Predict(s.Model, s.HW, s.Cfg)
	if err == nil {
		s.Result = r
	}
}

// --- palette --------------------------------------------------------------

var (
	cBase    = lipgloss.Color("#6C7086") // gray   — base/CUDA overhead
	cWeights = lipgloss.Color("#A6E3A1") // green  — weights
	cKV      = lipgloss.Color("#89DCEB") // cyan   — kv cache
	cCompute = lipgloss.Color("#CBA6F7") // mauve  — compute buffer
	cSpillW  = lipgloss.Color("#F9E2AF") // yellow — weights spilled to RAM
	cSpillK  = lipgloss.Color("#FAB387") // orange — kv spilled to RAM
	cDim     = lipgloss.Color("#45475A") // empty track
	cText    = lipgloss.Color("#CDD6F4")
	cMuted   = lipgloss.Color("#9399B2")

	dimStyle   = lipgloss.NewStyle().Foreground(cDim)
	mutedStyle = lipgloss.NewStyle().Foreground(cMuted)
	textStyle  = lipgloss.NewStyle().Foreground(cText)
	boldStyle  = lipgloss.NewStyle().Foreground(cText).Bold(true)
	keyStyle   = lipgloss.NewStyle().Foreground(cKV).Bold(true)

	frame = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#585B70")).
		Padding(0, 2)
)

func verdictStyle(m fit.Mode) lipgloss.Style {
	bg := lipgloss.Color("#A6E3A1") // GPU green
	switch m {
	case fit.ModeHybrid:
		bg = lipgloss.Color("#89DCEB")
	case fit.ModeCPU:
		bg = lipgloss.Color("#F9E2AF")
	case fit.ModeOOM:
		bg = lipgloss.Color("#F38BA8")
	}
	return lipgloss.NewStyle().Background(bg).Foreground(lipgloss.Color("#11111B")).
		Bold(true).Padding(0, 2)
}

// --- bars -----------------------------------------------------------------

type barSeg struct {
	bytes int64
	color lipgloss.Color
}

// renderBar draws a proportional, color-segmented bar of fixed cell width.
func renderBar(width int, total int64, segs []barSeg) string {
	if total <= 0 {
		total = 1
	}
	var b strings.Builder
	cells := 0
	for _, s := range segs {
		if s.bytes <= 0 {
			continue
		}
		n := int(float64(s.bytes)/float64(total)*float64(width) + 0.5)
		if n < 1 {
			n = 1 // keep a nonzero allocation visible
		}
		if cells+n > width {
			n = width - cells
		}
		if n <= 0 {
			break
		}
		b.WriteString(lipgloss.NewStyle().Foreground(s.color).Render(strings.Repeat("█", n)))
		cells += n
	}
	if cells < width {
		b.WriteString(dimStyle.Render(strings.Repeat("░", width-cells)))
	}
	return b.String()
}

// --- view -----------------------------------------------------------------

// RenderView produces the full styled view for the current state.
func RenderView(s State) string {
	if s.HW.FreeVRAM <= 0 {
		return renderCPU(s)
	}
	r := s.Result
	w := s.Width
	if w < 48 {
		w = 64
	}
	barW := w - 20
	if barW < 16 {
		barW = 16
	}

	title := boldStyle.Render("llamadeck") + mutedStyle.Render("  ·  "+s.Source)
	if s.Quant != "" {
		title += keyStyle.Render("  " + s.Quant)
	}

	sub := mutedStyle.Render(fmt.Sprintf("%s · %d layers · %d/%d kv/attn heads · vocab %d",
		s.Model.Arch, s.Model.NLayers, s.Model.NKVHeads, s.Model.NHeads, s.Model.VocabSize))
	if r.MLA {
		sub += lipgloss.NewStyle().Foreground(lipgloss.Color("#FAB387")).Render("  · MLA (KV approx)")
	}
	if r.Recurrent {
		sub += lipgloss.NewStyle().Foreground(lipgloss.Color("#FAB387")).Render("  · hybrid SSM (approx)")
	}

	cfg := fmt.Sprintf("%s %s   %s %d   %s %s",
		mutedStyle.Render("ctx"), keyStyle.Render(fmt.Sprint(s.Cfg.Ctx)),
		mutedStyle.Render("ubatch"), s.Cfg.UBatch,
		mutedStyle.Render("kv"), s.Cfg.KVType)

	// VRAM composition — exact side-of-the-fence totals from the engine (a
	// layers×per-layer product lies for hybrids' non-uniform layers).
	vramSegs := []barSeg{}
	if r.LayersOnGPU > 0 {
		vramSegs = []barSeg{
			{r.BaseBytes, cBase},
			{r.GPUWeightBytes, cWeights},
			{r.GPUKVBytes, cKV},
			{r.ComputeBytes, cCompute},
		}
	}
	vramBar := renderBar(barW, s.HW.FreeVRAM, vramSegs)

	// RAM composition (spill).
	ramSegs := []barSeg{
		{r.CPUWeightBytes, cSpillW},
		{r.CPUKVBytes, cSpillK},
	}
	if r.LayersOnGPU == 0 {
		ramSegs = append(ramSegs, barSeg{r.ComputeBytes, cCompute})
	}
	ramBar := renderBar(barW, s.HW.FreeRAM, ramSegs)

	gauge := fmt.Sprintf("%s %s %s",
		mutedStyle.Render("GPU layers"),
		renderBar(barW, int64(r.NLayers), []barSeg{{int64(r.LayersOnGPU), cWeights}}),
		boldStyle.Render(fmt.Sprintf("%d/%d", r.LayersOnGPU, r.NLayers)))

	vramLine := fmt.Sprintf("%s %s %s", lab("VRAM"), vramBar,
		mutedStyle.Render(fmt.Sprintf("%s / %s", fit.HumanBytes(r.VRAMUsed), fit.HumanBytes(s.HW.FreeVRAM))))

	// Per-device split: only when targeting all GPUs with known per-device free
	// VRAM (>1 device) AND something is actually offloaded (else the pooled VRAM
	// bar is clearer). Single-GPU and pinned paths leave vramLine/verdict as-is.
	var deviceBlock, deviceVerdict string
	if len(s.HW.GPUsFree) > 1 && r.LayersOnGPU > 0 {
		if dr, err := fit.PredictDevices(s.Model, s.HW, s.Cfg); err == nil {
			deviceBlock = renderDevices(dr, barW)
			safe := 0
			if dr.Mode == fit.ModeOOM || dr.Tight {
				safe = fit.SafeMaxNGL(s.Model, s.HW, s.Cfg)
			}
			deviceVerdict = deviceVerdictLine(dr, s.Cfg.NGL > 0, safe)
		}
	}
	ramOver := ""
	if r.Paged {
		// mmap'd weights beyond free RAM: runs, but pages from disk.
		ramOver = tightStyle.Render(fmt.Sprintf("  ⚠ +%s pages from disk (slow) — mmap",
			fit.HumanBytes(r.RAMUsed-s.HW.FreeRAM)))
	} else if r.RAMUsed > s.HW.FreeRAM {
		ramOver = lipgloss.NewStyle().Foreground(lipgloss.Color("#F38BA8")).Bold(true).
			Render(fmt.Sprintf("  ⚠ +%s", fit.HumanBytes(r.RAMUsed-s.HW.FreeRAM)))
	}
	ramLine := fmt.Sprintf("%s %s %s%s", lab("RAM "), ramBar,
		mutedStyle.Render(fmt.Sprintf("%s / %s", fit.HumanBytes(r.RAMUsed), fit.HumanBytes(s.HW.FreeRAM))), ramOver)

	legend := strings.Join([]string{
		swatch(cWeights, "weights"), swatch(cKV, "kv"),
		swatch(cCompute, "compute"), swatch(cBase, "overhead"),
		swatch(cSpillW, "spill"),
	}, mutedStyle.Render("  "))

	verdict := verdictStyle(r.Mode).Render(string(r.Mode)) +
		textStyle.Render(fmt.Sprintf("   suggested %s", boldStyle.Render(fmt.Sprintf("-ngl %d", r.RecommendedNGL))))
	if r.Tight && r.Mode != fit.ModeOOM {
		why := "llama.cpp needs extra VRAM at load"
		if s.Cfg.NGL > 0 {
			why = safeNGLHint(fit.SafeMaxNGL(s.Model, s.HW, s.Cfg))
		}
		verdict += "  " + tightStyle.Render("⚠ TIGHT — may crash at load; "+why)
	}
	if deviceVerdict != "" {
		verdict = deviceVerdict
	}

	// On the multi-GPU split, the per-device bars replace the single pooled VRAM bar.
	vramRows := vramLine
	if deviceBlock != "" {
		vramRows = deviceBlock
	}

	rows := []string{
		title,
		sub,
		"",
		cfg,
		"",
		vramRows,
		ramLine,
		gauge,
		"",
		legend,
		"",
		verdict,
	}
	if !s.HideHint {
		rows = append(rows, "", mutedStyle.Render("←/→ context   ↑/↓ ubatch   q quit"))
	}
	body := strings.Join(rows, "\n")

	return frame.Render(body) + "\n"
}

// renderCPU is the RAM-only frame for a machine with no usable GPU: weights, KV,
// and compute all live in system RAM, so there is no VRAM bar, no -ngl gauge, and
// no per-device split. Verdict collapses to RUN / PAGED / OOM.
func renderCPU(s State) string {
	r := s.Result
	w := s.Width
	if w < 48 {
		w = 64
	}
	barW := w - 20
	if barW < 16 {
		barW = 16
	}

	title := boldStyle.Render("llamadeck") + mutedStyle.Render("  ·  "+s.Source)
	if s.Quant != "" {
		title += keyStyle.Render("  " + s.Quant)
	}
	sub := mutedStyle.Render(fmt.Sprintf("%s · %d layers · vocab %d",
		s.Model.Arch, s.Model.NLayers, s.Model.VocabSize)) +
		tightStyle.Render("  · CPU mode — no GPU offload; sizing system RAM only")

	cfg := fmt.Sprintf("%s %s   %s %d   %s %s",
		mutedStyle.Render("ctx"), keyStyle.Render(fmt.Sprint(s.Cfg.Ctx)),
		mutedStyle.Render("ubatch"), s.Cfg.UBatch,
		mutedStyle.Render("kv"), s.Cfg.KVType)

	ramSegs := []barSeg{
		{r.CPUWeightBytes, cWeights},
		{r.CPUKVBytes, cKV},
		{r.ComputeBytes, cCompute},
	}
	ramOver := ""
	if r.Paged {
		ramOver = tightStyle.Render(fmt.Sprintf("  ⚠ +%s pages from disk (slow) — mmap",
			fit.HumanBytes(r.RAMUsed-s.HW.FreeRAM)))
	} else if r.RAMUsed > s.HW.FreeRAM {
		ramOver = alertStyle.Render(fmt.Sprintf("  ⚠ +%s", fit.HumanBytes(r.RAMUsed-s.HW.FreeRAM)))
	}
	ramLine := fmt.Sprintf("%s %s %s%s", lab("RAM "), renderBar(barW, s.HW.FreeRAM, ramSegs),
		mutedStyle.Render(fmt.Sprintf("%s / %s", fit.HumanBytes(r.RAMUsed), fit.HumanBytes(s.HW.FreeRAM))), ramOver)

	legend := strings.Join([]string{
		swatch(cWeights, "weights"), swatch(cKV, "kv"), swatch(cCompute, "compute"),
	}, mutedStyle.Render("  "))

	// RUN / PAGED / OOM — reuse verdictStyle for the palette.
	var verdict string
	switch {
	case r.Mode == fit.ModeOOM:
		verdict = verdictStyle(fit.ModeOOM).Render("OOM")
	case r.Paged:
		verdict = verdictStyle(fit.ModeCPU).Render("PAGED (slow)")
	default:
		verdict = verdictStyle(fit.ModeGPU).Render("RUN")
	}
	verdict += textStyle.Render("   runs entirely on CPU")

	rows := []string{title, sub, "", cfg, "", ramLine, "", legend, "", verdict}
	if !s.HideHint {
		rows = append(rows, "", mutedStyle.Render("←/→ context   ↑/↓ ubatch   q quit"))
	}
	return frame.Render(strings.Join(rows, "\n")) + "\n"
}

var alertStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#F38BA8")).Bold(true)

// renderDevices draws one used/free bar per GPU for the `--gpus all` split,
// flagging the device(s) that overflow.
func renderDevices(dr *fit.DeviceResult, barW int) string {
	lines := make([]string, 0, len(dr.Devices))
	for _, d := range dr.Devices {
		// Same colored buckets as the pooled VRAM bar; an OOM device goes red.
		segs := []barSeg{
			{d.BaseBytes, cBase},
			{d.WeightBytes, cWeights},
			{d.KVBytes, cKV},
			{d.ComputeBytes, cCompute},
		}
		if d.OOM {
			segs = []barSeg{{d.Used, lipgloss.Color("#F38BA8")}}
		}
		bar := renderBar(barW, d.Free, segs)
		tag := ""
		if d.Main {
			tag += mutedStyle.Render(" ·main")
		}
		if d.OOM {
			tag += alertStyle.Render(fmt.Sprintf("  ⚠ OOM +%s", fit.HumanBytes(d.OverBy)))
		} else if d.Tight {
			tag += tightStyle.Render("  ⚠ tight")
		}
		lines = append(lines, fmt.Sprintf("%s %s %s%s",
			lab(fmt.Sprintf("GPU%d", d.Index)), bar,
			mutedStyle.Render(fmt.Sprintf("%s / %s  %dL", fit.HumanBytes(d.Used), fit.HumanBytes(d.Free), d.Layers)),
			tag))
	}
	return strings.Join(lines, "\n")
}

var tightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#F9E2AF")).Bold(true)

// safeNGLHint names the safe fallback for a crashing/tight -ngl choice.
func safeNGLHint(safe int) string {
	if safe > 0 {
		return fmt.Sprintf("set -ngl ≤ %d or auto", safe)
	}
	return "no explicit -ngl is safe — lower ctx/ubatch or pick a smaller quant"
}

// deviceVerdictLine is the bottom verdict for the multi-GPU split: a fit badge,
// a TIGHT warning (fits on paper, inside the load-time margin — may crash), or
// the bottleneck device + remedy when a device OOMs. An explicit -ngl means
// llama.cpp splits proportionally (no auto-fit rebalancing), so the first
// remedy there is the computed safe -ngl / going back to auto.
func deviceVerdictLine(dr *fit.DeviceResult, explicitNGL bool, safe int) string {
	switch {
	case dr.Mode == fit.ModeOOM || !dr.Fits:
		remedy := fmt.Sprintf("GPU%d over by %s — pin a single GPU, lower ctx, or use q8_0 KV",
			dr.Bottleneck, fit.HumanBytes(dr.OverBy))
		if explicitNGL {
			remedy = fmt.Sprintf("GPU%d over by %s — %s, or lower ctx",
				dr.Bottleneck, fit.HumanBytes(dr.OverBy), safeNGLHint(safe))
		}
		return verdictStyle(fit.ModeOOM).Render("WON'T FIT (--gpus all)") + "  " + alertStyle.Render(remedy)
	case dr.Tight:
		why := "llama.cpp needs extra VRAM at load"
		if explicitNGL {
			why = safeNGLHint(safe)
		}
		return verdictStyle(dr.Mode).Render(string(dr.Mode)) + "  " +
			tightStyle.Render("⚠ TIGHT — may crash at load; "+why)
	}
	return verdictStyle(dr.Mode).Render(string(dr.Mode)) +
		textStyle.Render(fmt.Sprintf("   %d GPUs · per-device split", len(dr.Devices)))
}

func lab(s string) string { return mutedStyle.Render(s) }

func swatch(c lipgloss.Color, label string) string {
	return lipgloss.NewStyle().Foreground(c).Render("█") + mutedStyle.Render(" "+label)
}
