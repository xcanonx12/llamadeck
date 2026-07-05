package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
	"llamadeck/hub"
	"llamadeck/infra"
	"llamadeck/tui"
)

const (
	fitMinCtx    = 512
	fitMaxCtx    = 1 << 20
	fitMinUBatch = 64
	fitMaxUBatch = 4096
)

type fitMode int

const (
	fitMain      fitMode = iota // graph + inline settings + LAUNCH (default)
	fitQuantLoad                // scanning quants
	fitQuant                    // quant picker list
	fitConfirm                  // launch confirmation prompt
)

// quantReportMsg carries the async per-quant fit scan back to the Fit tab.
type quantReportMsg struct {
	repo string
	rows []hub.QuantRow
	pick int
	err  error
}

// fitTab is the cockpit: the live VRAM/RAM graph for the selected model, an
// inline launch-settings panel (ctx/ubatch/cache-type feed the graph live), an
// in-place quant picker, and a confirmed LAUNCH — all on one screen.
type fitTab struct {
	sh       *shared
	mode     fitMode
	settings *fitSettings
	suggNGL  int // latest predicted recommended ngl (for "auto")
	rows     []hub.QuantRow
	qsel     int
	status   string
}

func newFitTab(sh *shared) tab {
	return &fitTab{sh: sh, settings: newFitSettings(sh, 0)}
}

func (t *fitTab) Title() string { return "Fit" }

// Focused steals all keys (incl. tab/digits/q) only while an overlay is open;
// on the main screen the global nav keys stay live and ←→/↑↓ edit settings.
func (t *fitTab) Focused() bool { return t.mode != fitMain || t.settings.editing }
func (t *fitTab) Init() tea.Cmd { return nil }

func (t *fitTab) Update(msg tea.Msg) (tab, tea.Cmd) {
	switch msg := msg.(type) {
	case launchedMsg:
		if msg.err != nil {
			t.status = stErr.Render("launch failed: " + msg.err.Error())
		} else {
			t.status = stOK.Render(fmt.Sprintf("launched %s on :%d — Monitor tab (3), then p to plug it into your coding agent", msg.name, msg.port))
		}
		return t, nil
	case quantReportMsg:
		if t.mode != fitQuantLoad {
			return t, nil // user navigated away while scanning
		}
		if msg.err != nil {
			t.mode, t.status = fitMain, stErr.Render("quant scan: "+msg.err.Error())
			return t, nil
		}
		t.rows, t.qsel, t.mode = msg.rows, max(0, msg.pick), fitQuant
		return t, nil
	case tea.KeyMsg:
		return t.handleKey(msg)
	}
	return t, nil
}

func (t *fitTab) handleKey(key tea.KeyMsg) (tab, tea.Cmd) {
	if t.sh.selected == nil {
		return t, nil
	}
	switch t.mode {
	case fitQuant:
		return t.handleQuantKey(key)
	case fitConfirm:
		return t.handleConfirmKey(key)
	case fitQuantLoad:
		if key.String() == "esc" {
			t.mode = fitMain
		}
		return t, nil
	}
	// fitMain
	if t.settings.editing {
		t.settings.editKey(key.String())
		return t, nil
	}
	switch key.String() {
	case "up", "k":
		t.settings.up()
	case "down", "j":
		t.settings.down()
	case "left", "right", " ":
		t.settings.adjust(key.String())
	case "e":
		t.settings.startEdit()
	case "m":
		if t.sh.hw.NumGPUs > 0 { // no-op on a GPU-less host — CPU mode is forced on
			t.sh.cpuMode = !t.sh.cpuMode
			if t.sh.cpuMode {
				t.sh.cfg.KVType = "f16" // quantized KV needs flash-attn we don't use on CPU
			}
		}
	case "c":
		return t.startQuantScan()
	case "enter":
		if !t.sh.imageOK {
			t.status = stWarn.Render("build the server image first (Config tab)")
			return t, nil
		}
		if t.settings.port == 0 {
			t.settings.port = infra.FreePort(8080)
		}
		t.refreshHW() // decide on launch-moment free VRAM, not a stale tick
		t.mode = fitConfirm
	}
	return t, nil
}

// refreshHW re-probes the GPUs so the confirm verdict reflects free VRAM at
// the moment of the launch decision (same fields the Monitor tick maintains).
func (t *fitTab) refreshHW() {
	gpus := infra.GPUs()
	if len(gpus) == 0 {
		return
	}
	free := make([]int64, len(gpus))
	var total int64
	for i, g := range gpus {
		free[i] = g.MemFree
		total += g.MemFree
	}
	t.sh.gpuFree = free
	t.sh.hw.FreeVRAM = total
	t.sh.hw.NumGPUs = len(gpus)
	t.sh.hw.FreeRAM = infra.FreeRAM()
}

func (t *fitTab) handleConfirmKey(key tea.KeyMsg) (tab, tea.Cmd) {
	switch key.String() {
	case "y", "enter":
		hf := t.sh.selected.src
		if t.sh.selected.quant != "" {
			hf += ":" + t.sh.selected.quant
		}
		spec := t.settings.toSpec(hf, cacheDir())
		t.mode = fitMain
		t.status = fmt.Sprintf("launching %s on :%d (ngl %d)…", hf, spec.Port, spec.NGL)
		return t, func() tea.Msg {
			name, err := infra.Launch(spec)
			return launchedMsg{name: name, port: spec.Port, err: err}
		}
	case "n", "esc":
		t.mode = fitMain
	}
	return t, nil
}

func (t *fitTab) startQuantScan() (tab, tea.Cmd) {
	ref, ok := fit.ParseRef(t.sh.selected.src)
	if !ok {
		t.status = stWarn.Render("quant picker needs a Hugging Face repo")
		return t, nil
	}
	t.mode, t.status = fitQuantLoad, ""
	repo, hw, cfg := ref.Repo, t.sh.hw, t.sh.cfg
	return t, func() tea.Msg {
		rows, pick, err := hub.QuantReport(repo, hw, cfg)
		return quantReportMsg{repo: repo, rows: rows, pick: pick, err: err}
	}
}

func (t *fitTab) handleQuantKey(key tea.KeyMsg) (tab, tea.Cmd) {
	switch key.String() {
	case "esc", "c":
		t.mode = fitMain
	case "up", "k":
		if t.qsel > 0 {
			t.qsel--
		}
	case "down", "j":
		if t.qsel < len(t.rows)-1 {
			t.qsel++
		}
	case "enter":
		if t.qsel < len(t.rows) {
			r := t.rows[t.qsel]
			t.sh.selected.quant = r.Quant
			if mm := t.sh.selected.model; mm != nil {
				// Weights track the chosen quant. Rescale the exact tensor table to
				// the new file size so the graph actually updates (it otherwise sizes
				// from Tensors and ignores FileBytes).
				if mm.Tensors != nil {
					mm.Tensors = mm.Tensors.ScaledTo(r.Size)
				}
				mm.FileBytes = r.Size
			}
		}
		t.mode = fitMain
	}
	return t, nil
}

func (t *fitTab) View(width, height int) string {
	if t.sh.selected == nil {
		return stMuted.Render("\n  No model selected.\n  Go to the ") +
			stKey.Render("Models") + stMuted.Render(" tab, pick one, press ") +
			stKey.Render("enter") + stMuted.Render(" twice to send it here.")
	}

	switch t.mode {
	case fitQuantLoad:
		return stMuted.Render("\n  scanning quants for " + t.sh.selected.src + " …")
	case fitQuant:
		return t.quantView(width)
	}

	// Live graph (reflects ctx / ubatch / KV-cache-type / GPU target from the
	// settings panel — pinning a single GPU predicts against that device alone).
	s := t.sh.selected
	hw := t.settings.hw()
	// cfg is a value copy so the -ngl / MoE knobs feed the graph live without
	// mutating the shared config other tabs read. MoE only applies to MoE models.
	cfg := t.sh.cfg
	if t.settings.moeSupported() {
		cfg.NCPUMoE = t.settings.moe
	}
	if !t.settings.nglAuto {
		cfg.NGL = t.settings.ngl
	}
	st := tui.State{Source: s.src, Quant: s.quant, Model: s.model, HW: hw, Cfg: cfg,
		Width: min(width, 80), HideHint: true}
	st.Recompute()
	graph := tui.RenderView(st)

	// Keep the suggested ngl current for "auto" and for launch.
	if r, err := fit.Predict(s.model, hw, cfg); err == nil {
		t.suggNGL = r.RecommendedNGL
	}

	if t.mode == fitConfirm {
		return graph + "\n\n" + t.confirmPanel()
	}

	// Fixed chrome: graph on top, LAUNCH (+status) pinned at the bottom. The
	// settings list gets whatever height is left and scrolls within it, so the
	// app header never gets pushed off-screen (crucial: the graph stays put). On a
	// terminal too short for both, the graph + LAUNCH win and the list drops out.
	sep := stDim.Render(strings.Repeat("─", min(width, 80)))
	header := stBold.Render("Launch settings") + stMuted.Render("   ↑↓ field · ←→ adjust · e edit · c quant · m cpu")
	launch := t.launchButton()
	statusH := 0
	if t.status != "" {
		statusH = 1
	}
	// Safe max -ngl for the settings row warning — only worth computing when
	// the user has taken manual control of -ngl.
	safeMax := 0
	if !t.settings.nglAuto {
		safeMax = fit.SafeMaxNGL(s.model, hw, cfg)
	}
	out := graph
	// listRows: rows left after graph, sep, header, launch, status.
	listRows := height - lipgloss.Height(graph) - 3 - statusH
	if listRows >= 1 {
		out += "\n" + sep + "\n" + header + "\n" + t.settings.render(t.suggNGL, safeMax, listRows)
	}
	out += "\n" + launch
	if t.status != "" {
		out += "\n" + t.status
	}
	return out
}

func (t *fitTab) launchButton() string {
	if !t.sh.imageOK {
		return stWarn.Render("  ⚠ build the server image (Config tab) before launching")
	}
	btn := lipgloss.NewStyle().Background(cGreen).Foreground(cBg).Bold(true).Padding(0, 2).Render("▶ LAUNCH")
	return btn + stMuted.Render("  press ") + stKey.Render("enter") + stMuted.Render(" to start this model")
}

func (t *fitTab) confirmPanel() string {
	s := t.sh.selected
	hf := s.src
	if s.quant != "" {
		hf += ":" + s.quant
	}
	ngl := fmt.Sprint(t.settings.effNGL())
	if t.settings.nglAuto {
		ngl = fmt.Sprintf("auto (~%d)", t.suggNGL) // launches -ngl 999 → llama auto-fits
	}
	if t.sh.cpuMode {
		ngl = "n/a (CPU)"
	}
	mode := ""
	cfg := t.sh.cfg
	if t.settings.moeSupported() {
		cfg.NCPUMoE = t.settings.moe
	}
	if !t.settings.nglAuto {
		cfg.NGL = t.settings.ngl // same knobs the graph predicts with
	}
	// Crash gate: judge the launch on fresh hardware (refreshHW ran on enter).
	// Per-device when available — that's where single-GPU overloads hide.
	hw := t.settings.hw()
	warn := ""
	if len(hw.GPUsFree) > 1 {
		if dr, err := fit.PredictDevices(s.model, hw, cfg); err == nil {
			mode = "  predicted " + string(dr.Mode)
			switch {
			case !dr.Fits:
				warn = fmt.Sprintf("predicted to CRASH at load: GPU%d needs %s, has %s",
					dr.Bottleneck, human(dr.Devices[dr.Bottleneck].Used), human(dr.Devices[dr.Bottleneck].Free))
			case dr.Tight:
				warn = "tight fit — may crash at load (llama.cpp needs extra VRAM while loading)"
			}
		}
	} else if r, err := fit.Predict(s.model, hw, cfg); err == nil {
		mode = "  predicted " + string(r.Mode)
		switch {
		case t.sh.cpuMode:
			if r.Mode == fit.ModeOOM {
				warn = fmt.Sprintf("predicted to CRASH: needs %s RAM, has %s",
					human(r.RAMUsed), human(hw.FreeRAM))
			}
		case r.VRAMUsed > hw.FreeVRAM || r.Mode == fit.ModeOOM:
			warn = fmt.Sprintf("predicted to CRASH at load: needs %s VRAM, has %s",
				human(r.VRAMUsed), human(hw.FreeVRAM))
		case r.Tight:
			warn = "tight fit — may crash at load (llama.cpp needs extra VRAM while loading)"
		}
	}
	launchKey := stKey.Render("  y") + stText.Render(" launch")
	warnLine := ""
	if warn != "" {
		if !t.sh.cpuMode {
			if safe := fit.SafeMaxNGL(s.model, hw, cfg); safe > 0 {
				warn += fmt.Sprintf(" — safe -ngl ≤ %d", safe)
			}
		}
		warnLine = "\n" + stErr.Render("  ⚠ "+warn) + "\n"
		launchKey = stKey.Render("  y") + stErr.Render(" launch anyway")
	}
	box := stBold.Render("Launch this model?") + "\n\n" +
		stText.Render("  "+hf) + "\n" +
		stMuted.Render(fmt.Sprintf("  port %d · ctx %d · ngl %s · KV %s%s",
			t.settings.port, t.sh.cfg.Ctx, ngl, t.sh.cfg.KVType, mode)) + "\n" +
		warnLine + "\n" +
		launchKey + stMuted.Render("   ·   ") +
		stKey.Render("n") + stMuted.Render(" cancel")
	return box
}

func (t *fitTab) quantView(width int) string {
	var b strings.Builder
	b.WriteString(stBold.Render("Quants for "+t.sh.selected.src) + "\n")
	b.WriteString(stMuted.Render(fmt.Sprintf("  %s %s %s %s",
		pad("QUANT", 12), pad("SIZE", 10), pad("MAX GPU CTX", 12), "AT CTX "+fmt.Sprint(t.sh.cfg.Ctx))) + "\n")
	for i, r := range t.rows {
		maxctx := "—"
		if r.MaxGPU > 0 {
			maxctx = fmt.Sprint(r.MaxGPU)
		}
		line := fmt.Sprintf("%s %s %s %s",
			pad(r.Quant, 12), pad(human(r.Size), 10), pad(maxctx, 12), verdictBadge(r.Mode))
		if i == t.qsel {
			b.WriteString(stSelected.Render("› "+line) + "\n")
		} else {
			b.WriteString("  " + line + "\n")
		}
	}
	b.WriteString("\n" + stMuted.Render("↑↓ select · ") + stKey.Render("enter") +
		stMuted.Render(" use this quant · ") + stKey.Render("esc") + stMuted.Render(" cancel"))
	return b.String()
}
