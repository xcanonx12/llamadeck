package app

import (
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
	"llamadeck/hub"
	"llamadeck/infra"
)

type searchResultMsg struct {
	models []hub.Model
	err    error
}
type topModelsMsg struct{ models []hub.Model }
type fitResultMsg struct {
	label, quant string
	model        *fit.Model
	res          *fit.Result
	quants       []string
	caps         []string
	err          error
}
type launchedMsg struct {
	name string
	port int
	err  error
}

type modelsTab struct {
	sh         *shared
	input      textinput.Model
	searching  bool
	searched   bool // user ran a search — don't clobber with the async top list
	results    []hub.Model
	sel        int
	top        int // scroll offset (index of first visible row)
	loading    bool
	preview    *fit.Result
	pModel     *fit.Model
	pRepo      string
	pQuant     string
	pQuants    []string // available quants for the previewed repo
	pCaps      []string // detected capabilities (vision/reasoning/…)
	confirming bool     // preview shown; next enter confirms → jump to Fit
	status     string
	errMsg     string
}

func newModelsTab(sh *shared) tab {
	ti := textinput.New()
	ti.Placeholder = "search Hugging Face for GGUF models…"
	ti.CharLimit = 80
	// Seed instantly with the curated set; Init fetches the live top-50 async.
	return &modelsTab{sh: sh, input: ti, results: hub.CuratedModels()}
}

func (t *modelsTab) Title() string { return "Models" }
func (t *modelsTab) Focused() bool { return t.searching }
func (t *modelsTab) Init() tea.Cmd {
	return func() tea.Msg { return topModelsMsg{models: hub.TopModels()} }
}

func (t *modelsTab) current() *hub.Model {
	if t.sel >= 0 && t.sel < len(t.results) {
		return &t.results[t.sel]
	}
	return nil
}

func (t *modelsTab) Update(msg tea.Msg) (tab, tea.Cmd) {
	switch msg := msg.(type) {
	case topModelsMsg:
		t.loading = false
		if t.searched || t.searching { // user moved on; don't clobber
			return t, nil
		}
		if len(msg.models) > 0 {
			t.results, t.sel, t.top = msg.models, 0, 0
		}
		return t, nil
	case searchResultMsg:
		t.loading = false
		t.searched = true
		if msg.err != nil {
			t.errMsg = msg.err.Error()
			return t, nil
		}
		t.results, t.sel, t.top, t.errMsg = msg.models, 0, 0, ""
		if len(t.results) == 0 {
			t.errMsg = "no GGUF repos matched"
		}
		return t, nil
	case fitResultMsg:
		t.loading = false
		if msg.err != nil {
			t.errMsg, t.confirming = msg.err.Error(), false
			return t, nil
		}
		t.preview, t.pModel, t.pRepo, t.pQuant, t.errMsg = msg.res, msg.model, msg.label, msg.quant, ""
		t.pQuants, t.pCaps, t.confirming = msg.quants, msg.caps, true // metadata shown — await confirm
		t.sh.selected = &selection{src: msg.label, quant: msg.quant, model: msg.model}
		return t, nil
	case launchedMsg:
		if msg.err != nil {
			t.status = stErr.Render("launch failed: " + msg.err.Error())
		} else {
			t.status = stOK.Render(fmt.Sprintf("launched %s on :%d — see the Monitor tab", msg.name, msg.port))
		}
		return t, nil
	case tea.KeyMsg:
		if t.searching {
			switch msg.String() {
			case "esc":
				t.searching = false
				t.input.Blur()
				return t, nil
			case "enter":
				t.searching = false
				t.input.Blur()
				q := strings.TrimSpace(t.input.Value())
				if q == "" {
					return t, nil
				}
				t.loading = true
				return t, searchCmd(q)
			}
			var cmd tea.Cmd
			t.input, cmd = t.input.Update(msg)
			return t, cmd
		}
		switch msg.String() {
		case "/":
			t.searching = true
			return t, t.input.Focus()
		case "esc":
			t.confirming, t.preview = false, nil // dismiss the metadata/confirm panel
		case "up", "k":
			if t.sel > 0 {
				t.sel--
				t.confirming = false // selection changed — preview is stale
			}
		case "down", "j":
			if t.sel < len(t.results)-1 {
				t.sel++
				t.confirming = false
			}
		case "t":
			t.searched, t.loading, t.confirming = false, true, false
			return t, func() tea.Msg { return topModelsMsg{models: hub.TopModels()} }
		case "enter":
			if t.confirming { // second enter on the previewed model → use it
				t.confirming = false
				return t, func() tea.Msg { return gotoTabMsg(tabFit) }
			}
			if c := t.current(); c != nil {
				t.loading, t.errMsg = true, ""
				return t, fitCmd(c.Repo, t.sh.hw, t.sh.cfg)
			}
		case "l":
			return t, t.launch()
		}
	}
	return t, nil
}

func (t *modelsTab) launch() tea.Cmd {
	if !t.sh.imageOK {
		t.status = stWarn.Render("build the server image first (Config tab)")
		return nil
	}
	c := t.current()
	if c == nil {
		return nil
	}
	// Prefer the quant resolved during the fit preview for this repo.
	hf := c.Repo
	if t.pRepo == c.Repo && t.pQuant != "" {
		hf = c.Repo + ":" + t.pQuant
	}
	port := infra.FreePort(8080)
	t.status = fmt.Sprintf("launching %s on :%d…", hf, port)
	ctx := t.sh.cfg.Ctx
	return func() tea.Msg {
		name, err := infra.Launch(infra.LaunchSpec{
			HF: hf, Port: port, Ctx: ctx, NGL: 999,
			CacheDir: cacheDir(), FlashAttn: true,
		})
		return launchedMsg{name: name, port: port, err: err}
	}
}

func searchCmd(q string) tea.Cmd {
	return func() tea.Msg {
		ms, err := hub.Search(q)
		return searchResultMsg{models: ms, err: err}
	}
}

func fitCmd(src string, hw fit.Hardware, cfg fit.Config) tea.Cmd {
	return func() tea.Msg {
		m, label, quant, err := hub.Load(src)
		if err != nil {
			return fitResultMsg{err: err}
		}
		res, err := fit.Predict(m, hw, cfg)
		// ponytail: a few separate HF metadata fetches (siblings, quants, caps)
		// for one on-demand preview. Fine at this cadence; fold into a single
		// RepoInfo call if previews ever feel slow.
		return fitResultMsg{label: label, quant: quant, model: m, res: res,
			quants: hub.Quants(src), caps: hub.Capabilities(src), err: err}
	}
}

func cacheDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h + "/.cache/huggingface"
	}
	return "./.models"
}

// splitWidths returns (leftW, rightW) for a total width. The name column is
// ~40% of the width, clamped to [28, 64] so names grow with the window up to a
// long repo name, then extra width flows to the description. 3 columns go to the
// " │ " divider. On a narrow terminal the description keeps a usable minimum.
func splitWidths(width int) (leftW, rightW int) {
	leftW = width * 2 / 5
	if leftW < 28 {
		leftW = 28
	}
	if leftW > 64 {
		leftW = 64
	}
	rightW = width - leftW - 3
	if rightW < 24 {
		leftW = width - 24 - 3
		if leftW < 16 {
			leftW = 16
		}
		rightW = width - leftW - 3
	}
	return leftW, rightW
}

func (t *modelsTab) View(width, height int) string {
	leftW, rightW := splitWidths(width)

	// --- search line ---
	var head string
	if t.searching {
		head = t.input.View()
	} else {
		head = stMuted.Render("press ") + stKey.Render("/") + stMuted.Render(" to search Hugging Face") +
			stMuted.Render("   ·   ") + stKey.Render("t") + stMuted.Render(" top models")
	}

	// --- left: scrollable list ---
	// Each entry is up to 2 lines (label + meta). Keep the selected row in view
	// and only render the window that fits, with ↑/↓ "more" affordances.
	const perEntry = 2
	avail := height - 4 // head (2) + a line for each scroll marker
	if avail < perEntry {
		avail = perEntry
	}
	visN := avail / perEntry
	if visN < 1 {
		visN = 1
	}
	if t.sel < t.top {
		t.top = t.sel
	}
	if t.sel >= t.top+visN {
		t.top = t.sel - visN + 1
	}
	if t.top < 0 {
		t.top = 0
	}
	end := t.top + visN
	if end > len(t.results) {
		end = len(t.results)
	}

	var lb strings.Builder
	if t.top > 0 {
		lb.WriteString(stMuted.Render(fmt.Sprintf("  ↑ %d more", t.top)) + "\n")
	}
	for i := t.top; i < end; i++ {
		m := t.results[i]
		label := m.Repo
		meta := m.Note
		if meta == "" && m.Downloads > 0 {
			meta = fmt.Sprintf("↓%s", compactNum(m.Downloads))
		}
		var row string
		if i == t.sel {
			row = stSelected.Render("› " + pad(truncate(label, leftW-3), leftW-3))
		} else {
			row = "  " + stText.Render(truncate(label, leftW-3))
		}
		lb.WriteString(row + "\n")
		if meta != "" {
			lb.WriteString("    " + stMuted.Render(truncate(meta, leftW-5)) + "\n")
		}
	}
	if end < len(t.results) {
		lb.WriteString(stMuted.Render(fmt.Sprintf("  ↓ %d more", len(t.results)-end)) + "\n")
	}
	left := lb.String()

	// --- right: fit preview ---
	right := t.previewPanel(rightW)

	cols := lipgloss.JoinHorizontal(lipgloss.Top,
		lipgloss.NewStyle().Width(leftW).Render(left),
		stDim.Render(" │ "),
		lipgloss.NewStyle().Width(rightW).Render(right))

	out := head + "\n\n" + cols
	if t.status != "" {
		out += "\n" + t.status
	}
	return out
}

func (t *modelsTab) previewPanel(w int) string {
	if t.loading {
		return stMuted.Render("loading…")
	}
	if t.errMsg != "" {
		return stErr.Render(t.errMsg)
	}
	if t.preview == nil || t.pModel == nil {
		return stMuted.Render("select a model and press ") + stKey.Render("enter") +
			stMuted.Render(" to see its details,\nconfirm, and jump to the Fit tab.")
	}
	r := t.preview
	name := t.pRepo
	if t.pQuant != "" {
		name += "  " + stKey.Render(t.pQuant)
	}
	verdict := verdictBadge(r.Mode)
	lines := []string{
		stBold.Width(w).Render(name),
		stMuted.Render(fmt.Sprintf("%s · %d layers · vocab %d", t.pModel.Arch, t.pModel.NLayers, t.pModel.VocabSize)),
		"",
		verdict + "   " + stText.Render(fmt.Sprintf("ngl %d/%d", r.LayersOnGPU, r.NLayers)),
		"",
		fmt.Sprintf("%s %s", stMuted.Render("total VRAM"), human(r.VRAMUsed)+" / "+human(t.sh.hw.FreeVRAM)),
		fmt.Sprintf("%s %s", stMuted.Render("total RAM "), human(r.RAMUsed)+" / "+human(t.sh.hw.FreeRAM)),
		stMuted.Render(fmt.Sprintf("weights %s · kv %s · compute %s @ ctx %d",
			human(r.WeightsBytes), human(r.KVBytes), human(r.ComputeBytes), t.sh.cfg.Ctx)),
	}
	if len(t.pCaps) > 0 {
		lines = append(lines, stMuted.Render("caps:"),
			stOK.Width(w).Render(strings.Join(t.pCaps, " · ")))
	}
	if len(t.pQuants) > 0 {
		lines = append(lines, "", stMuted.Render("quants:"),
			stText.Width(w).Render(strings.Join(t.pQuants, "  ")))
	}
	if t.confirming {
		lines = append(lines, "",
			stKey.Render("enter")+stText.Render(" use this model → Fit")+stMuted.Render("   ·   ")+
				stKey.Render("esc")+stMuted.Render(" back"))
	}
	return strings.Join(lines, "\n")
}

func verdictBadge(m fit.Mode) string {
	bg := cGreen
	switch m {
	case fit.ModeHybrid:
		bg = cCyan
	case fit.ModeCPU:
		bg = cYellow
	case fit.ModeOOM:
		bg = cRed
	}
	return lipgloss.NewStyle().Background(bg).Foreground(cBg).Bold(true).Padding(0, 1).Render(string(m))
}

func compactNum(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1e3)
	}
	return fmt.Sprintf("%d", n)
}
