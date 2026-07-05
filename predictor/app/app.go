// Package app is the unified operational TUI: a tabbed, nvtop-style control
// center over the fit predictor, model discovery, container monitoring, and image
// management. Each tab is a small self-contained model; the root routes keys and
// broadcasts async results.
package app

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
	"llamadeck/infra"
)

// Tab indices — the order tabs are built in New(). Used for jump-to-tab.
const (
	tabModels = iota
	tabFit
	tabMonitor
	tabConfig
)

// gotoTabMsg asks the root to switch the active tab (e.g. Models → Fit after a
// model is confirmed). Handled in the root, not broadcast.
type gotoTabMsg int

// tab is one screen of the app. Async result messages are broadcast to every
// tab; key messages go only to the active one.
type tab interface {
	Init() tea.Cmd
	Update(msg tea.Msg) (tab, tea.Cmd)
	View(width, height int) string
	Title() string
	Focused() bool // true when a text input owns the keyboard
}

// shared is global state every tab can read (hardware, selected model, config).
type shared struct {
	hw       fit.Hardware
	gpuFree  []int64 // per-device free VRAM (bytes), for single-GPU pinning
	cfg      fit.Config
	dockerOK bool
	imageOK  bool
	cpuMode  bool // no-GPU mode: predict against RAM only, launch without --gpus
	selected *selection // model chosen in Models, used by Fit
}

type selection struct {
	src   string // owner/repo[:quant]
	quant string
	model *fit.Model
}

// Model is the root Bubble Tea model.
type Model struct {
	tabs   []tab
	active int
	w, h   int
	sh     *shared
}

// New builds the app with probed hardware and calibration applied.
func New() Model {
	dockerOK := infra.DockerAvailable()
	gpuCount, totalVRAM := infra.GPUSummary()
	sh := &shared{
		hw:       fit.Hardware{FreeVRAM: totalVRAM, FreeRAM: infra.FreeRAM(), NumGPUs: gpuCount},
		cfg:      fit.DefaultConfig(),
		dockerOK: dockerOK,
		imageOK:  dockerOK && infra.ServerImageExists(),
	}
	sh.cpuMode = gpuCount == 0 // no GPU detected → CPU mode by default
	if p, err := fit.LoadProfile(); err == nil {
		p.Apply(&sh.cfg)
	}
	sh.cfg.Ctx = 8192
	m := Model{sh: sh}
	m.tabs = []tab{
		tabModels:  newModelsTab(sh),
		tabFit:     newFitTab(sh),
		tabMonitor: newMonitorTab(sh),
		tabConfig:  newUpdateTab(sh),
	}
	return m
}

func (m Model) Init() tea.Cmd {
	var cmds []tea.Cmd
	for _, t := range m.tabs {
		if c := t.Init(); c != nil {
			cmds = append(cmds, c)
		}
	}
	return tea.Batch(cmds...)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.w, m.h = msg.Width, msg.Height
		return m.broadcast(msg)

	case gotoTabMsg:
		if i := int(msg); i >= 0 && i < len(m.tabs) {
			m.active = i
		}
		return m, nil

	case tea.KeyMsg:
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
		if !m.tabs[m.active].Focused() {
			switch msg.String() {
			case "q":
				return m, tea.Quit
			case "tab":
				m.active = (m.active + 1) % len(m.tabs)
				return m, nil
			case "shift+tab":
				m.active = (m.active - 1 + len(m.tabs)) % len(m.tabs)
				return m, nil
			case "1", "2", "3", "4":
				if i := int(msg.String()[0] - '1'); i < len(m.tabs) {
					m.active = i
				}
				return m, nil
			}
		}
		nt, cmd := m.tabs[m.active].Update(msg)
		m.tabs[m.active] = nt
		return m, cmd

	default:
		return m.broadcast(msg)
	}
}

// broadcast delivers a message to every tab (async results, ticks, resize).
func (m Model) broadcast(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	for i := range m.tabs {
		nt, c := m.tabs[i].Update(msg)
		m.tabs[i] = nt
		if c != nil {
			cmds = append(cmds, c)
		}
	}
	return m, tea.Batch(cmds...)
}

func (m Model) View() string {
	if m.w == 0 {
		return "starting…"
	}
	titles := make([]string, len(m.tabs))
	for i, t := range m.tabs {
		titles[i] = t.Title()
	}

	header := lipgloss.JoinHorizontal(lipgloss.Top,
		stTitle.Render("llamadeck"), "  ", tabBar(titles, m.active))

	banner := ""
	switch {
	case !m.sh.dockerOK:
		banner = stWarn.Render("  ⚠ Docker not found — fit prediction works; launch / monitor / build need Docker\n")
	case !m.sh.imageOK:
		banner = stWarn.Render("  ⚠ server image not built — go to the Config tab (4) to build it\n")
	}

	bodyH := m.h - 4
	if banner != "" {
		bodyH--
	}
	body := m.tabs[m.active].View(m.w, bodyH)

	footer := stMuted.Render(m.footerHelp())

	return header + "\n" + banner + body + "\n" + footer
}

func (m Model) footerHelp() string {
	base := "tab/1-4 switch · q quit"
	switch m.active {
	case 0:
		return "/ search · ↑↓ select · enter details→use · l launch · " + base
	case 1:
		return "↑↓ field · ←→ adjust · e edit · c quant · m cpu · enter launch · " + base
	case 2:
		return "↑↓ select · p plug into agent · s stop · x remove · r refresh · " + base
	case 3:
		return "b build · r rebuild · d dev-fork · k set HF key · " + base
	}
	return base
}

// Run starts the full-screen app (requires a TTY).
func Run() error {
	p := tea.NewProgram(New(), tea.WithAltScreen())
	_, err := p.Run()
	return err
}

// RenderOnce renders a single static frame of the app at the given size, for
// non-TTY contexts (screenshots, smoke tests). active selects the starting tab.
func RenderOnce(active, w, h int) string {
	m := New()
	if active >= 0 && active < len(m.tabs) {
		m.active = active
	}
	m.w, m.h = w, h
	// Drive each tab's initial load synchronously so the static frame has data.
	for i := range m.tabs {
		if c := m.tabs[i].Init(); c != nil {
			if msg := c(); msg != nil {
				nt, _ := m.tabs[i].Update(msg)
				m.tabs[i] = nt
			}
		}
	}
	return fmt.Sprint(m.View())
}
