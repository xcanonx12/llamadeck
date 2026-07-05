package tui

import (
	tea "github.com/charmbracelet/bubbletea"
)

const (
	minCtx    = 512
	maxCtx    = 1 << 20
	minUBatch = 64
	maxUBatch = 4096
)

type model struct{ st State }

// New builds a Bubble Tea model around an initial state.
func New(st State) tea.Model {
	st.Recompute()
	return model{st: st}
}

func (m model) Init() tea.Cmd { return nil }

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.st.Width = msg.Width - 6 // account for border + padding
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		case "right", "l", "+", "=":
			if m.st.Cfg.Ctx < maxCtx {
				m.st.Cfg.Ctx *= 2
				m.st.Recompute()
			}
		case "left", "h", "-", "_":
			if m.st.Cfg.Ctx > minCtx {
				m.st.Cfg.Ctx /= 2
				m.st.Recompute()
			}
		case "up", "k":
			if m.st.Cfg.UBatch < maxUBatch {
				m.st.Cfg.UBatch *= 2
				m.st.Recompute()
			}
		case "down", "j":
			if m.st.Cfg.UBatch > minUBatch {
				m.st.Cfg.UBatch /= 2
				m.st.Recompute()
			}
		}
	}
	return m, nil
}

func (m model) View() string { return RenderView(m.st) }

// Run starts the interactive TUI (requires a TTY).
func Run(st State) error {
	p := tea.NewProgram(New(st), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
