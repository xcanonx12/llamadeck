package app

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"llamadeck/fit"
)

// human formats a byte count (GiB/MiB), shared by all tabs.
func human(b int64) string { return fit.HumanBytes(b) }

// Catppuccin-ish palette, shared across all tabs for a consistent product feel.
var (
	cBg      = lipgloss.Color("#11111B")
	cText    = lipgloss.Color("#CDD6F4")
	cMuted   = lipgloss.Color("#9399B2")
	cDim     = lipgloss.Color("#45475A")
	cAccent  = lipgloss.Color("#89B4FA") // blue
	cGreen   = lipgloss.Color("#A6E3A1")
	cCyan    = lipgloss.Color("#89DCEB")
	cYellow  = lipgloss.Color("#F9E2AF")
	cRed     = lipgloss.Color("#F38BA8")
	cMauve   = lipgloss.Color("#CBA6F7")
	cPeach   = lipgloss.Color("#FAB387")
	cSurface = lipgloss.Color("#313244")

	stText  = lipgloss.NewStyle().Foreground(cText)
	stMuted = lipgloss.NewStyle().Foreground(cMuted)
	stDim   = lipgloss.NewStyle().Foreground(cDim)
	stBold  = lipgloss.NewStyle().Foreground(cText).Bold(true)
	stKey   = lipgloss.NewStyle().Foreground(cCyan).Bold(true)
	stWarn  = lipgloss.NewStyle().Foreground(cYellow)
	stErr   = lipgloss.NewStyle().Foreground(cRed).Bold(true)
	stOK    = lipgloss.NewStyle().Foreground(cGreen)

	stTitle = lipgloss.NewStyle().Foreground(cBg).Background(cAccent).Bold(true).Padding(0, 1)

	stTabActive   = lipgloss.NewStyle().Foreground(cBg).Background(cAccent).Bold(true).Padding(0, 2)
	stTabInactive = lipgloss.NewStyle().Foreground(cMuted).Padding(0, 2)

	stSelected = lipgloss.NewStyle().Foreground(cBg).Background(cSurface)
)

// tabBar renders the top navigation strip.
func tabBar(titles []string, active int) string {
	var cells []string
	for i, t := range titles {
		label := fmt.Sprintf("%d %s", i+1, t)
		if i == active {
			cells = append(cells, stTabActive.Render(label))
		} else {
			cells = append(cells, stTabInactive.Render(label))
		}
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, cells...)
}

// bar renders an htop-style proportional bar: used/total, colored by fill ratio.
func bar(width int, frac float64, c lipgloss.Color) string {
	if width < 1 {
		width = 1
	}
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	fill := int(frac*float64(width) + 0.5)
	if fill > width {
		fill = width
	}
	full := lipgloss.NewStyle().Foreground(c).Render(strings.Repeat("█", fill))
	rest := stDim.Render(strings.Repeat("░", width-fill))
	return full + rest
}

// statusColor maps a free-form health/state string to a palette color.
func statusColor(s string) lipgloss.Color {
	switch s {
	case "healthy", "running":
		return cGreen
	case "starting":
		return cYellow
	case "unhealthy", "exited", "dead":
		return cRed
	default:
		return cMuted
	}
}

// pad right-pads (or truncates) a styled-free string to n display cells.
func pad(s string, n int) string {
	w := lipgloss.Width(s)
	if w > n {
		return truncate(s, n)
	}
	return s + strings.Repeat(" ", n-w)
}

func truncate(s string, n int) string {
	if lipgloss.Width(s) <= n {
		return s
	}
	r := []rune(s)
	if n <= 1 {
		return string(r[:n])
	}
	return string(r[:n-1]) + "…"
}
