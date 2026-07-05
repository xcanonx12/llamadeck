package app

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"llamadeck/infra"
)

type monitorDataMsg struct {
	containers []infra.Container
	gpus       []infra.GPU
	logs       map[string]string // container name → latest progress line (non-healthy only)
	err        error
}
type monTickMsg struct{}
type monAnimMsg struct{} // fast frame tick, just advances the spinner
type monActionMsg struct{ err error }

type monitorTab struct {
	sh         *shared
	containers []infra.Container
	gpus       []infra.GPU
	logs       map[string]string
	spin       int
	animOn     bool // a spinner-frame loop is currently scheduled
	sel        int
	status     string
	plugState  // the `p = plug` overlay (monitorplug.go)
}

// spinFrames is the braille spinner shown next to a starting/loading server.
var spinFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// animInterval drives the launching/loading spinner. Fast enough to read as
// motion (the 2s data poll is far too slow to animate on).
const animInterval = 90 * time.Millisecond

// pctRe pulls a percentage (e.g. "42.1%") out of a download/load progress line.
var pctRe = regexp.MustCompile(`(\d{1,3}(?:\.\d+)?)\s*%`)

func parsePercent(s string) (float64, bool) {
	// The downloader redraws progress with carriage returns, so a single captured
	// line can be "10%\r…\r63%" — take the LAST percentage, i.e. the current one.
	ms := pctRe.FindAllStringSubmatch(s, -1)
	for i := len(ms) - 1; i >= 0; i-- {
		if v, err := strconv.ParseFloat(ms[i][1], 64); err == nil && v >= 0 && v <= 100 {
			return v / 100, true
		}
	}
	return 0, false
}

// spinning reports whether a container is mid-startup (loading/downloading) — the
// state that gets an animated spinner, and that keeps the fast anim loop alive.
func spinning(c infra.Container) bool {
	return c.State == "running" && c.Health != "healthy" && c.Health != "n/a"
}

func (t *monitorTab) anySpinning() bool {
	for _, c := range t.containers {
		if spinning(c) {
			return true
		}
	}
	return false
}

func newMonitorTab(sh *shared) tab { return &monitorTab{sh: sh} }

func (t *monitorTab) Title() string { return "Monitor" }
func (t *monitorTab) Focused() bool { return false }
func (t *monitorTab) Init() tea.Cmd { return monLoad }

func monAnim() tea.Cmd {
	return tea.Tick(animInterval, func(time.Time) tea.Msg { return monAnimMsg{} })
}

func monLoad() tea.Msg {
	cs, err := infra.Containers()
	// Pull a progress line only for servers still coming up (downloading the
	// GGUF or loading) — healthy ones need no log spam.
	logs := map[string]string{}
	for _, c := range cs {
		// Anything not cleanly healthy-and-running gets a log line: a starting
		// server shows download/load progress, a crashed/exited one shows WHY
		// (e.g. "failed to allocate compute buffers").
		if c.State != "running" || c.Health != "healthy" {
			if ln := lastLine(infra.TailLog(c.Name, 30)); ln != "" {
				logs[c.Name] = ln
			}
		}
	}
	return monitorDataMsg{containers: cs, gpus: infra.GPUs(), logs: logs, err: err}
}

// lastLine returns the last non-blank line of s.
func lastLine(s string) string {
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if t := strings.TrimSpace(lines[i]); t != "" {
			return t
		}
	}
	return ""
}
func monTick() tea.Cmd {
	return tea.Tick(2*time.Second, func(time.Time) tea.Msg { return monTickMsg{} })
}

func (t *monitorTab) Update(msg tea.Msg) (tab, tea.Cmd) {
	// The plug overlay gets first look while active; data ticks fall through so
	// the tab underneath stays fresh.
	if t.plugMode != monPlugOff {
		if nt, cmd, handled := t.plugUpdate(msg); handled {
			return nt, cmd
		}
	}
	switch msg := msg.(type) {
	case monitorDataMsg:
		t.containers, t.gpus, t.logs = msg.containers, msg.gpus, msg.logs
		t.refreshHW() // keep shared free VRAM/RAM live for the Fit tab
		if t.sel >= len(t.containers) {
			t.sel = max(0, len(t.containers)-1)
		}
		cmds := []tea.Cmd{monTick()} // schedule the next data refresh
		// Start the fast spinner loop when something is loading (and isn't already
		// running); it stops itself once everything is healthy/idle.
		if t.anySpinning() && !t.animOn {
			t.animOn = true
			cmds = append(cmds, monAnim())
		}
		return t, tea.Batch(cmds...)
	case monAnimMsg:
		t.spin = (t.spin + 1) % len(spinFrames)
		if t.anySpinning() {
			return t, monAnim()
		}
		t.animOn = false // nothing loading — let the frame loop go idle
		return t, nil
	case monTickMsg:
		return t, monLoad
	case monActionMsg:
		if msg.err != nil {
			t.status = "error: " + msg.err.Error()
		}
		return t, monLoad
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if t.sel > 0 {
				t.sel--
			}
		case "down", "j":
			if t.sel < len(t.containers)-1 {
				t.sel++
			}
		case "r":
			return t, monLoad
		case "s":
			if c := t.current(); c != nil {
				name := c.Name
				t.status = "stopping " + name + "…"
				return t, func() tea.Msg { return monActionMsg{infra.Stop(name)} }
			}
		case "x":
			if c := t.current(); c != nil {
				name := c.Name
				t.status = "removing " + name + "…"
				return t, func() tea.Msg { return monActionMsg{infra.Remove(name)} }
			}
		case "p":
			if c := t.current(); c != nil {
				return t.startPlug(*c)
			}
		}
	}
	return t, nil
}

// refreshHW updates the shared hardware snapshot from the latest GPU poll so the
// Fit tab reflects VRAM freed/used by launching or dropping containers, instead
// of the static boot-time value. Skips when no GPU was probed this tick.
func (t *monitorTab) refreshHW() {
	if len(t.gpus) == 0 {
		return
	}
	var free int64
	gf := make([]int64, len(t.gpus))
	for i, g := range t.gpus {
		f := g.MemFree
		if f == 0 {
			f = g.MemTotal - g.MemUsed
		}
		gf[i] = f
		free += f
	}
	t.sh.hw.FreeVRAM = free
	t.sh.hw.NumGPUs = len(t.gpus)
	t.sh.gpuFree = gf
	t.sh.hw.FreeRAM = infra.FreeRAM()
}

func (t *monitorTab) current() *infra.Container {
	if t.sel >= 0 && t.sel < len(t.containers) {
		return &t.containers[t.sel]
	}
	return nil
}

func (t *monitorTab) View(width, height int) string {
	if t.plugMode != monPlugOff {
		return t.plugView(width, height)
	}
	var b strings.Builder
	barW := min(width-28, 40)
	if barW < 10 {
		barW = 10
	}

	// --- GPU section ---
	b.WriteString(stBold.Render("GPU") + "\n")
	if len(t.gpus) == 0 {
		b.WriteString(stMuted.Render("  no NVIDIA GPU detected\n"))
	}
	for _, g := range t.gpus {
		memFrac := 0.0
		if g.MemTotal > 0 {
			memFrac = float64(g.MemUsed) / float64(g.MemTotal)
		}
		memC := cGreen
		if memFrac > 0.9 {
			memC = cRed
		} else if memFrac > 0.7 {
			memC = cYellow
		}
		b.WriteString(fmt.Sprintf("  %s\n", stText.Render(g.Name)))
		b.WriteString(fmt.Sprintf("  mem  %s %s\n", bar(barW, memFrac, memC),
			stMuted.Render(fmt.Sprintf("%s / %s", human(g.MemUsed), human(g.MemTotal)))))
		b.WriteString(fmt.Sprintf("  util %s %s\n", bar(barW, float64(g.UtilPct)/100, cMauve),
			stMuted.Render(fmt.Sprintf("%d%%", g.UtilPct))))
	}

	// --- containers section ---
	b.WriteString("\n" + stBold.Render("Servers") + "\n")
	if !t.sh.dockerOK {
		b.WriteString(stWarn.Render("  Docker not available — install/start Docker to launch & monitor servers\n"))
	} else if len(t.containers) == 0 {
		b.WriteString(stMuted.Render("  no managed servers running — launch one from the Models tab\n"))
	} else {
		b.WriteString(stMuted.Render(fmt.Sprintf("  %s %s %s %s",
			pad("NAME", 34), pad("STATE", 10), pad("HEALTH", 10), "PORT")) + "\n")
		for i, c := range t.containers {
			line := fmt.Sprintf("%s %s %s %s",
				pad(c.Name, 34),
				lipgloss.NewStyle().Foreground(statusColor(c.State)).Render(pad(c.State, 10)),
				lipgloss.NewStyle().Foreground(statusColor(c.Health)).Render(pad(c.Health, 10)),
				c.Port)
			if i == t.sel {
				line = stSelected.Render("› " + line)
			} else {
				line = "  " + line
			}
			b.WriteString(line + "\n")

			// Plain-English "what's going on / can I use it": HEALTH alone is jargon.
			b.WriteString("    " + healthNote(c) + "\n")

			ln := t.logs[c.Name]
			switch {
			case c.State == "exited" || c.State == "dead":
				if ln != "" { // the crash reason
					b.WriteString(stErr.Render(fmt.Sprintf("    ✗ %s", truncate(ln, width-8))) + "\n")
				}
			case spinning(c):
				// Loading: show a real progress bar if the log line has a %, else
				// the spinner + latest line (download/load step).
				if frac, ok := parsePercent(ln); ok {
					b.WriteString(fmt.Sprintf("    %s %s %s\n",
						spinFrames[t.spin], bar(min(width-30, 30), frac, cCyan),
						stMuted.Render(fmt.Sprintf("%.0f%%  downloading", frac*100))))
				} else if ln != "" {
					b.WriteString(stMuted.Render(fmt.Sprintf("    %s %s", spinFrames[t.spin], truncate(ln, width-8))) + "\n")
				}
			}
		}
	}
	if t.status != "" {
		b.WriteString("\n" + stMuted.Render(t.status) + "\n")
	}
	return b.String()
}

// healthNote turns Docker's terse STATE/HEALTH into a one-line plain-English
// answer to "can I use this server yet?".
func healthNote(c infra.Container) string {
	url := ""
	if c.Port != "" {
		url = " on http://localhost:" + c.Port
	}
	switch {
	case c.State == "exited" || c.State == "dead":
		return stErr.Render("stopped — not serving (see the reason below; press r to refresh)")
	case c.State != "running":
		return stMuted.Render(c.State + "…")
	case c.Health == "healthy":
		return stOK.Render("✓ ready" + url + " — the model is loaded and answering requests")
	case c.Health == "starting":
		return stWarn.Render("loading… downloading / loading the model (first run can take minutes) — not ready yet")
	case c.Health == "unhealthy":
		return stErr.Render("⚠ not answering /health yet — still loading, or the server errored; check the line below")
	default: // "n/a" — no healthcheck defined
		return stMuted.Render("running" + url + " — no healthcheck, so readiness is unknown")
	}
}
