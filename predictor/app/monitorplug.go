package app

// The Monitor tab's `p = plug` overlay: pick a coding agent for the selected
// running server, see the exact provider config, copy it (y) or write it into
// the agent's config (w → y/N confirm, timestamped backup). Thin TUI over the
// plug package — the CLI (`llamadeck plug`) and this share every fact and all
// of the write-safety machinery.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/atotto/clipboard"
	tea "github.com/charmbracelet/bubbletea"

	"llamadeck/infra"
	"llamadeck/plug"
)

// monitor plug-mode states (monitorTab.plugMode).
const (
	monPlugOff     = iota
	monPlugProbe   // probing the selected server
	monPlugPick    // agent picker
	monPlugShow    // snippet view (w write · y copy · esc back)
	monPlugConfirm // y/N gate before writing
)

type plugAgentEntry struct {
	agent    plug.Agent
	detected bool
}

type monPlugSrvMsg struct {
	srv plug.Server
	ok  bool
}

type monPlugWriteMsg struct {
	path, backup string
	err          error
}

// plugState is the overlay's state, embedded in monitorTab.
type plugState struct {
	plugMode   int
	plugSrv    plug.Server
	plugAgents []plugAgentEntry
	plugSel    int
	plugNote   string // transient result line inside the overlay (copied / written / error)
}

// startPlug begins the flow for the selected container: probe it (async — two
// HTTP probes + a docker inspect), then open the agent picker.
func (t *monitorTab) startPlug(c infra.Container) (tab, tea.Cmd) {
	if c.State != "running" || c.Port == "" {
		t.status = "plug: server must be running (with a port) — pick a running one"
		return t, nil
	}
	t.plugMode, t.plugNote = monPlugProbe, ""
	cc := c
	return t, func() tea.Msg {
		srv, ok := plug.ProbeContainer(cc)
		return monPlugSrvMsg{srv: srv, ok: ok}
	}
}

// plugUpdate handles messages while the overlay is active. Returns handled=false
// for messages the normal Monitor loop should still process (data ticks keep
// flowing underneath so the tab is fresh when the overlay closes).
func (t *monitorTab) plugUpdate(msg tea.Msg) (tab, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case monPlugSrvMsg:
		if !msg.ok {
			t.plugMode, t.status = monPlugOff, "plug: server stopped while probing"
			return t, nil, true
		}
		t.plugSrv = msg.srv
		// Detected agents first, original order otherwise (stable).
		entries := make([]plugAgentEntry, 0, 6)
		for _, a := range plug.Agents() {
			entries = append(entries, plugAgentEntry{agent: a, detected: a.Detected()})
		}
		sort.SliceStable(entries, func(i, j int) bool { return entries[i].detected && !entries[j].detected })
		t.plugAgents, t.plugSel, t.plugMode = entries, 0, monPlugPick
		return t, nil, true

	case monPlugWriteMsg:
		if msg.err != nil {
			t.plugNote = stErr.Render("✗ " + msg.err.Error())
		} else {
			note := "✓ written to " + msg.path
			if msg.backup != "" {
				note += "   (backup: " + msg.backup + ")"
			}
			t.plugNote = stOK.Render(note)
		}
		t.plugMode = monPlugShow
		return t, nil, true

	case tea.KeyMsg:
		if t.plugMode == monPlugOff || t.plugMode == monPlugProbe {
			return t, nil, false
		}
		return t.plugKey(msg.String())
	}
	return t, nil, false
}

func (t *monitorTab) plugKey(key string) (tab, tea.Cmd, bool) {
	switch t.plugMode {
	case monPlugPick:
		switch key {
		case "up", "k":
			if t.plugSel > 0 {
				t.plugSel--
			}
		case "down", "j":
			if t.plugSel < len(t.plugAgents)-1 {
				t.plugSel++
			}
		case "enter":
			t.plugMode, t.plugNote = monPlugShow, ""
		case "esc", "q":
			t.plugMode = monPlugOff
		}
		return t, nil, true

	case monPlugShow:
		a := t.plugAgents[t.plugSel].agent
		switch key {
		case "y":
			if err := clipboard.WriteAll(a.Snippet(t.plugSrv)); err != nil {
				t.plugNote = stErr.Render("copy failed: " + err.Error() + " — select the text manually")
			} else {
				t.plugNote = stOK.Render("✓ snippet copied to clipboard")
			}
		case "w":
			if a.CanWrite {
				t.plugMode, t.plugNote = monPlugConfirm, ""
			} else {
				t.plugNote = stWarn.Render(a.Name + " has no writable config — follow the steps above (y copies them)")
			}
		case "esc", "q":
			t.plugMode, t.plugNote = monPlugPick, ""
		}
		return t, nil, true

	case monPlugConfirm:
		a := t.plugAgents[t.plugSel].agent
		switch key {
		case "y":
			srv, path := t.plugSrv, a.Path(false)
			t.plugMode = monPlugShow
			t.plugNote = stMuted.Render("writing…")
			return t, func() tea.Msg {
				backup, err := a.Write(srv, path)
				return monPlugWriteMsg{path: path, backup: backup, err: err}
			}, true
		case "n", "esc", "q":
			t.plugMode, t.plugNote = monPlugShow, stMuted.Render("nothing changed")
		}
		return t, nil, true
	}
	return t, nil, false
}

// plugView renders the overlay (replaces the Monitor body while active).
func (t *monitorTab) plugView(width, height int) string {
	var b strings.Builder
	switch t.plugMode {
	case monPlugProbe:
		return stMuted.Render("\n  probing server…")

	case monPlugPick:
		b.WriteString(stBold.Render(fmt.Sprintf("Plug %s (:%s, ctx %d) into…",
			t.plugSrv.ModelID, t.plugSrv.Port, t.plugSrv.Ctx)) + "\n\n")
		for i, e := range t.plugAgents {
			mark, note := " ", ""
			if e.detected {
				mark, note = stOK.Render("✓"), stMuted.Render("  installed")
			}
			line := fmt.Sprintf("%s %s%s", mark, pad(e.agent.Name, 14), note)
			if i == t.plugSel {
				line = stSelected.Render("› " + line)
			} else {
				line = "  " + line
			}
			b.WriteString(line + "\n")
		}
		b.WriteString("\n" + stMuted.Render("↑↓ select · ") + stKey.Render("enter") +
			stMuted.Render(" show config · ") + stKey.Render("esc") + stMuted.Render(" cancel"))

	case monPlugShow, monPlugConfirm:
		a := t.plugAgents[t.plugSel].agent
		path := ""
		if a.CanWrite {
			path = a.Path(false)
		}
		b.WriteString(stBold.Render(fmt.Sprintf("%s ← %s (:%s)", a.Name, t.plugSrv.ModelID, t.plugSrv.Port)) + "\n\n")
		b.WriteString(stText.Render(a.Snippet(t.plugSrv)) + "\n\n")
		b.WriteString(stMuted.Render(a.HowTo(t.plugSrv, path)) + "\n")
		if !t.plugSrv.Jinja {
			b.WriteString(stWarn.Render("⚠ launched WITHOUT --jinja — tool calling will misbehave; relaunch with Jinja on") + "\n")
		}
		if t.plugNote != "" {
			b.WriteString("\n" + t.plugNote + "\n")
		}
		if t.plugMode == monPlugConfirm {
			b.WriteString("\n" + stBold.Render(fmt.Sprintf("write this into %s?", path)) + "  " +
				stKey.Render("y") + stText.Render(" write") + stMuted.Render(" · ") +
				stKey.Render("n") + stMuted.Render(" cancel"))
		} else {
			keys := stMuted.Render("· ") + stKey.Render("y") + stMuted.Render(" copy · ") +
				stKey.Render("esc") + stMuted.Render(" back")
			if a.CanWrite {
				keys = stKey.Render("w") + stMuted.Render(" write ") + keys
			}
			b.WriteString("\n" + keys)
		}
	}
	return clampLines(b.String(), height)
}

// clampLines keeps the overlay inside the tab's height budget.
func clampLines(s string, height int) string {
	if height <= 0 {
		return s
	}
	lines := strings.Split(s, "\n")
	if len(lines) <= height {
		return s
	}
	return strings.Join(lines[:height-1], "\n") + "\n" + stMuted.Render("… (terminal too short for the full snippet — y copies all of it)")
}
