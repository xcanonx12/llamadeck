package app

// Plug-overlay flow tests: mode transitions, picker rendering, and the
// write-confirm gate — all with an injected server (no docker, no network).

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"llamadeck/plug"
)

func key(s string) tea.KeyMsg {
	if len(s) == 1 {
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	}
	panic(s)
}

func pluggedMonitor(t *testing.T) *monitorTab {
	t.Helper()
	lipgloss.SetColorProfile(0)
	mt := &monitorTab{sh: &shared{}}
	// Inject the probed server (what monPlugSrvMsg would deliver).
	nt, _, handled := mt.plugUpdate(monPlugSrvMsg{ok: true, srv: plug.Server{
		Name: "llama-x", Port: "8123", ModelID: "llama-1b-local", Ctx: 32768, Jinja: true,
	}})
	if !handled {
		t.Fatal("probe result must be handled")
	}
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugPick || len(mt.plugAgents) != 6 {
		t.Fatalf("expected picker with 6 agents, got mode=%d n=%d", mt.plugMode, len(mt.plugAgents))
	}
	return mt
}

func TestPlugOverlayFlow(t *testing.T) {
	mt := pluggedMonitor(t)

	// Detected agents sort first (stable) — verify ordering invariant.
	for i := 1; i < len(mt.plugAgents); i++ {
		if mt.plugAgents[i].detected && !mt.plugAgents[i-1].detected {
			t.Fatal("detected agents must sort before undetected ones")
		}
	}

	// Picker renders every agent + the server identity.
	view := mt.plugView(100, 40)
	for _, want := range []string{"llama-1b-local", "Claude Code", "Cursor", "Hermes"} {
		if !strings.Contains(view, want) {
			t.Fatalf("picker missing %q:\n%s", want, view)
		}
	}

	// enter → snippet view with the shared plug facts + key hints.
	nt, _, _ := mt.plugUpdate(key("enter"))
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugShow {
		t.Fatalf("enter should show the snippet, mode=%d", mt.plugMode)
	}
	view = mt.plugView(100, 60)
	if !strings.Contains(view, ":8123") || !strings.Contains(view, "copy") {
		t.Fatalf("snippet view incomplete:\n%s", view)
	}

	// esc → back to picker; esc → overlay closed.
	nt, _, _ = mt.plugUpdate(key("esc"))
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugPick {
		t.Fatal("esc from snippet should return to the picker")
	}
	nt, _, _ = mt.plugUpdate(key("esc"))
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugOff {
		t.Fatal("esc from picker should close the overlay")
	}
}

func TestPlugWriteConfirmGate(t *testing.T) {
	mt := pluggedMonitor(t)
	// Select a writable agent deterministically: find codex in the (sorted) list.
	for i, e := range mt.plugAgents {
		if e.agent.Key == "codex" {
			mt.plugSel = i
		}
	}
	nt, _, _ := mt.plugUpdate(key("enter"))
	mt = nt.(*monitorTab)

	// w → confirm gate (nothing written yet), n → back out unchanged.
	nt, cmd, _ := mt.plugUpdate(key("w"))
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugConfirm || cmd != nil {
		t.Fatalf("w must gate behind confirm without writing, mode=%d", mt.plugMode)
	}
	if v := mt.plugView(100, 60); !strings.Contains(v, "write this into") {
		t.Fatalf("confirm view must name the target file:\n%s", v)
	}
	nt, cmd, _ = mt.plugUpdate(key("n"))
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugShow || cmd != nil {
		t.Fatal("n must cancel without a write command")
	}

	// Print-only agent: w explains instead of gating.
	for i, e := range mt.plugAgents {
		if e.agent.Key == "cursor" {
			mt.plugSel = i
		}
	}
	mt.plugMode = monPlugShow
	nt, cmd, _ = mt.plugUpdate(key("w"))
	mt = nt.(*monitorTab)
	if mt.plugMode != monPlugShow || cmd != nil {
		t.Fatal("cursor has no writable config — w must not gate or write")
	}
	if !strings.Contains(mt.plugNote, "no writable config") {
		t.Fatalf("expected the explanation note, got %q", mt.plugNote)
	}
}

// The overlay must respect the height budget (short terminals) and point at
// the clipboard escape hatch.
func TestPlugViewClamped(t *testing.T) {
	mt := pluggedMonitor(t)
	mt.plugUpdate(key("enter"))
	view := mt.plugView(100, 8)
	if lines := strings.Count(view, "\n") + 1; lines > 8 {
		t.Fatalf("view is %d lines, budget 8", lines)
	}
	if !strings.Contains(view, "y copies") {
		t.Fatal("clamped view must mention the copy escape hatch")
	}
}
