package app

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

// The Docker fix-up hint is longer than a narrow terminal, and the layout
// budgets exactly one row for the banner — so it must truncate, not wrap.
func TestBannerTruncatesToWidth(t *testing.T) {
	const width = 50
	m := New()
	m.sh.dockerOK = false
	m.sh.dockerWhy = "daemon socket denied — run: sudo usermod -aG docker $USER, then re-login (or newgrp docker)"
	m.w, m.h = width, 24

	var found bool
	for _, line := range strings.Split(m.View(), "\n") {
		if !strings.Contains(line, "⚠ Docker") {
			continue
		}
		found = true
		if w := lipgloss.Width(line); w > width {
			t.Errorf("banner is %d cols wide, terminal is %d: %q", w, width, line)
		}
	}
	if !found {
		t.Fatal("no Docker banner rendered")
	}
}
