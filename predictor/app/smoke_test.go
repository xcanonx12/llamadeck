package app

import (
	"strings"
	"testing"
)

// TestRenderSmoke renders each tab to a static frame: a cheap regression net that
// the app composes without panicking, and (under -v) a visual of every screen.
func TestRenderSmoke(t *testing.T) {
	tabs := []string{"Models", "Fit", "Monitor", "Config"}
	for i, name := range tabs {
		out := RenderOnce(i, 96, 28)
		if !strings.Contains(out, "llamadeck") {
			t.Errorf("%s frame missing header chrome", name)
		}
		t.Logf("\n=== %s ===\n%s", name, out)
	}
}
