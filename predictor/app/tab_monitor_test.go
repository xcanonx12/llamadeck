package app

import (
	"strings"
	"testing"

	"llamadeck/infra"
)

func TestParsePercent(t *testing.T) {
	cases := map[string]struct {
		want float64
		ok   bool
	}{
		"downloading model.gguf: 42.5%": {0.425, true},
		"###########          45%":      {0.45, true},
		"100%":                          {1.0, true},
		"loading tensors":               {0, false},
		"999%":                          {0, false},   // out of range → ignored
		"10%\r30%\r63%":                 {0.63, true}, // \r frames → take the last
	}
	for in, exp := range cases {
		got, ok := parsePercent(in)
		if ok != exp.ok || (ok && got != exp.want) {
			t.Errorf("parsePercent(%q) = %v,%v want %v,%v", in, got, ok, exp.want, exp.ok)
		}
	}
}

func TestHealthNoteExplains(t *testing.T) {
	cases := []struct {
		c    infra.Container
		want string // substring the note must contain
	}{
		{infra.Container{State: "running", Health: "healthy", Port: "8090"}, "ready"},
		{infra.Container{State: "running", Health: "starting"}, "loading"},
		{infra.Container{State: "running", Health: "unhealthy"}, "not answering"},
		{infra.Container{State: "running", Health: "n/a", Port: "8090"}, "no healthcheck"},
		{infra.Container{State: "exited"}, "stopped"},
	}
	for _, tc := range cases {
		if note := healthNote(tc.c); !strings.Contains(note, tc.want) {
			t.Errorf("healthNote(%+v) = %q, want substring %q", tc.c, note, tc.want)
		}
	}
}

func TestSpinningOnlyWhileLoading(t *testing.T) {
	if spinning(infra.Container{State: "running", Health: "healthy"}) {
		t.Error("a healthy server should not spin")
	}
	if !spinning(infra.Container{State: "running", Health: "starting"}) {
		t.Error("a starting server should spin")
	}
	if spinning(infra.Container{State: "exited", Health: "unhealthy"}) {
		t.Error("an exited container should not spin")
	}
}
