package infra

import "testing"

func TestParseMemAvailable(t *testing.T) {
	sample := `MemTotal:       65827284 kB
MemFree:         1234567 kB
MemAvailable:   48000000 kB
Buffers:          123456 kB`
	got := parseMemAvailable(sample)
	want := int64(48000000) << 10
	if got != want {
		t.Errorf("parseMemAvailable = %d, want %d", got, want)
	}
}

func TestParseMemAvailableMissing(t *testing.T) {
	if got := parseMemAvailable("MemTotal: 100 kB\n"); got != 0 {
		t.Errorf("expected 0 when MemAvailable absent, got %d", got)
	}
}
