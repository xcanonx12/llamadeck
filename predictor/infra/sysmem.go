package infra

// Cross-platform host memory + GPU free-memory probes, with graceful fallbacks
// so the tool degrades cleanly instead of mispredicting on machines without an
// NVIDIA GPU, without Docker, or without /proc (macOS).

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// GPUSummary returns the GPU count and total free VRAM across all of them — the
// pooled budget llama.cpp distributes a model across by default.
func GPUSummary() (count int, totalFreeVRAM int64) {
	for _, g := range GPUs() {
		count++
		totalFreeVRAM += g.MemFree
	}
	return count, totalFreeVRAM
}

// FreeRAM returns a sensible RAM budget in bytes:
//   - Linux: MemAvailable from /proc/meminfo (what's actually allocatable).
//   - macOS/BSD: total physical RAM via sysctl (the ceiling) when /proc is absent.
//   - 0 only if everything fails (callers should treat 0 as "unknown").
func FreeRAM() int64 {
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		if v := parseMemAvailable(string(b)); v > 0 {
			return v
		}
	}
	if out, err := exec.Command("sysctl", "-n", "hw.memsize").Output(); err == nil {
		if n, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64); err == nil {
			return n
		}
	}
	return 0
}

// parseMemAvailable extracts MemAvailable (bytes) from /proc/meminfo text.
func parseMemAvailable(meminfo string) int64 {
	for _, line := range strings.Split(meminfo, "\n") {
		if strings.HasPrefix(line, "MemAvailable:") {
			f := strings.Fields(line) // "MemAvailable:  12345 kB"
			if len(f) >= 2 {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					return kb << 10
				}
			}
		}
	}
	return 0
}
