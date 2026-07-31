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
// pooled budget llama.cpp distributes a model across by default. unified is true
// when that memory IS the host RAM (DGX Spark / Jetson), so callers must not
// budget VRAM and RAM as separate pools.
func GPUSummary() (count int, totalFreeVRAM int64, unified bool) {
	for _, g := range GPUs() {
		count++
		totalFreeVRAM += g.MemFree
		unified = unified || g.Unified
	}
	return count, totalFreeVRAM, unified
}

// TotalRAM returns total physical RAM in bytes (0 if unknown). Used as the
// memory ceiling on unified-memory machines, where it is also the VRAM ceiling.
func TotalRAM() int64 {
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		if v := parseMemField(string(b), "MemTotal:"); v > 0 {
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
func parseMemAvailable(meminfo string) int64 { return parseMemField(meminfo, "MemAvailable:") }

// parseMemField extracts one kB-valued /proc/meminfo field as bytes.
func parseMemField(meminfo, key string) int64 {
	for _, line := range strings.Split(meminfo, "\n") {
		if strings.HasPrefix(line, key) {
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
