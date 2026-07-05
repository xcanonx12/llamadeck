package infra

import (
	"strings"
	"testing"
)

// joinArgs makes the slice greppable as one string for presence checks.
func joinArgs(a []string) string { return strings.Join(a, "\x00") }

func has(args []string, flag, val string) bool {
	for i, a := range args {
		if a == flag && i+1 < len(args) && args[i+1] == val {
			return true
		}
	}
	return false
}

func TestLaunchArgsDefaults(t *testing.T) {
	args := launchArgs("llama-x", LaunchSpec{HF: "owner/repo:Q4_K_M", Port: 8080, Ctx: 8192, NGL: 999})
	// Required flags always present.
	for _, want := range [][2]string{
		{"--host", "0.0.0.0"}, {"--port", "8080"}, {"--ctx-size", "8192"}, {"-ngl", "999"}, {"-hf", "owner/repo:Q4_K_M"},
	} {
		if !has(args, want[0], want[1]) {
			t.Errorf("missing %s %s", want[0], want[1])
		}
	}
	// Zero-valued optionals must be omitted entirely.
	for _, flag := range []string{"--threads", "--n-cpu-moe", "--ubatch-size", "--parallel", "--cache-type-k", "--cache-type-v", "--jinja", "--no-mmap", "--mlock", "--flash-attn", "--ulimit"} {
		if strings.Contains(joinArgs(args), flag) {
			t.Errorf("unexpected %s for a default spec", flag)
		}
	}
}

func TestLaunchArgsTuning(t *testing.T) {
	s := LaunchSpec{
		HF: "o/r:Q5_K_M", Port: 9001, Ctx: 4096, NGL: 20, Host: "127.0.0.1",
		FlashAttn: true, Threads: 8, NCPUMoE: 4, UBatch: 256, Parallel: 8,
		CacheTypeK: "q8_0", CacheTypeV: "q8_0", Jinja: true, NoMmap: true, Mlock: true,
	}
	args := launchArgs("llama-y", s)
	checks := [][2]string{
		{"--host", "127.0.0.1"}, {"--threads", "8"}, {"--n-cpu-moe", "4"},
		{"--ubatch-size", "256"}, {"--parallel", "8"}, {"--cache-type-k", "q8_0"}, {"--cache-type-v", "q8_0"},
		{"--flash-attn", "on"},
	}
	for _, c := range checks {
		if !has(args, c[0], c[1]) {
			t.Errorf("missing %s %s", c[0], c[1])
		}
	}
	// Boolean-only flags present without a value.
	j := joinArgs(args)
	for _, flag := range []string{"--jinja", "--no-mmap", "--mlock"} {
		if !strings.Contains(j, flag) {
			t.Errorf("missing %s", flag)
		}
	}
	// Port maps host:container 1:1.
	if !has(args, "-p", "9001:9001") {
		t.Error("expected -p 9001:9001")
	}
	// Mlock needs the docker-side ulimit lifted or it silently no-ops; the
	// --ulimit is a docker option so it must come BEFORE the image tag.
	if !has(args, "--ulimit", "memlock=-1") {
		t.Error("Mlock spec must add --ulimit memlock=-1")
	}
	ulimit, image := -1, -1
	for i, a := range args {
		if a == "--ulimit" && ulimit < 0 {
			ulimit = i
		}
		if a == ImageTag && image < 0 {
			image = i
		}
	}
	if ulimit > image {
		t.Errorf("--ulimit at %d must precede the image tag at %d", ulimit, image)
	}
}

func TestLaunchArgsHealthcheckUsesActualPort(t *testing.T) {
	args := launchArgs("x", LaunchSpec{HF: "o/r", Port: 8090})
	if !has(args, "--health-cmd", "curl -fsS http://localhost:8090/health || exit 1") {
		t.Errorf("healthcheck must target the launched port 8090, got:\n%s", strings.Join(args, " "))
	}
	// The image's hardcoded :8080 check would falsely report unhealthy.
	if strings.Contains(joinArgs(args), ":8080/health") {
		t.Error("must not health-check the wrong port")
	}
}

func TestLaunchArgsGPUPinning(t *testing.T) {
	all := launchArgs("x", LaunchSpec{HF: "o/r", Port: 8080})
	if !has(all, "--gpus", "all") {
		t.Error("default should be --gpus all")
	}
	pinned := launchArgs("x", LaunchSpec{HF: "o/r", Port: 8080, Device: "1"})
	if !has(pinned, "--gpus", "device=1") {
		t.Error("Device=1 should give --gpus device=1")
	}
}
