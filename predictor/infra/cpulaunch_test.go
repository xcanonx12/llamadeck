package infra

import (
	"testing"
)

// hasFlag reports whether args contains the exact token flag.
func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

// flagVal returns the token following flag, or "" if absent.
func flagVal(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func TestLaunchArgs_CPU_NoGPUsFlag(t *testing.T) {
	s := LaunchSpec{HF: "org/model:Q4_K_M", Port: 8080, Ctx: 4096, NGL: 0,
		CacheDir: "/models", CPU: true}
	args := launchArgs("llama-x", s)

	if hasFlag(args, "--gpus") {
		t.Errorf("CPU launch must not pass --gpus; got %v", args)
	}
	if got := flagVal(args, "-ngl"); got != "0" {
		t.Errorf("-ngl = %q, want 0", got)
	}
	// Sanity: the model/port args are still present.
	if got := flagVal(args, "-hf"); got != "org/model:Q4_K_M" {
		t.Errorf("-hf = %q", got)
	}
}

func TestLaunchArgs_GPU_KeepsGpusFlag(t *testing.T) {
	s := LaunchSpec{HF: "org/model", Port: 8080, Ctx: 4096, NGL: 999, CacheDir: "/models"}
	args := launchArgs("llama-x", s)
	if !hasFlag(args, "--gpus") {
		t.Errorf("GPU launch must pass --gpus; got %v", args)
	}
	if got := flagVal(args, "--gpus"); got != "all" {
		t.Errorf("--gpus = %q, want all", got)
	}
}
