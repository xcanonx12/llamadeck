package infra

import "testing"

func TestIsUnifiedGPU(t *testing.T) {
	unified := []string{"NVIDIA GB10", "NVIDIA DGX Spark", "Orin (nvgpu)", "NVIDIA Tegra Thor"}
	discrete := []string{"NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3", "NVIDIA GB200"}
	for _, n := range unified {
		if !isUnifiedGPU(n) {
			t.Errorf("isUnifiedGPU(%q) = false, want true", n)
		}
	}
	for _, n := range discrete {
		if isUnifiedGPU(n) {
			t.Errorf("isUnifiedGPU(%q) = true, want false", n)
		}
	}
}

func TestDockerFailReason(t *testing.T) {
	cases := map[string]string{
		"permission denied while trying to connect to the Docker daemon socket": "usermod",
		"Cannot connect to the Docker daemon at unix:///var/run/docker.sock":    "systemctl",
		"something else entirely":                                              "something else",
	}
	for out, want := range cases {
		if got := dockerFailReason(out); !contains(got, want) {
			t.Errorf("dockerFailReason(%q) = %q, want it to mention %q", out, got, want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
