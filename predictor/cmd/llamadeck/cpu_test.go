package main

import "testing"

func TestHardware_CPUFlagZerosVRAM(t *testing.T) {
	o := opts{cpu: true, ramMB: 32000} // fixed RAM so the test is host-independent
	hw := o.hardware()
	if hw.FreeVRAM != 0 {
		t.Errorf("--cpu hw.FreeVRAM = %d, want 0", hw.FreeVRAM)
	}
	if hw.NumGPUs != 0 {
		t.Errorf("--cpu hw.NumGPUs = %d, want 0", hw.NumGPUs)
	}
	if hw.FreeRAM != 32000<<20 {
		t.Errorf("--cpu hw.FreeRAM = %d, want the --ram-mb override", hw.FreeRAM)
	}
}

func TestParseArgs_CPUFlag(t *testing.T) {
	o := parseArgs([]string{"org/model", "--cpu"})
	if !o.cpu {
		t.Error("--cpu should set opts.cpu")
	}
}
