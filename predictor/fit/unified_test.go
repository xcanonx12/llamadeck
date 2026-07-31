package fit

import "testing"

// On unified-memory machines (DGX Spark GB10) FreeVRAM and FreeRAM are the same
// physical bytes. A hybrid split whose GPU + host halves each fit the pool but
// together exceed it must be OOM, not "fits" — the bug the Unified flag closes.
func TestUnifiedMemoryIsNotCountedTwice(t *testing.T) {
	m := qwen35ish()
	c := DefaultConfig()
	c.Ctx = 8192
	c.NoMmap = true // pin the host weights so RAM demand is a hard requirement
	c.NGL = m.NLayers / 2

	pool := int64(6) * gib // small enough that half-on-GPU + half-in-RAM overflows

	split, err := Predict(m, Hardware{FreeVRAM: pool, FreeRAM: pool, NumGPUs: 1}, c)
	if err != nil {
		t.Fatal(err)
	}
	if split.VRAMUsed+split.RAMUsed <= pool {
		t.Fatalf("test pool too large to exercise the overlap: vram=%d ram=%d pool=%d",
			split.VRAMUsed, split.RAMUsed, pool)
	}
	if split.Mode == ModeOOM {
		t.Skip("split already OOM on one budget alone; nothing for Unified to add")
	}

	uni, err := Predict(m, Hardware{FreeVRAM: pool, FreeRAM: pool, NumGPUs: 1, Unified: true}, c)
	if err != nil {
		t.Fatal(err)
	}
	if uni.Mode != ModeOOM {
		t.Errorf("unified: vram %s + ram %s > pool %s but mode = %v, want OOM",
			HumanBytes(uni.VRAMUsed), HumanBytes(uni.RAMUsed), HumanBytes(pool), uni.Mode)
	}
}

// A model that comfortably fits the whole pool must still fit when unified.
func TestUnifiedMemoryFitsWhenItShould(t *testing.T) {
	c := DefaultConfig()
	c.Ctx = 8192
	pool := int64(120) * gib // a Spark-sized pool
	r, err := Predict(qwen35ish(), Hardware{FreeVRAM: pool, FreeRAM: pool, NumGPUs: 1, Unified: true}, c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Mode == ModeOOM {
		t.Errorf("120 GiB unified pool: mode = OOM, want a fit (vram %s + ram %s)",
			HumanBytes(r.VRAMUsed), HumanBytes(r.RAMUsed))
	}
}
