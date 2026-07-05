package hub

// Regression for the quant-picker bug: QuantReport predicted every quant with
// the FIRST quant's tensor table (m.Tensors shared by pointer), so all rows got
// the same Mode. The scan cache is seeded directly — no network.

import (
	"testing"

	"llamadeck/fit"
)

func TestQuantReportModesDifferPerQuant(t *testing.T) {
	const n = 16
	const mib = int64(1 << 20)
	const gib = int64(1 << 30)
	per := make([]int64, n)
	for i := range per {
		per[i] = 40 * mib
	}
	m := &fit.Model{
		Arch: "llama", NLayers: n, NHeads: 32, NKVHeads: 8,
		HeadDim: 64, EmbedLength: 2048, VocabSize: 128256,
		FileBytes: 763 * mib,
		Tensors: &fit.TensorStats{
			Total: 763 * mib, Embedding: 60 * mib, Output: 60 * mib, PerLayer: per,
		},
	}
	repo := "test/quant-report-fake"
	scanMu.Lock()
	scanCache[repo] = &repoScan{
		model:  m,
		quants: []string{"Q4_K_M", "Q8_0"},
		sizes:  []int64{763 * mib, 40 * gib}, // small fits; huge cannot
	}
	scanMu.Unlock()
	defer func() {
		scanMu.Lock()
		delete(scanCache, repo)
		scanMu.Unlock()
	}()

	hw := fit.Hardware{FreeVRAM: 8 * gib, FreeRAM: 16 * gib}
	rows, _, err := QuantReport(repo, hw, fit.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(rows))
	}
	if rows[0].Mode != fit.ModeGPU {
		t.Fatalf("small quant should be 100%% GPU, got %s", rows[0].Mode)
	}
	if rows[1].Mode == rows[0].Mode {
		t.Fatalf("40 GiB quant must not share the small quant's verdict (%s)", rows[1].Mode)
	}
	// The cached header must not be mutated by the scan.
	if m.Tensors.Total != 763*mib || m.FileBytes != 763*mib {
		t.Fatal("QuantReport mutated the cached model")
	}
}
