package hub

// Quant scanning: size + fit verdict for every quant in a repo. Shared by the
// CLI `recommend` command and the TUI's Fit-tab quant selector so the analysis
// lives in one place. The network half (file list + per-quant sizes + header)
// is cached per repo for the process lifetime, so re-opening the picker — or
// re-scanning at a different context — is instant instead of re-fetching.

import (
	"fmt"
	"sort"
	"sync"

	"llamadeck/fit"
)

// QuantRow is one quant's fit analysis for a repo.
type QuantRow struct {
	Quant  string
	Size   int64
	MaxGPU int      // largest context that still runs fully on GPU (0 = none)
	Mode   fit.Mode // verdict at the requested context
}

// repoScan is the cached, network-derived data for a repo (structure + sizes).
type repoScan struct {
	model  *fit.Model
	quants []string
	sizes  []int64 // parallel to quants
}

var (
	scanMu    sync.Mutex
	scanCache = map[string]*repoScan{}
)

// fetchScan returns the repo's structure + per-quant sizes, fetching once and
// caching for the session. ponytail: no TTL — model files are immutable enough
// for a session; restart to refresh.
func fetchScan(repo string) (*repoScan, error) {
	scanMu.Lock()
	cached := scanCache[repo]
	scanMu.Unlock()
	if cached != nil {
		return cached, nil
	}

	files, err := Siblings(repo)
	if err != nil {
		return nil, err
	}
	quants := fit.ListQuants(files)
	if len(quants) == 0 {
		return nil, fmt.Errorf("no GGUF quants found in %s", repo)
	}
	headShards, _, err := fit.SelectGGUF(files, quants[0])
	if err != nil {
		return nil, err
	}
	m, err := ParseHeader(fit.ResolveURL(repo, headShards[0]), 0)
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	sizes := make([]int64, len(quants))
	for i, q := range quants {
		shards, _, err := fit.SelectGGUF(files, q)
		if err != nil {
			continue
		}
		for _, s := range shards {
			if sz, err := HeadSize(fit.ResolveURL(repo, s)); err == nil {
				sizes[i] += sz
			}
		}
	}
	sc := &repoScan{model: m, quants: quants, sizes: sizes}
	scanMu.Lock()
	scanCache[repo] = sc
	scanMu.Unlock()
	return sc, nil
}

// QuantReport returns per-quant fit rows (sorted ascending by size) plus the
// index of the recommended pick: the largest quant that runs fully on GPU at
// cfg.Ctx, else the largest that avoids OOM, else -1. The verdict/max-ctx math
// is recomputed cheaply on every call; only the sizes are cached.
func QuantReport(repo string, hw fit.Hardware, cfg fit.Config) (rows []QuantRow, pick int, err error) {
	sc, err := fetchScan(repo)
	if err != nil {
		return nil, -1, err
	}
	m := *sc.model // copy: we set FileBytes per quant; don't mutate the cached header
	for i, q := range sc.quants {
		if sc.sizes[i] == 0 {
			continue
		}
		m.FileBytes = sc.sizes[i]
		// Rescale the exact tensor table to this quant's size — Predict uses
		// m.Tensors.PerLayer when present and ignores FileBytes, so without this
		// every quant is judged with the FIRST quant's weights (same verdict rows).
		if sc.model.Tensors != nil {
			m.Tensors = sc.model.Tensors.ScaledTo(sc.sizes[i])
		}
		r, err := fit.Predict(&m, hw, cfg)
		if err != nil {
			continue
		}
		rows = append(rows, QuantRow{q, sc.sizes[i], fit.MaxCtxForMode(&m, hw, cfg, fit.ModeGPU), r.Mode})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Size < rows[j].Size })
	return rows, pickQuant(rows), nil
}

// pickQuant returns the recommended row index: largest fully-GPU quant, else
// largest non-OOM quant, else -1.
func pickQuant(rows []QuantRow) int {
	best, fallback := -1, -1
	for i, r := range rows {
		if r.Mode == fit.ModeGPU {
			best = i
		}
		if r.Mode != fit.ModeOOM {
			fallback = i
		}
	}
	if best >= 0 {
		return best
	}
	return fallback
}
