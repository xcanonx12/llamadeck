package fit

// Corpus replay: every saved real-launch log under testdata/audit is re-judged
// by the current engine, bucket by bucket. A failure names the corpus entry AND
// the bucket — "exactly where the math starts to lie". Entries whose GGUF isn't
// in the local HF cache are skipped (the corpus carries logs, not weights).
//
// Regenerate/extend the corpus by saving `docker logs` + a manifest next to it;
// `llamadeck audit` runs the same checks against a live container.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type auditManifest struct {
	GGUF        string  `json:"gguf"` // path under ~/.cache/huggingface/hub
	Ctx         int     `json:"ctx"`
	UBatch      int     `json:"ubatch"`
	KV          string  `json:"kv"`
	NGL         int     `json:"ngl"`
	Parallel    int     `json:"parallel"`
	GPUsFreeMiB []int64 `json:"gpus_free_mib"`
	Outcome     string  `json:"outcome"` // ran | crashed
}

func TestAuditCorpus(t *testing.T) {
	manifests, _ := filepath.Glob("testdata/audit/*.json")
	if len(manifests) == 0 {
		t.Skip("no audit corpus present")
	}
	home, _ := os.UserHomeDir()
	for _, mf := range manifests {
		name := strings.TrimSuffix(filepath.Base(mf), ".json")
		t.Run(name, func(t *testing.T) {
			b, err := os.ReadFile(mf)
			if err != nil {
				t.Fatal(err)
			}
			var man auditManifest
			if err := json.Unmarshal(b, &man); err != nil {
				t.Fatal(err)
			}
			logB, err := os.ReadFile(filepath.Join("testdata/audit", name+".log"))
			if err != nil {
				t.Fatal(err)
			}
			gguf := filepath.Join(home, ".cache/huggingface/hub", man.GGUF)
			f, err := os.Open(gguf)
			if err != nil {
				t.Skipf("model not cached locally: %v", err)
			}
			defer f.Close()
			st, _ := f.Stat()
			m, err := ParseGGUF(f, st.Size())
			if err != nil {
				t.Fatal(err)
			}
			mm, err := ParseServerLog(string(logB))
			if err != nil {
				t.Fatalf("corpus log unparseable: %v", err)
			}

			free := make([]int64, len(man.GPUsFreeMiB))
			var total int64
			for i, mib := range man.GPUsFreeMiB {
				free[i] = mib << 20
				total += free[i]
			}
			hw := Hardware{FreeVRAM: total, FreeRAM: 26 * gib, NumGPUs: len(free)}
			if len(free) > 1 {
				hw.GPUsFree = free
			}
			c := DefaultConfig()
			c.Ctx = man.Ctx
			c.UBatch = man.UBatch
			c.KVType = man.KV
			c.NSeqs = man.Parallel
			if man.NGL > 0 && man.NGL < 999 {
				c.NGL = man.NGL
			}

			for _, ch := range AuditLaunch(m, hw, c, mm, man.Outcome) {
				if !ch.OK {
					t.Errorf("%s: pred %s vs real %s — %s",
						ch.Name, HumanBytes(ch.Pred), HumanBytes(ch.Real), ch.Detail)
				}
			}
		})
	}
}
