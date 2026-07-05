package fit

// Hugging Face reference resolution. Turns "owner/repo[:QUANT]" into the GGUF
// file(s) to read. Pure logic only — the actual HTTP lives in the CLI so this
// stays unit-testable against captured file listings.

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// HFRef is a parsed "owner/repo[:quant]" reference.
type HFRef struct {
	Repo  string
	Quant string
}

var (
	refRe   = regexp.MustCompile(`^[\w.-]+/[\w.-]+(:[\w.-]+)?$`)
	quantRe = regexp.MustCompile(`(IQ[0-9]+_[A-Z]+|Q[0-9]+_[0-9A-Z_]+|BF16|F16|F32)`)
	shardRe = regexp.MustCompile(`-[0-9]+-of-[0-9]+\.gguf$`)
)

// ParseRef recognises an "owner/repo[:quant]" string. It returns ok=false for
// URLs and filesystem paths, which the caller handles directly.
func ParseRef(s string) (HFRef, bool) {
	if strings.Contains(s, "://") || strings.HasPrefix(s, "/") || strings.HasPrefix(s, ".") {
		return HFRef{}, false
	}
	if !refRe.MatchString(s) {
		return HFRef{}, false
	}
	repo, quant, _ := strings.Cut(s, ":")
	return HFRef{Repo: repo, Quant: quant}, true
}

// ResolveURL builds the direct download URL for a file in a repo (main branch).
func ResolveURL(repo, filename string) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename)
}

func isGGUF(name string) bool {
	low := strings.ToLower(name)
	return strings.HasSuffix(low, ".gguf") &&
		!strings.Contains(low, "mmproj") && !strings.Contains(low, "projector")
}

// ListQuants returns the unique quant tags present among a repo's GGUF files.
func ListQuants(files []string) []string {
	seen := map[string]bool{}
	for _, f := range files {
		if !isGGUF(f) {
			continue
		}
		if q := quantRe.FindString(f); q != "" {
			seen[q] = true
		}
	}
	out := make([]string, 0, len(seen))
	for q := range seen {
		out = append(out, q)
	}
	sort.Strings(out)
	return out
}

// SelectGGUF picks the GGUF file(s) for the requested quant. With no quant it
// auto-selects the first available (sorted). It returns all shards for the
// chosen quant in order — read [0] for the header, sum all for total weights.
func SelectGGUF(files []string, quant string) (shards []string, chosen string, err error) {
	quants := ListQuants(files)
	if len(quants) == 0 {
		return nil, "", fmt.Errorf("no GGUF files found in repo")
	}
	if quant == "" {
		quant = quants[0] // mirror launch.sh: auto-pick the first
	}
	want := strings.ToLower(quant)

	var matches []string
	for _, f := range files {
		if isGGUF(f) && strings.Contains(strings.ToLower(f), want) {
			matches = append(matches, f)
		}
	}
	if len(matches) == 0 {
		return nil, "", fmt.Errorf("quant %q not found; available: %s",
			quant, strings.Join(quants, ", "))
	}
	sort.Strings(matches) // zero-padded shard indices sort correctly
	return matches, quant, nil
}

// IsSharded reports whether a filename is one shard of a multi-file model.
func IsSharded(name string) bool { return shardRe.MatchString(name) }
