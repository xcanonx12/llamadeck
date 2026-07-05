package plug

// The safety contract of --write: foreign keys survive byte-for-byte at the
// semantic level, our entry is overwritten on re-run, invalid input aborts
// BEFORE any modification, and every touch of an existing file leaves a backup.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func readJSON(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, b)
	}
	return v
}

func TestMergeJSONCreatesAndOverwritesOnlyOurs(t *testing.T) {
	f := filepath.Join(t.TempDir(), "sub", "opencode.json")

	// Fresh file (parents created), no backup expected.
	backup, err := mergeJSONFile(f, []string{"provider", "llamadeck"}, map[string]any{"v": 1})
	if err != nil {
		t.Fatal(err)
	}
	if backup != "" {
		t.Fatalf("no backup expected for a new file, got %s", backup)
	}

	// Simulate the user's own edits around our entry.
	v := readJSON(t, f)
	v["theme"] = "dark"
	v["provider"].(map[string]any)["anthropic"] = map[string]any{"apiKey": "sk-user"}
	b, _ := json.Marshal(v)
	os.WriteFile(f, b, 0o644)

	// Re-run: our entry replaced, everything else intact, backup written.
	backup, err = mergeJSONFile(f, []string{"provider", "llamadeck"}, map[string]any{"v": 2})
	if err != nil {
		t.Fatal(err)
	}
	if backup == "" {
		t.Fatal("existing file must be backed up")
	}
	if _, err := os.Stat(backup); err != nil {
		t.Fatalf("backup missing: %v", err)
	}
	got := readJSON(t, f)
	if got["theme"] != "dark" {
		t.Fatal("foreign top-level key lost")
	}
	prov := got["provider"].(map[string]any)
	if prov["anthropic"].(map[string]any)["apiKey"] != "sk-user" {
		t.Fatal("foreign provider lost")
	}
	if prov["llamadeck"].(map[string]any)["v"] != float64(2) {
		t.Fatal("our entry must be overwritten on re-run")
	}
}

func TestMergeJSONRefusesBrokenInput(t *testing.T) {
	f := filepath.Join(t.TempDir(), "conf.json")
	os.WriteFile(f, []byte("{not json"), 0o644)
	if _, err := mergeJSONFile(f, []string{"provider", "llamadeck"}, 1); err == nil {
		t.Fatal("invalid JSON must be refused")
	}
	raw, _ := os.ReadFile(f)
	if string(raw) != "{not json" {
		t.Fatal("refusal must not modify the file")
	}

	// A scalar in the path must abort, not be clobbered.
	os.WriteFile(f, []byte(`{"provider": "oops"}`), 0o644)
	if _, err := mergeJSONFile(f, []string{"provider", "llamadeck"}, 1); err == nil {
		t.Fatal("non-object intermediate must be refused")
	}

	jc := filepath.Join(t.TempDir(), "conf.jsonc")
	os.WriteFile(jc, []byte("{}"), 0o644)
	if _, err := mergeJSONFile(jc, []string{"a"}, 1); err == nil {
		t.Fatal("jsonc must be refused (comments would be lost)")
	}
}

func TestMarkerBlockAppendReplaceAndRefuse(t *testing.T) {
	f := filepath.Join(t.TempDir(), "config.toml")

	// Fresh file.
	if _, err := writeMarkerBlock(f, "a = 1", false); err != nil {
		t.Fatal(err)
	}
	// Foreign content + re-run: block replaced, foreign bytes untouched.
	raw, _ := os.ReadFile(f)
	os.WriteFile(f, append([]byte("# user comment\nmodel = \"gpt-5\"\n\n"), raw...), 0o644)
	if _, err := writeMarkerBlock(f, "a = 2", false); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(f)
	s := string(got)
	if !strings.Contains(s, "# user comment\nmodel = \"gpt-5\"") {
		t.Fatalf("foreign content damaged:\n%s", s)
	}
	if !strings.Contains(s, "a = 2") || strings.Contains(s, "a = 1") {
		t.Fatalf("block not replaced:\n%s", s)
	}
	if strings.Count(s, markerBegin) != 1 {
		t.Fatalf("exactly one managed block expected:\n%s", s)
	}

	// mustCreate (Hermes): foreign file without our markers is refused.
	g := filepath.Join(t.TempDir(), "config.yaml")
	os.WriteFile(g, []byte("model:\n  default: other\n"), 0o644)
	if _, err := writeMarkerBlock(g, "model:\n  default: ours", true); err == nil {
		t.Fatal("foreign YAML must be refused with mustCreate")
	}
	// ...but a file WE created (markers present) is replaceable.
	h := filepath.Join(t.TempDir(), "config.yaml")
	if _, err := writeMarkerBlock(h, "model:\n  default: v1", true); err != nil {
		t.Fatal(err)
	}
	if _, err := writeMarkerBlock(h, "model:\n  default: v2", true); err != nil {
		t.Fatalf("our own block must be replaceable: %v", err)
	}
	got2, _ := os.ReadFile(h)
	if !strings.Contains(string(got2), "default: v2") || strings.Contains(string(got2), "default: v1") {
		t.Fatalf("hermes block not replaced:\n%s", got2)
	}
}
