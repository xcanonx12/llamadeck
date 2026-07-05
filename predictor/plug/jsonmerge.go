package plug

// Safe JSON config editing: parse the whole file, set exactly ONE namespaced
// path (ours), rewrite. Anything we don't understand aborts BEFORE touching the
// file — the user gets the snippet to paste manually instead of a broken config.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// mergeJSONFile sets path (e.g. ["provider","llamadeck"]) to value in the JSON
// file at fname, creating the file (and parents) if absent. Existing unrelated
// keys are preserved; an existing entry at the same path is overwritten (ours
// to overwrite). A timestamped backup is written first. Returns the backup
// path ("" when the file didn't exist).
func mergeJSONFile(fname string, path []string, value any) (backup string, err error) {
	root := map[string]any{}
	raw, err := os.ReadFile(fname)
	switch {
	case err == nil:
		if strings.HasSuffix(fname, ".jsonc") {
			return "", fmt.Errorf("%s is JSONC (comments) — paste the snippet manually to preserve them", fname)
		}
		if len(strings.TrimSpace(string(raw))) > 0 {
			if err := json.Unmarshal(raw, &root); err != nil {
				return "", fmt.Errorf("%s is not valid JSON (%v) — fix it or paste the snippet manually", fname, err)
			}
		}
		backup = fname + ".bak-" + time.Now().Format("20060102-150405.000")
		if err := os.WriteFile(backup, raw, 0o644); err != nil {
			return "", fmt.Errorf("backup: %w", err)
		}
	case os.IsNotExist(err):
		if err := os.MkdirAll(filepath.Dir(fname), 0o755); err != nil {
			return "", err
		}
	default:
		return "", err
	}

	// Walk/create intermediate objects, then set the leaf.
	node := root
	for _, key := range path[:len(path)-1] {
		child, ok := node[key].(map[string]any)
		if !ok {
			if _, exists := node[key]; exists {
				return backup, fmt.Errorf("%s: %q is not an object — paste the snippet manually", fname, key)
			}
			child = map[string]any{}
			node[key] = child
		}
		node = child
	}
	node[path[len(path)-1]] = value

	out, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return backup, err
	}
	// Atomic-ish: write a temp file next to the target, then rename.
	tmp := fname + ".llamadeck-tmp"
	if err := os.WriteFile(tmp, append(out, '\n'), 0o644); err != nil {
		return backup, err
	}
	return backup, os.Rename(tmp, fname)
}
