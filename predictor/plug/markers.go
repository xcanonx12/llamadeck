package plug

// Marker-block editing for TOML/YAML configs we refuse to parse (zero deps,
// zero risk to foreign content): our contribution lives between two comment
// markers; re-runs replace the block, everything else is untouched bytes.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	markerBegin = "# llamadeck:begin (managed — re-running `llamadeck plug` replaces this block)"
	markerEnd   = "# llamadeck:end"
)

// writeMarkerBlock appends block (or replaces a previous one) in fname,
// creating the file if absent. mustCreate restricts writing to fresh files or
// files that already carry our markers (Hermes: a foreign config is never
// edited). Returns the backup path ("" when the file didn't exist).
func writeMarkerBlock(fname, block string, mustCreate bool) (backup string, err error) {
	wrapped := markerBegin + "\n" + strings.TrimRight(block, "\n") + "\n" + markerEnd + "\n"

	raw, err := os.ReadFile(fname)
	if os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(fname), 0o755); err != nil {
			return "", err
		}
		return "", os.WriteFile(fname, []byte(wrapped), 0o644)
	}
	if err != nil {
		return "", err
	}

	content := string(raw)
	begin := strings.Index(content, markerBegin)
	end := strings.Index(content, markerEnd)
	switch {
	case begin >= 0 && end > begin:
		tail := end + len(markerEnd)
		if tail < len(content) && content[tail] == '\n' {
			tail++
		}
		content = content[:begin] + wrapped + content[tail:]
	case mustCreate:
		return "", fmt.Errorf("%s already exists without a llamadeck block — paste the snippet manually (editing foreign YAML risks breaking it)", fname)
	default:
		if !strings.HasSuffix(content, "\n") && content != "" {
			content += "\n"
		}
		content += "\n" + wrapped
	}

	backup = fname + ".bak-" + time.Now().Format("20060102-150405.000")
	if err := os.WriteFile(backup, raw, 0o644); err != nil {
		return "", fmt.Errorf("backup: %w", err)
	}
	tmp := fname + ".llamadeck-tmp"
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		return backup, err
	}
	return backup, os.Rename(tmp, fname)
}
