package hub

// Model loading: turn a source string (owner/repo[:quant], a .gguf URL, or a
// local path) into a parsed *fit.Model by reading only the GGUF header. Shared by
// the CLI and the TUI so the resolution logic lives in exactly one place.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"llamadeck/fit"
)

const headerWindow = 32 << 20 // bytes range-fetched from a remote GGUF header

// Load resolves a source to a parsed model. It returns the model, a display
// label (repo or URL/path), and the chosen quant ("" for non-HF sources).
func Load(src string) (m *fit.Model, label, quant string, err error) {
	if ref, ok := fit.ParseRef(src); ok {
		return loadHF(ref)
	}
	if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
		total, err := HeadSize(src)
		if err != nil {
			return nil, "", "", err
		}
		mm, err := ParseHeader(src, total)
		return mm, src, "", err
	}
	f, err := os.Open(src)
	if err != nil {
		return nil, "", "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, "", "", err
	}
	mm, err := fit.ParseGGUF(f, info.Size())
	return mm, src, "", err
}

func loadHF(ref fit.HFRef) (*fit.Model, string, string, error) {
	files, err := Siblings(ref.Repo)
	if err != nil {
		return nil, "", "", err
	}
	shards, quant, err := fit.SelectGGUF(files, ref.Quant)
	if err != nil {
		return nil, "", "", err
	}
	var total int64
	for _, s := range shards {
		sz, err := HeadSize(fit.ResolveURL(ref.Repo, s))
		if err != nil {
			return nil, "", "", fmt.Errorf("sizing %s: %w", s, err)
		}
		total += sz
	}
	m, err := ParseHeader(fit.ResolveURL(ref.Repo, shards[0]), total)
	return m, ref.Repo, quant, err
}

// Quants returns the available quant tags for an HF ref source; nil for a
// direct URL or local path (where there's a single file, not a quant family).
func Quants(src string) []string {
	ref, ok := fit.ParseRef(src)
	if !ok {
		return nil
	}
	files, err := Siblings(ref.Repo)
	if err != nil {
		return nil
	}
	return fit.ListQuants(files)
}

// Siblings lists a repo's files via the Hugging Face models API.
func Siblings(repo string) ([]string, error) {
	resp, err := hfGet("https://huggingface.co/api/models/" + repo)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, httpErr(resp, "HF API "+repo)
	}
	var data struct {
		Siblings []struct {
			Rfilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	files := make([]string, len(data.Siblings))
	for i, s := range data.Siblings {
		files[i] = s.Rfilename
	}
	return files, nil
}

// HeadSize returns a URL's Content-Length via a HEAD request.
func HeadSize(url string) (int64, error) {
	resp, err := hfDo("HEAD", url)
	if err != nil {
		return 0, err
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return 0, httpErr(resp, "HF file "+url)
	}
	if resp.ContentLength <= 0 {
		return 0, fmt.Errorf("could not determine file size (Content-Length missing)")
	}
	return resp.ContentLength, nil
}

// ParseHeader range-fetches just the header window of a remote GGUF and parses
// it. totalBytes is the full weight size (summed across shards by the caller).
func ParseHeader(url string, totalBytes int64) (*fit.Model, error) {
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", headerWindow-1))
	setAuth(req)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status fetching header: %s", resp.Status)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return fit.ParseGGUF(bytes.NewReader(data), totalBytes)
}
