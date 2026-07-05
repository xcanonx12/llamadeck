package hub

// Capability detection: what a model can do, inferred from cheap Hugging Face
// metadata (pipeline tag, repo tags, and whether a multimodal projector file
// ships in the repo) — no weight download. We only report capabilities we can
// positively detect; absence means "unknown", never "unsupported", so we never
// assert something false about a model.

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"

	"llamadeck/fit"
)

var (
	capsMu    sync.Mutex
	capsCache = map[string][]string{} // repo → caps, cached on success for the session
)

// Capabilities returns detected capability labels for an HF ref source
// ("vision", "audio", "reasoning", "tools", "embeddings"); nil for non-HF
// sources or on any error. Successful lookups are cached per repo so re-opening
// a model's details doesn't re-hit the Hub.
func Capabilities(src string) []string {
	ref, ok := fit.ParseRef(src)
	if !ok {
		return nil
	}
	capsMu.Lock()
	if c, ok := capsCache[ref.Repo]; ok {
		capsMu.Unlock()
		return c
	}
	capsMu.Unlock()

	resp, err := hfGet("https://huggingface.co/api/models/" + ref.Repo)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var data struct {
		PipelineTag string   `json:"pipeline_tag"`
		Tags        []string `json:"tags"`
		Siblings    []struct {
			Rfilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if json.NewDecoder(resp.Body).Decode(&data) != nil {
		return nil
	}
	files := make([]string, len(data.Siblings))
	for i, s := range data.Siblings {
		files[i] = s.Rfilename
	}
	caps := deriveCaps(data.PipelineTag, data.Tags, files, ref.Repo)
	capsMu.Lock()
	capsCache[ref.Repo] = caps
	capsMu.Unlock()
	return caps
}

// deriveCaps is the pure detection logic (no I/O) so it can be unit-tested.
func deriveCaps(pipeline string, tags, files []string, repo string) []string {
	tagHas := func(subs ...string) bool {
		for _, t := range tags {
			lt := strings.ToLower(t)
			for _, s := range subs {
				if strings.Contains(lt, s) {
					return true
				}
			}
		}
		return false
	}
	mmproj := false
	for _, f := range files {
		if strings.Contains(strings.ToLower(f), "mmproj") {
			mmproj = true
			break
		}
	}
	name := strings.ToLower(repo)
	nameHas := func(subs ...string) bool {
		for _, s := range subs {
			if strings.Contains(name, s) {
				return true
			}
		}
		return false
	}

	var caps []string
	if mmproj || pipeline == "image-text-to-text" || tagHas("vision", "multimodal", "image-text", "vlm") {
		caps = append(caps, "vision")
	}
	if tagHas("audio", "speech", "asr") || pipeline == "automatic-speech-recognition" {
		caps = append(caps, "audio")
	}
	if tagHas("reasoning") || nameHas("-r1", "qwq", "thinking", "reasoning", "deepseek-r1") {
		caps = append(caps, "reasoning")
	}
	if tagHas("function-calling", "tool-use", "tool-calling", "function_calling") {
		caps = append(caps, "tools")
	}
	if pipeline == "feature-extraction" || pipeline == "sentence-similarity" || tagHas("sentence-similarity", "embeddings") {
		caps = append(caps, "embeddings")
	}
	return caps
}
