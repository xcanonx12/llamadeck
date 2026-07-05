// Package hub provides model discovery: a curated list of popular GGUF repos to
// browse out of the box, plus live Hugging Face search.
package hub

import (
	"encoding/json"
	"net/http"
	"net/url"
)

// Model is a discoverable model repo.
type Model struct {
	Repo      string // owner/name
	Note      string // human hint (params / family)
	Downloads int
	Likes     int
}

// TopModels returns the most-downloaded GGUF repos on the Hub (up to 50) so the
// Models tab is a browsable list out of the box. Falls back to a curated set if
// the Hub is unreachable.
func TopModels() []Model {
	ms, err := queryModels(url.Values{
		"filter": {"gguf"},
		"sort":   {"downloads"},
		"limit":  {"50"},
	})
	if err != nil || len(ms) == 0 {
		return curatedModels()
	}
	return ms
}

// Search queries the Hub for GGUF repos matching q, ranked by downloads.
func Search(q string) ([]Model, error) {
	if q == "" {
		return nil, nil
	}
	return queryModels(url.Values{
		"search": {q},
		"filter": {"gguf"},
		"sort":   {"downloads"},
		"limit":  {"50"},
	})
}

// queryModels runs a Hub models query and maps the result to []Model.
func queryModels(v url.Values) ([]Model, error) {
	resp, err := hfGet("https://huggingface.co/api/models?" + v.Encode())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, httpErr(resp, "HF search")
	}
	var raw []struct {
		ID        string `json:"id"`
		Downloads int    `json:"downloads"`
		Likes     int    `json:"likes"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}
	out := make([]Model, 0, len(raw))
	for _, r := range raw {
		out = append(out, Model{Repo: r.ID, Downloads: r.Downloads, Likes: r.Likes})
	}
	return out, nil
}

// CuratedModels is the offline fallback starter set, used to seed the UI
// instantly before the live top-models fetch returns.
func CuratedModels() []Model { return curatedModels() }

// curatedModels is the offline fallback starter set of widely-used GGUF repos.
func curatedModels() []Model {
	return []Model{
		{Repo: "unsloth/Llama-3.2-1B-Instruct-GGUF", Note: "Llama 3.2 · 1B · tiny/fast"},
		{Repo: "unsloth/Llama-3.2-3B-Instruct-GGUF", Note: "Llama 3.2 · 3B"},
		{Repo: "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF", Note: "Llama 3.1 · 8B · general"},
		{Repo: "unsloth/Qwen2.5-7B-Instruct-GGUF", Note: "Qwen2.5 · 7B"},
		{Repo: "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", Note: "Qwen2.5 Coder · 7B · code"},
		{Repo: "unsloth/Qwen2.5-14B-Instruct-GGUF", Note: "Qwen2.5 · 14B"},
		{Repo: "unsloth/gemma-2-9b-it-GGUF", Note: "Gemma 2 · 9B"},
		{Repo: "unsloth/Mistral-7B-Instruct-v0.3-GGUF", Note: "Mistral · 7B"},
		{Repo: "unsloth/phi-4-GGUF", Note: "Phi-4 · 14B"},
		{Repo: "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", Note: "DeepSeek-R1 distill · 7B · reasoning"},
	}
}
