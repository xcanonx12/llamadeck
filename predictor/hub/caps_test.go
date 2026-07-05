package hub

import (
	"reflect"
	"testing"
)

func TestDeriveCaps(t *testing.T) {
	cases := []struct {
		name     string
		pipeline string
		tags     []string
		files    []string
		repo     string
		want     []string
	}{
		{"plain instruct", "text-generation", []string{"conversational"}, []string{"model-Q4_K_M.gguf"}, "unsloth/Qwen2.5-7B-Instruct-GGUF", nil},
		{"vision via mmproj", "text-generation", nil, []string{"model.gguf", "mmproj-F16.gguf"}, "x/llava-GGUF", []string{"vision"}},
		{"vision via pipeline", "image-text-to-text", nil, []string{"m.gguf"}, "x/y", []string{"vision"}},
		{"reasoning by name", "text-generation", nil, []string{"m.gguf"}, "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", []string{"reasoning"}},
		{"embeddings", "feature-extraction", nil, []string{"m.gguf"}, "x/embed", []string{"embeddings"}},
		{"audio tag", "text-generation", []string{"audio"}, []string{"m.gguf"}, "x/qwen-audio", []string{"audio"}},
	}
	for _, c := range cases {
		got := deriveCaps(c.pipeline, c.tags, c.files, c.repo)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}
