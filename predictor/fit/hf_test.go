package fit

import (
	"reflect"
	"testing"
)

func TestParseRef(t *testing.T) {
	cases := []struct {
		in    string
		repo  string
		quant string
		ok    bool
	}{
		{"unsloth/Llama-3.2-1B-Instruct-GGUF", "unsloth/Llama-3.2-1B-Instruct-GGUF", "", true},
		{"unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M", true},
		{"https://huggingface.co/x/y/resolve/main/z.gguf", "", "", false},
		{"./model.gguf", "", "", false},
		{"/abs/model.gguf", "", "", false},
		{"model.gguf", "", "", false}, // no slash → not a repo ref
	}
	for _, c := range cases {
		ref, ok := ParseRef(c.in)
		if ok != c.ok || ref.Repo != c.repo || ref.Quant != c.quant {
			t.Errorf("ParseRef(%q) = %+v,%v; want {%q,%q},%v", c.in, ref, ok, c.repo, c.quant, c.ok)
		}
	}
}

var sampleFiles = []string{
	"README.md",
	"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
	"Llama-3.2-1B-Instruct-Q8_0.gguf",
	"Llama-3.2-1B-Instruct-BF16.gguf",
	"mmproj-F16.gguf", // must be excluded
	"config.json",
}

func TestListQuants(t *testing.T) {
	got := ListQuants(sampleFiles)
	want := []string{"BF16", "Q4_K_M", "Q8_0"} // sorted, mmproj excluded
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListQuants = %v, want %v", got, want)
	}
}

func TestSelectGGUF(t *testing.T) {
	shards, chosen, err := SelectGGUF(sampleFiles, "Q4_K_M")
	if err != nil {
		t.Fatal(err)
	}
	if chosen != "Q4_K_M" || len(shards) != 1 || shards[0] != "Llama-3.2-1B-Instruct-Q4_K_M.gguf" {
		t.Errorf("got %v / %q", shards, chosen)
	}
}

func TestSelectGGUFAutoPick(t *testing.T) {
	_, chosen, err := SelectGGUF(sampleFiles, "")
	if err != nil || chosen != "BF16" { // first sorted
		t.Errorf("auto-pick = %q, err %v; want BF16", chosen, err)
	}
}

func TestSelectGGUFMissing(t *testing.T) {
	_, _, err := SelectGGUF(sampleFiles, "Q2_K")
	if err == nil {
		t.Fatal("expected error for missing quant")
	}
}

func TestSelectGGUFShardsSorted(t *testing.T) {
	files := []string{
		"big-Q4_K_M-00003-of-00003.gguf",
		"big-Q4_K_M-00001-of-00003.gguf",
		"big-Q4_K_M-00002-of-00003.gguf",
	}
	shards, _, err := SelectGGUF(files, "Q4_K_M")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"big-Q4_K_M-00001-of-00003.gguf",
		"big-Q4_K_M-00002-of-00003.gguf",
		"big-Q4_K_M-00003-of-00003.gguf",
	}
	if !reflect.DeepEqual(shards, want) {
		t.Errorf("shards = %v, want sorted %v", shards, want)
	}
}
