package plug

// Golden checks on the per-agent facts pinned from official docs (July 2026).
// If one of these fails after an edit, you are about to ship a config that the
// agent will reject — go re-verify against the docs, don't loosen the test.

import (
	"encoding/json"
	"strings"
	"testing"
)

var srv = Server{
	Name: "llama-test", Port: "8080",
	ModelID: "qwen3.5-9b", Label: "unsloth/Qwen3.5-9B-GGUF:Q4_K_S",
	Ctx: 32768, Jinja: true,
}

func TestAgentRoster(t *testing.T) {
	want := []string{"claude", "opencode", "codex", "cursor", "pi", "hermes"}
	got := Agents()
	if len(got) != len(want) {
		t.Fatalf("agents = %d, want %d", len(got), len(want))
	}
	for i, a := range got {
		if a.Key != want[i] {
			t.Fatalf("agent %d = %s, want %s", i, a.Key, want[i])
		}
		if a.Snippet(srv) == "" || a.HowTo(srv, "x") == "" {
			t.Fatalf("%s: empty snippet/howto", a.Key)
		}
		if (a.Write != nil) != a.CanWrite {
			t.Fatalf("%s: Write presence disagrees with CanWrite", a.Key)
		}
	}
	if _, ok := FindAgent("CODEX"); !ok {
		t.Fatal("FindAgent must be case-insensitive")
	}
	if _, ok := FindAgent("emacs"); ok {
		t.Fatal("unknown agent must not resolve")
	}
}

// Claude Code talks Anthropic wire format: base URL has NO /v1 suffix (the
// client appends /v1/messages), and a dummy token is mandatory.
func TestClaudeSnippetFacts(t *testing.T) {
	a, _ := FindAgent("claude")
	snip := a.Snippet(srv)
	if !strings.Contains(snip, `"ANTHROPIC_BASE_URL": "http://localhost:8080"`) {
		t.Fatalf("claude base URL must have NO /v1:\n%s", snip)
	}
	if !strings.Contains(snip, `"ANTHROPIC_AUTH_TOKEN": "dummy"`) {
		t.Fatal("claude needs a dummy auth token (login fallback otherwise)")
	}
	if !strings.Contains(snip, `"ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.5-9b"`) {
		t.Fatal("background/fast model must also point local")
	}
	var v map[string]any
	if err := json.Unmarshal([]byte(snip), &v); err != nil {
		t.Fatalf("claude snippet must be valid JSON: %v", err)
	}
}

// OpenCode + Pi use the OpenAI-compatible endpoint: /v1 IS in the URL, entries
// are namespaced under "llamadeck", and the snippets are valid JSON.
func TestOpenAICompatibleSnippetFacts(t *testing.T) {
	for _, key := range []string{"opencode", "pi"} {
		a, _ := FindAgent(key)
		snip := a.Snippet(srv)
		if !strings.Contains(snip, `"http://localhost:8080/v1"`) {
			t.Fatalf("%s: base URL must include /v1:\n%s", key, snip)
		}
		if !strings.Contains(snip, `"llamadeck"`) {
			t.Fatalf("%s: entry must be namespaced llamadeck", key)
		}
		var v map[string]any
		if err := json.Unmarshal([]byte(snip), &v); err != nil {
			t.Fatalf("%s snippet must be valid JSON: %v", key, err)
		}
	}
	if snip := mustAgent("pi").Snippet(srv); !strings.Contains(snip, `"api": "openai-completions"`) {
		t.Fatal("pi must use openai-completions for llama.cpp")
	}
	if snip := mustAgent("opencode").Snippet(srv); !strings.Contains(snip, "@ai-sdk/openai-compatible") {
		t.Fatal("opencode must use @ai-sdk/openai-compatible (NOT @ai-sdk/openai)")
	}
}

// Codex: /v1 in base_url, wire_api and env_key must be ABSENT (responses is
// the default; chat is deprecated; no auth header for a local server), and we
// only add a profile — never top-level model keys.
func TestCodexSnippetFacts(t *testing.T) {
	snip := mustAgent("codex").Snippet(srv)
	for _, banned := range []string{"wire_api", "env_key"} {
		if strings.Contains(snip, banned) {
			t.Fatalf("codex snippet must omit %s:\n%s", banned, snip)
		}
	}
	if !strings.Contains(snip, `base_url = "http://localhost:8080/v1"`) {
		t.Fatal("codex base_url must include /v1")
	}
	if !strings.Contains(snip, "[profiles.llamadeck]") {
		t.Fatal("codex must ship a profile, not top-level model keys")
	}
	if strings.HasPrefix(snip, "model") {
		t.Fatal("codex snippet must not set top-level model keys (would hijack the default provider)")
	}
}

// Cursor is print-only and must say WHY (backend routing → tunnel required).
func TestCursorIsHonest(t *testing.T) {
	a := mustAgent("cursor")
	if a.CanWrite || a.Write != nil {
		t.Fatal("cursor must be print-only")
	}
	how := a.HowTo(srv, "")
	for _, want := range []string{"backend", "localhost", "tunnel"} {
		if !strings.Contains(strings.ToLower(how), want) {
			t.Fatalf("cursor howto must mention %q:\n%s", want, how)
		}
	}
}

// Hermes: valid YAML-ish model block with /v1, and the write path refuses
// foreign configs (mustCreate semantics tested in markers_test.go).
func TestHermesSnippetFacts(t *testing.T) {
	snip := mustAgent("hermes").Snippet(srv)
	for _, want := range []string{"model:", "default: qwen3.5-9b", "base_url: http://localhost:8080/v1"} {
		if !strings.Contains(snip, want) {
			t.Fatalf("hermes snippet missing %q:\n%s", want, snip)
		}
	}
}

func mustAgent(key string) Agent {
	a, ok := FindAgent(key)
	if !ok {
		panic(key)
	}
	return a
}
