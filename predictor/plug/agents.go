package plug

// The six target agents. Every fact here (paths, schemas, which URL gets /v1,
// wire formats, auth quirks) was pinned against official docs in July 2026 —
// see the PR/commit description. llama-server serves BOTH wire formats
// natively: Anthropic /v1/messages (Claude Code) and OpenAI
// /v1/chat/completions + /v1/responses (everything else) — no proxy needed.

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Agent describes one coding agent target.
type Agent struct {
	Key      string // CLI name: llamadeck plug <key>
	Name     string
	Bin      string                                      // binary to detect in PATH ("" = GUI app)
	Path     func(global bool) string                    // config file to write ("" = print-only)
	Snippet  func(s Server) string                       // what goes in the config (or GUI values)
	HowTo    func(s Server, path string) string          // where it goes + how to use it
	Write    func(s Server, path string) (string, error) // apply; nil = print-only agent
	CanWrite bool
}

// ctxOr returns the server ctx or a safe fallback for config fields that want one.
func ctxOr(s Server, fallback int) int {
	if s.Ctx > 0 {
		return s.Ctx
	}
	return fallback
}

// display returns a human name for the model (the launch label, else the id).
func display(s Server) string {
	if s.Label != "" {
		return s.Label
	}
	return s.ModelID
}

// Agents returns the six targets, in menu order.
func Agents() []Agent {
	return []Agent{claudeAgent(), opencodeAgent(), codexAgent(), cursorAgent(), piAgent(), hermesAgent()}
}

// FindAgent resolves a CLI key ("claude", "opencode", ...).
func FindAgent(key string) (Agent, bool) {
	for _, a := range Agents() {
		if a.Key == strings.ToLower(key) {
			return a, true
		}
	}
	return Agent{}, false
}

// Detected reports whether the agent looks installed (binary in PATH or its
// config dir exists).
func (a Agent) Detected() bool {
	if a.Bin != "" {
		if _, err := exec.LookPath(a.Bin); err == nil {
			return true
		}
	}
	if p := a.Path(true); p != "" {
		if _, err := os.Stat(filepath.Dir(p)); err == nil {
			return true
		}
	}
	return false
}

func home() string { h, _ := os.UserHomeDir(); return h }

// --- Claude Code -----------------------------------------------------------
// Anthropic wire format: llama-server serves /v1/messages natively (Jan 2026).
// Base URL has NO /v1 suffix — Claude Code appends /v1/messages itself. A
// dummy auth token is REQUIRED or the CLI falls back to subscription login.

func claudeEnv(s Server) map[string]any {
	return map[string]any{
		"ANTHROPIC_BASE_URL":            s.AnthropicURL(),
		"ANTHROPIC_AUTH_TOKEN":          "dummy",
		"ANTHROPIC_MODEL":               s.ModelID,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL": s.ModelID, // background/fast calls must not target an Anthropic model
	}
}

func claudeAgent() Agent {
	return Agent{
		Key: "claude", Name: "Claude Code", Bin: "claude", CanWrite: true,
		Path: func(global bool) string {
			if global {
				return filepath.Join(home(), ".claude", "settings.json")
			}
			return filepath.Join(".claude", "settings.local.json") // project-scoped, gitignored by default
		},
		Snippet: func(s Server) string {
			return fmt.Sprintf(`{
  "env": {
    "ANTHROPIC_BASE_URL": %q,
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": %q,
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": %q
  }
}`, s.AnthropicURL(), s.ModelID, s.ModelID)
		},
		HowTo: func(s Server, path string) string {
			return fmt.Sprintf(`goes in:   %s   (project-scoped; --global targets ~/.claude/settings.json and redirects EVERY claude session)
one-shot:  ANTHROPIC_BASE_URL=%s ANTHROPIC_AUTH_TOKEN=dummy ANTHROPIC_MODEL=%q claude
uses llama-server's native Anthropic /v1/messages endpoint — no proxy needed`,
				path, s.AnthropicURL(), s.ModelID)
		},
		Write: func(s Server, path string) (string, error) {
			return mergeJSONFile(path, []string{"env"}, claudeEnv(s))
		},
	}
}

// --- OpenCode ---------------------------------------------------------------
// @ai-sdk/openai-compatible targets /v1/chat/completions; provider key is
// namespaced "llamadeck" so re-runs overwrite only our entry.

func opencodeProvider(s Server) map[string]any {
	return map[string]any{
		"npm":     "@ai-sdk/openai-compatible",
		"name":    "llamadeck (local llama.cpp)",
		"options": map[string]any{"baseURL": s.BaseURL()},
		"models": map[string]any{
			s.ModelID: map[string]any{
				"name":  display(s),
				"limit": map[string]any{"context": ctxOr(s, 32768), "output": 8192},
			},
		},
	}
}

func opencodeAgent() Agent {
	return Agent{
		Key: "opencode", Name: "OpenCode", Bin: "opencode", CanWrite: true,
		Path: func(bool) string {
			cfg, _ := os.UserConfigDir()
			return filepath.Join(cfg, "opencode", "opencode.json")
		},
		Snippet: func(s Server) string {
			return fmt.Sprintf(`{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamadeck": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llamadeck (local llama.cpp)",
      "options": { "baseURL": %q },
      "models": {
        %q: { "name": %q, "limit": { "context": %d, "output": 8192 } }
      }
    }
  }
}`, s.BaseURL(), s.ModelID, display(s), ctxOr(s, 32768))
		},
		HowTo: func(s Server, path string) string {
			return fmt.Sprintf(`goes in:   %s   (deep-merged — your other providers are untouched)
use it:    opencode --model llamadeck/%s   (or /models in the TUI)`, path, s.ModelID)
		},
		Write: func(s Server, path string) (string, error) {
			return mergeJSONFile(path, []string{"provider", "llamadeck"}, opencodeProvider(s))
		},
	}
}

// --- Codex CLI ---------------------------------------------------------------
// base_url INCLUDES /v1 (Codex appends /responses). wire_api is omitted on
// purpose: "responses" is the default and "chat" is deprecated — llama-server
// serves /v1/responses natively. env_key omitted = no auth header (local).
// We add a PROFILE, never the top-level model keys, so the user's default
// provider is untouched: `codex --profile llamadeck`.

func codexBlock(s Server) string {
	return fmt.Sprintf(`[model_providers.llamadeck]
name = "llamadeck (local llama.cpp)"
base_url = %q

[profiles.llamadeck]
model = %q
model_provider = "llamadeck"
model_context_window = %d`, s.BaseURL(), s.ModelID, ctxOr(s, 32768))
}

func codexAgent() Agent {
	return Agent{
		Key: "codex", Name: "Codex CLI", Bin: "codex", CanWrite: true,
		Path:    func(bool) string { return filepath.Join(home(), ".codex", "config.toml") },
		Snippet: func(s Server) string { return codexBlock(s) },
		HowTo: func(s Server, path string) string {
			return fmt.Sprintf(`goes in:   %s   (appended as a marked block; your default model/provider are untouched)
use it:    codex --profile llamadeck`, path)
		},
		Write: func(s Server, path string) (string, error) {
			return writeMarkerBlock(path, codexBlock(s), false)
		},
	}
}

// --- Cursor ------------------------------------------------------------------
// Print-only, and honestly so: settings live in an undocumented SQLite store,
// and every request routes through Cursor's BACKEND — localhost is unreachable
// from there, so a public HTTPS tunnel is required. Only chat honors the
// override (Tab/Composer keep using Cursor's models).

func cursorAgent() Agent {
	return Agent{
		Key: "cursor", Name: "Cursor", Bin: "cursor", CanWrite: false,
		Path: func(bool) string { return "" },
		Snippet: func(s Server) string {
			return fmt.Sprintf(`Cursor Settings (Ctrl/Cmd+Shift+J) → Models → API Keys:
  OpenAI API Key:            sk-local-dummy        (any non-empty string)
  Override OpenAI Base URL:  https://<your-tunnel>/v1    ← NOT localhost, see below
  Add model:                 %s
  → click Verify, then enable ONLY this model in the picker`, s.ModelID)
		},
		HowTo: func(s Server, path string) string {
			return fmt.Sprintf(`⚠ Cursor routes every request through its own backend — it cannot reach localhost.
   Expose the server first:   cloudflared tunnel --url http://localhost:%s
                              (or: ngrok http %s)
   then use that https URL + /v1 as the override.
⚠ Only CHAT honors the override; Tab autocomplete and some features keep using
   Cursor's own models, and prompts still transit Cursor's servers.
   (No config file to write — these settings live in Cursor's internal store.)`, s.Port, s.Port)
		},
		Write: nil,
	}
}

// --- Pi ----------------------------------------------------------------------
// ~/.pi/agent/models.json, providers map; api MUST be "openai-completions" for
// llama.cpp; dummy apiKey literal (local servers ignore auth).

func piProvider(s Server) map[string]any {
	return map[string]any{
		"baseUrl": s.BaseURL(),
		"api":     "openai-completions",
		"apiKey":  "dummy",
		"models": []any{map[string]any{
			"id":            s.ModelID,
			"name":          display(s),
			"contextWindow": ctxOr(s, 32768),
			"maxTokens":     8192,
		}},
	}
}

func piAgent() Agent {
	return Agent{
		Key: "pi", Name: "Pi", Bin: "pi", CanWrite: true,
		Path: func(bool) string { return filepath.Join(home(), ".pi", "agent", "models.json") },
		Snippet: func(s Server) string {
			return fmt.Sprintf(`{
  "providers": {
    "llamadeck": {
      "baseUrl": %q,
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        { "id": %q, "name": %q, "contextWindow": %d, "maxTokens": 8192 }
      ]
    }
  }
}`, s.BaseURL(), s.ModelID, display(s), ctxOr(s, 32768))
		},
		HowTo: func(s Server, path string) string {
			return fmt.Sprintf(`goes in:   %s   (deep-merged — other providers untouched)
use it:    pi --model llamadeck/%s`, path, s.ModelID)
		},
		Write: func(s Server, path string) (string, error) {
			return mergeJSONFile(path, []string{"providers", "llamadeck"}, piProvider(s))
		},
	}
}

// --- Hermes -------------------------------------------------------------------
// ~/.hermes/config.yaml has ONE top-level model: block — writing it sets the
// agent's default model. We only create the file or replace our own marker
// block; a foreign config is never edited (YAML surgery breaks things).

func hermesBlock(s Server) string {
	return fmt.Sprintf(`model:
  default: %s
  provider: llamacpp
  base_url: %s`, s.ModelID, s.BaseURL())
}

func hermesAgent() Agent {
	return Agent{
		Key: "hermes", Name: "Hermes", Bin: "hermes", CanWrite: true,
		Path:    func(bool) string { return filepath.Join(home(), ".hermes", "config.yaml") },
		Snippet: func(s Server) string { return hermesBlock(s) },
		HowTo: func(s Server, path string) string {
			return fmt.Sprintf(`goes in:   %s   (top-level model: block — this SETS Hermes' default model)
note:      written only if the file is new or already carries a llamadeck block;
           an existing foreign config is never edited — paste manually there.
use it:    hermes   (or switch in-session with /model)`, path)
		},
		Write: func(s Server, path string) (string, error) {
			return writeMarkerBlock(path, hermesBlock(s), true)
		},
	}
}
