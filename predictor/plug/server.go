// Package plug turns a running llamadeck server into coding-agent config: it
// prints the exact provider snippet for Claude Code / OpenCode / Codex /
// Cursor / Pi / Hermes, and can safely write it into the agent's config file
// (namespaced entries only — re-runs overwrite OUR entry, never anything else).
package plug

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"llamadeck/infra"
)

// Server is one running llama-server with everything an agent config needs.
type Server struct {
	Name    string
	Port    string
	ModelID string // exact id the server reports at /v1/models (what agents must send)
	Label   string // repo:quant launch label (human display)
	Ctx     int    // server context size (from /props; 0 = unknown)
	Jinja   bool   // launched with --jinja (required for agent tool calling)
}

// BaseURL returns the OpenAI-compatible root (with /v1).
func (s Server) BaseURL() string { return "http://localhost:" + s.Port + "/v1" }

// AnthropicURL returns the base for Claude Code (no /v1 — the client appends
// /v1/messages itself).
func (s Server) AnthropicURL() string { return "http://localhost:" + s.Port }

const probeTimeout = 300 * time.Millisecond

// DiscoverServers lists running managed servers, probing each for its real
// model id (/v1/models) and context size (/props). Probes are best-effort:
// a slow/unresponsive server still appears, with fallbacks from its labels.
func DiscoverServers() ([]Server, error) {
	cs, err := infra.Containers()
	if err != nil {
		return nil, err
	}
	var out []Server
	for _, c := range cs {
		if s, ok := ProbeContainer(c); ok {
			out = append(out, s)
		}
	}
	return out, nil
}

// ProbeContainer turns one managed container into a plug-ready Server,
// probing it for the real model id and context size. False when the container
// isn't a running server with a port.
func ProbeContainer(c infra.Container) (Server, bool) {
	if c.State != "running" || c.Port == "" {
		return Server{}, false
	}
	s := Server{Name: c.Name, Port: c.Port, Label: c.Model, ModelID: c.Model}
	if id := probeModelID(c.Port); id != "" {
		s.ModelID = id
	}
	s.Ctx = probeCtx(c.Port)
	s.Jinja = containerHasJinja(c.Name)
	return s, true
}

func httpJSON(url string, v any) error {
	client := http.Client{Timeout: probeTimeout}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

// probeModelID asks the server what it calls its model — the id agents must
// put in the "model" field. Empty on any error.
func probeModelID(port string) string {
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := httpJSON("http://localhost:"+port+"/v1/models", &body); err != nil || len(body.Data) == 0 {
		return ""
	}
	return body.Data[0].ID
}

// probeCtx reads the server's real context size from /props. 0 on any error.
func probeCtx(port string) int {
	var body struct {
		DefaultGenerationSettings struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := httpJSON("http://localhost:"+port+"/props", &body); err != nil {
		return 0
	}
	return body.DefaultGenerationSettings.NCtx
}

// containerHasJinja reports whether the container was launched with --jinja
// (llama-server needs it for tool calling — agents degrade badly without it).
func containerHasJinja(name string) bool {
	out, err := exec.Command("docker", "inspect", "-f", "{{json .Args}}", name).Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), `"--jinja"`)
}
