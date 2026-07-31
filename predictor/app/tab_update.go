package app

import (
	"bufio"
	"io"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"llamadeck/fit"
	"llamadeck/hub"
	"llamadeck/infra"
)

// hubHasToken reports whether an HF token is available (env or saved profile).
func hubHasToken() bool { return hub.HFToken() != "" }

const maxLogLines = 16

// buildMsg is one streamed build-output line, or the terminal done/err event.
type buildMsg struct {
	line string
	done bool
	err  error
}

type updateTab struct {
	sh          *shared
	fork        textinput.Model
	editingFork bool
	hfKey       textinput.Model
	editingKey  bool
	hasKey      bool // a token is already saved/in-env
	building    bool
	status      string
	log         []string
	events      chan buildMsg
}

func newUpdateTab(sh *shared) tab {
	ti := textinput.New()
	ti.Placeholder = "https://github.com/your/llama.cpp-fork.git"
	ti.CharLimit = 200

	key := textinput.New()
	key.Placeholder = "hf_xxxxxxxx (stored in ~/.config/llamadeck/profile.json)"
	key.EchoMode = textinput.EchoPassword
	key.CharLimit = 200

	t := &updateTab{sh: sh, fork: ti, hfKey: key}
	t.hasKey = hubHasToken()
	return t
}

func (t *updateTab) Title() string { return "Config" }
func (t *updateTab) Focused() bool { return t.editingFork || t.editingKey }
func (t *updateTab) Init() tea.Cmd { return nil }

func (t *updateTab) Update(msg tea.Msg) (tab, tea.Cmd) {
	switch msg := msg.(type) {
	case buildMsg:
		if msg.done {
			t.building = false
			t.events = nil
			if msg.err != nil {
				t.status = stErr.Render("build failed: " + msg.err.Error())
			} else {
				t.status = stOK.Render("image ready ✓")
				t.sh.imageOK = infra.ServerImageExists()
			}
			return t, nil
		}
		t.appendLog(msg.line)
		return t, waitBuild(t.events) // keep listening for the next line

	case tea.KeyMsg:
		if t.editingFork {
			switch msg.String() {
			case "esc", "enter":
				t.editingFork = false
				t.fork.Blur()
				return t, nil
			}
			var cmd tea.Cmd
			t.fork, cmd = t.fork.Update(msg)
			return t, cmd
		}
		if t.editingKey {
			switch msg.String() {
			case "esc":
				t.editingKey = false
				t.hfKey.Blur()
				return t, nil
			case "enter":
				t.editingKey = false
				t.hfKey.Blur()
				t.saveHFKey(strings.TrimSpace(t.hfKey.Value()))
				return t, nil
			}
			var cmd tea.Cmd
			t.hfKey, cmd = t.hfKey.Update(msg)
			return t, cmd
		}
		if t.building {
			return t, nil // ignore actions mid-build
		}
		if msg.String() == "d" {
			t.editingFork = true
			return t, t.fork.Focus()
		}
		if msg.String() == "k" {
			t.editingKey = true
			return t, t.hfKey.Focus()
		}
		if !t.sh.dockerOK && (msg.String() == "b" || msg.String() == "r") {
			t.status = stErr.Render("Docker not available — install/start Docker first")
			return t, nil
		}
		switch msg.String() {
		case "b":
			return t, t.startBuild("", false)
		case "r":
			return t, t.startBuild(strings.TrimSpace(t.fork.Value()), true)
		}
	}
	return t, nil
}

// saveHFKey persists the Hugging Face token to the profile (preserving the rest
// of the calibration state). Empty input clears it.
func (t *updateTab) saveHFKey(token string) {
	p, err := fit.LoadProfile()
	if err != nil {
		t.status = stErr.Render("could not load profile: " + err.Error())
		return
	}
	p.HFToken = token
	if err := p.Save(); err != nil {
		t.status = stErr.Render("could not save key: " + err.Error())
		return
	}
	t.hfKey.SetValue("")
	t.hasKey = hubHasToken()
	if token == "" {
		t.status = stOK.Render("Hugging Face key cleared")
	} else {
		t.status = stOK.Render("Hugging Face key saved ✓")
	}
}

func (t *updateTab) appendLog(line string) {
	t.log = append(t.log, line)
	if len(t.log) > maxLogLines {
		t.log = t.log[len(t.log)-maxLogLines:]
	}
}

func (t *updateTab) startBuild(fork string, rebuild bool) tea.Cmd {
	root, err := infra.RepoRoot()
	if err != nil {
		t.status = stErr.Render(err.Error())
		return nil
	}
	t.building, t.log = true, nil
	if rebuild {
		t.status = "rebuilding…"
	} else {
		t.status = "building… (first build can take 20–30 min)"
	}
	events := make(chan buildMsg, 256)
	t.events = events
	go runBuild(root, fork, rebuild, events)
	return waitBuild(events)
}

// runBuild streams the build's combined output line-by-line into events, then a
// final done event. Lives in a goroutine; the TUI drains it via waitBuild.
func runBuild(root, fork string, rebuild bool, events chan buildMsg) {
	pr, pw := io.Pipe()
	scanDone := make(chan struct{})
	go func() {
		sc := bufio.NewScanner(pr)
		sc.Buffer(make([]byte, 1<<20), 1<<20) // tolerate long BuildKit lines
		for sc.Scan() {
			events <- buildMsg{line: sc.Text()}
		}
		close(scanDone)
	}()

	var err error
	if rebuild {
		err = infra.RebuildStream(root, fork, pw)
	} else {
		err = infra.BuildImageStream(root, pw)
	}
	pw.Close() // end the scanner
	<-scanDone // ensure every line is flushed before signalling done
	events <- buildMsg{done: true, err: err}
	close(events)
}

// waitBuild blocks for the next streamed event (one per tea.Cmd invocation).
func waitBuild(events chan buildMsg) tea.Cmd {
	if events == nil {
		return nil
	}
	return func() tea.Msg {
		ev, ok := <-events
		if !ok {
			return nil
		}
		return ev
	}
}

func (t *updateTab) View(width, height int) string {
	var b strings.Builder

	b.WriteString(stBold.Render("Server image") + "  ")
	switch {
	case !t.sh.dockerOK:
		b.WriteString(stErr.Render("Docker not available") + "\n")
		b.WriteString(stMuted.Render("  "+t.sh.dockerWhy) + "\n\n")
	case t.sh.imageOK:
		b.WriteString(stOK.Render("✓ "+infra.ImageTag) + "\n\n")
	default:
		b.WriteString(stErr.Render("✗ not built") + "\n\n")
	}

	b.WriteString(stBold.Render("Actions") + "\n")
	b.WriteString("  " + stKey.Render("b") + stText.Render("  build image") +
		stMuted.Render("   (CUDA + GPU-arch auto-detected)") + "\n")
	b.WriteString("  " + stKey.Render("r") + stText.Render("  rebuild") +
		stMuted.Render("   (remove image → update llama.cpp → build)") + "\n")
	b.WriteString("  " + stKey.Render("d") + stText.Render("  set dev fork URL") +
		stMuted.Render("   (rebuild from an alternative llama.cpp fork)") + "\n")
	b.WriteString("  " + stKey.Render("k") + stText.Render("  set Hugging Face key") +
		stMuted.Render("   (for gated/private models · fixes 401 errors)") + "\n\n")

	b.WriteString(stBold.Render("Dev fork") + "\n  ")
	if t.editingFork {
		b.WriteString(t.fork.View() + "\n")
	} else if v := strings.TrimSpace(t.fork.Value()); v != "" {
		b.WriteString(stText.Render(v) + stMuted.Render("   (used on next rebuild)") + "\n")
	} else {
		b.WriteString(stMuted.Render("upstream ggml-org/llama.cpp   (press d to override)") + "\n")
	}

	b.WriteString("\n" + stBold.Render("Hugging Face") + "\n  ")
	if t.editingKey {
		b.WriteString(t.hfKey.View() + "\n")
		b.WriteString(stMuted.Render("  enter to save · esc to cancel · blank clears it") + "\n")
	} else if t.hasKey {
		b.WriteString(stOK.Render("✓ access token configured") +
			stMuted.Render("   (press k to change · clears gated/401 errors)") + "\n")
	} else {
		b.WriteString(stMuted.Render("no token — press ") + stKey.Render("k") +
			stMuted.Render(" to add one for gated/private models (fixes 401 errors)") + "\n")
	}

	if t.building {
		b.WriteString("\n" + stWarn.Render("⟳ building… you can keep using other tabs") + "\n")
	}
	if t.status != "" {
		b.WriteString("\n" + t.status + "\n")
	}
	if len(t.log) > 0 {
		b.WriteString("\n" + stDim.Render("── build log ──") + "\n")
		for _, ln := range t.log {
			b.WriteString(stDim.Render(truncate(ln, width-2)) + "\n")
		}
	}
	return b.String()
}
