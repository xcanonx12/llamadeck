package main

// `llamadeck plug` — print (or write) the config that adds a running local
// server to a coding agent. Keyboard-first: everything is printed ready to
// paste, and --write applies it after a y/N confirm with a timestamped backup.

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"llamadeck/plug"
)

func plugUsage() {
	fmt.Fprintln(os.Stderr, `usage:
  llamadeck plug                       list running servers + supported agents
  llamadeck plug <agent>               print the config snippet for an agent
  llamadeck plug <agent> --write       add it to the agent's config (y/N + backup)

agents:  claude · opencode · codex · cursor · pi · hermes

flags:
  --server N|NAME   pick a server by port or container name (default: the only one)
  --write           apply the snippet to the agent's config file
  --yes             skip the confirmation prompt
  --global          claude only: write ~/.claude/settings.json instead of ./.claude/settings.local.json
  --config PATH     unusual install? point at the agent's config file explicitly
  --ctx N           override the probed context window`)
	os.Exit(2)
}

type plugOpts struct {
	agent, server, config string
	write, yes, global    bool
	ctx                   int
}

func parsePlugArgs(args []string) plugOpts {
	var o plugOpts
	for i := 0; i < len(args); i++ {
		next := func() string {
			i++
			if i >= len(args) {
				die("flag %s needs a value", args[i-1])
			}
			return args[i]
		}
		switch args[i] {
		case "--server":
			o.server = next()
		case "--config":
			o.config = next()
		case "--ctx":
			o.ctx = atoi(next())
		case "--write":
			o.write = true
		case "--yes":
			o.yes = true
		case "--global":
			o.global = true
		case "-h", "--help":
			plugUsage()
		default:
			if strings.HasPrefix(args[i], "-") || o.agent != "" {
				die("unknown argument: %s (try llamadeck plug --help)", args[i])
			}
			o.agent = args[i]
		}
	}
	return o
}

func runPlug(args []string) {
	o := parsePlugArgs(args)
	servers, err := plug.DiscoverServers()
	if err != nil {
		die("listing servers: %v", err)
	}
	if len(servers) == 0 {
		fmt.Println("no running llamadeck servers — launch one first (llamadeck → Fit tab → LAUNCH, or ./launch.sh)")
		return
	}

	if o.agent == "" {
		fmt.Println("running servers:")
		for _, s := range servers {
			jin := ""
			if !s.Jinja {
				jin = "   ⚠ no --jinja"
			}
			fmt.Printf("  :%-6s %-28s ctx %-7d %s%s\n", s.Port, s.ModelID, s.Ctx, s.Name, jin)
		}
		fmt.Println("\nagents:")
		for _, a := range plug.Agents() {
			mark := " "
			if a.Detected() {
				mark = "✓"
			}
			fmt.Printf("  %s %-9s llamadeck plug %s\n", mark, a.Key, a.Key)
		}
		fmt.Println("\npick one:  llamadeck plug <agent> [--server PORT] [--write]")
		return
	}

	agent, ok := plug.FindAgent(o.agent)
	if !ok {
		die("unknown agent %q — one of: claude, opencode, codex, cursor, pi, hermes", o.agent)
	}
	srv, err := pickServer(servers, o.server)
	if err != nil {
		die("%v", err)
	}
	if o.ctx > 0 {
		srv.Ctx = o.ctx
	}

	path := o.config
	if path == "" && agent.CanWrite {
		path = agent.Path(o.global)
	}

	// The snippet, always — pipes get exactly the config and nothing else on stdout.
	fmt.Printf("# %s ← %s (:%s, ctx %d)\n\n", agent.Name, srv.ModelID, srv.Port, srv.Ctx)
	fmt.Println(agent.Snippet(srv))
	fmt.Println()
	fmt.Println(agent.HowTo(srv, path))
	if !srv.Jinja {
		fmt.Println("\n⚠ this server was launched WITHOUT --jinja — tool calling will misbehave.")
		fmt.Println("  relaunch it with Jinja on before agent use (Fit tab → Jinja template → on).")
	}

	if !o.write {
		if agent.CanWrite {
			fmt.Printf("\napply it:  llamadeck plug %s --write\n", agent.Key)
		}
		return
	}
	if !agent.CanWrite {
		fmt.Printf("\n%s has no writable config — follow the steps above.\n", agent.Name)
		return
	}
	if !agent.Detected() && o.config == "" {
		die("%s doesn't look installed (no %q in PATH, no config dir).\nIf it lives somewhere unusual, point me at it: llamadeck plug %s --write --config PATH",
			agent.Name, agent.Key, agent.Key)
	}
	if !o.yes {
		if !isTTY(os.Stdout) {
			fmt.Println("\n(non-interactive: pass --yes to write without a prompt — nothing was changed)")
			return
		}
		warn := ""
		if agent.Key == "claude" && o.global {
			warn = " — this redirects EVERY claude session on this machine to the local server"
		}
		fmt.Printf("\nwrite this into %s%s? [y/N] ", path, warn)
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if l := strings.ToLower(strings.TrimSpace(line)); l != "y" && l != "yes" {
			fmt.Println("nothing changed.")
			return
		}
	}
	backup, err := agent.Write(srv, path)
	if err != nil {
		die("%v", err)
	}
	fmt.Printf("\n✓ written to %s", path)
	if backup != "" {
		fmt.Printf("   (backup: %s)", backup)
	}
	fmt.Println()
	fmt.Println("re-running `llamadeck plug` overwrites only the llamadeck entry — the rest of the file is never touched.")
}

// pickServer resolves --server (port or name); with one server it's implicit.
func pickServer(servers []plug.Server, sel string) (plug.Server, error) {
	if sel == "" {
		if len(servers) == 1 {
			return servers[0], nil
		}
		names := make([]string, len(servers))
		for i, s := range servers {
			names[i] = fmt.Sprintf(":%s %s", s.Port, s.Name)
		}
		return plug.Server{}, fmt.Errorf("multiple servers running — pick one with --server PORT|NAME:\n  %s",
			strings.Join(names, "\n  "))
	}
	for _, s := range servers {
		if s.Port == sel || s.Name == sel || strings.TrimPrefix(sel, ":") == s.Port {
			return s, nil
		}
	}
	return plug.Server{}, fmt.Errorf("no running server matches %q", sel)
}
