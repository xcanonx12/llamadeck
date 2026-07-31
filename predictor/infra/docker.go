// Package infra is the system-integration layer: talking to Docker and the GPU,
// and launching/stopping model servers. It shells out to the `docker` and
// `nvidia-smi` CLIs so the binary stays self-contained (no loose bash scripts to
// ship alongside it). All functions are safe to call when the tools are absent —
// they return empty results or errors rather than panicking.
package infra

import (
	"fmt"
	"net"
	"os/exec"
	"strconv"
	"strings"
)

// ImageTag is the server image the toolkit builds and runs.
const ImageTag = "local/llama.cpp:server-cuda"

// ManagedLabel marks every container this tool starts.
const ManagedLabel = "com.llamacpp.managed=true"

// Container is one managed llama-server instance.
type Container struct {
	Name   string
	Model  string // com.llamacpp.model label (repo:quant)
	State  string // running | exited | ...
	Health string // healthy | unhealthy | starting | n/a
	Port   string
}

// GPU is a single accelerator's live memory/util snapshot.
type GPU struct {
	Name     string
	MemUsed  int64 // bytes
	MemTotal int64 // bytes
	MemFree  int64 // bytes
	UtilPct  int
	Unified  bool // memory is the host's pool (DGX Spark / Jetson), not separate VRAM
}

func run(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).Output()
	return string(out), err
}

// DockerAvailable reports whether the docker daemon is reachable.
func DockerAvailable() bool { ok, _ := DockerStatus(); return ok }

// DockerStatus probes Docker and, on failure, returns WHY. Plain `docker info`
// exits non-zero for two very different reasons — no CLI at all, or a CLI that
// can't reach the daemon (socket permissions: the common case on DGX OS and any
// fresh install before the 'docker' group takes effect) — and reporting both as
// "Docker not found" sends people installing a Docker they already have.
// On success the string is the daemon version.
func DockerStatus() (bool, string) {
	if _, err := exec.LookPath("docker"); err != nil {
		return false, "docker CLI not in PATH"
	}
	out, err := exec.Command("docker", "info", "--format", "{{.ServerVersion}}").CombinedOutput()
	msg := strings.TrimSpace(string(out))
	if err == nil {
		return true, msg
	}
	return false, dockerFailReason(msg)
}

// dockerFailReason turns `docker info` stderr into one actionable line.
func dockerFailReason(out string) string {
	low := strings.ToLower(out)
	switch {
	case strings.Contains(low, "permission denied"):
		return "daemon socket denied — run: sudo usermod -aG docker $USER, then re-login (or newgrp docker)"
	case strings.Contains(low, "cannot connect") || strings.Contains(low, "is the docker daemon running"):
		return "daemon not running — run: sudo systemctl enable --now docker"
	}
	for _, line := range strings.Split(out, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return "docker info failed"
}

// ImageExists reports whether the given image tag is present locally.
func ImageExists(tag string) bool {
	out, err := run("docker", "images", "-q", tag)
	return err == nil && strings.TrimSpace(out) != ""
}

// ServerImageExists is a convenience for the toolkit's server image.
func ServerImageExists() bool { return ImageExists(ImageTag) }

// Containers lists every managed llama-server container with live status.
func Containers() ([]Container, error) {
	out, err := run("docker", "ps", "-a", "--filter", "label="+ManagedLabel, "--format", "{{.Names}}")
	if err != nil {
		return nil, err
	}
	var cs []Container
	for _, name := range strings.Fields(out) {
		c := Container{Name: name, Health: "n/a"}
		c.State = inspect(name, "{{.State.Status}}")
		if h := inspect(name, "{{if .State.Health}}{{.State.Health.Status}}{{end}}"); h != "" {
			c.Health = h
		}
		c.Model = inspect(name, `{{index .Config.Labels "com.llamacpp.model"}}`)
		ports := inspect(name, "{{range $p,$_ := .NetworkSettings.Ports}}{{$p}} {{end}}")
		if f := strings.FieldsFunc(ports, func(r rune) bool { return r == '/' || r == ' ' }); len(f) > 0 {
			c.Port = f[0]
		}
		cs = append(cs, c)
	}
	return cs, nil
}

func inspect(name, format string) string {
	out, err := run("docker", "inspect", "-f", format, name)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(out)
}

// GPUs returns a live snapshot of each NVIDIA GPU. Empty if nvidia-smi is absent.
func GPUs() []GPU {
	out, err := run("nvidia-smi",
		"--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu",
		"--format=csv,noheader,nounits")
	if err != nil {
		return nil
	}
	var gs []GPU
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 5 {
			continue
		}
		mb := func(s string) int64 {
			n, _ := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
			return n << 20
		}
		util, _ := strconv.Atoi(strings.TrimSpace(f[4]))
		g := GPU{
			Name:     strings.TrimSpace(f[0]),
			MemUsed:  mb(f[1]),
			MemTotal: mb(f[2]),
			MemFree:  mb(f[3]),
			UtilPct:  util,
		}
		// Unified-memory parts (DGX Spark GB10, Jetson) have no dedicated VRAM:
		// they share the host's LPDDR pool and nvidia-smi prints "[N/A]" for the
		// memory columns, which parses to 0 and makes the predictor think the GPU
		// is unusable. Substitute the host pool and mark it so the predictor
		// doesn't count that memory twice (once as VRAM, once as RAM).
		if g.Unified = isUnifiedGPU(g.Name) || g.MemTotal == 0; g.Unified {
			g.MemTotal, g.MemFree = TotalRAM(), FreeRAM()
			g.MemUsed = g.MemTotal - g.MemFree
		}
		gs = append(gs, g)
	}
	return gs
}

// isUnifiedGPU matches the SoC parts whose "VRAM" is the host's RAM.
// ponytail: name match, not a driver query — add a name here when a new
// unified part ships (nvidia-smi -L shows it).
func isUnifiedGPU(name string) bool {
	up := strings.ToUpper(name)
	for _, k := range []string{"GB10", "SPARK", "ORIN", "XAVIER", "THOR", "TEGRA"} {
		if strings.Contains(up, k) {
			return true
		}
	}
	return false
}

// Stop stops a container; Remove force-removes it.
func Stop(name string) error   { return exec.Command("docker", "stop", name).Run() }
func Remove(name string) error { return exec.Command("docker", "rm", "-f", name).Run() }

// Logs returns the last n lines of a container's combined output.
func Logs(name string, n int) (string, error) {
	out, err := exec.Command("docker", "logs", "--tail", strconv.Itoa(n), name).CombinedOutput()
	return string(out), err
}

// RemoveImage deletes the server image (used by the Update flow).
func RemoveImage(tag string) error { return exec.Command("docker", "image", "rm", tag).Run() }

// ContainerPID returns the host PID of a container's main process (0 on error).
func ContainerPID(name string) int {
	out, err := run("docker", "inspect", "-f", "{{.State.Pid}}", name)
	if err != nil {
		return 0
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(out))
	return pid
}

// GPUProcessMem returns the total VRAM (bytes) a given host PID holds across all
// GPUs, per nvidia-smi's compute-apps view. 0 if unavailable or no match.
func GPUProcessMem(pid int) int64 {
	out, err := run("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
	if err != nil {
		return 0
	}
	var total int64
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 2 {
			continue
		}
		if p, _ := strconv.Atoi(strings.TrimSpace(f[0])); p != pid {
			continue
		}
		if mb, err := strconv.ParseInt(strings.TrimSpace(f[1]), 10, 64); err == nil {
			total += mb << 20
		}
	}
	return total
}

// FreePort returns the first free TCP port at or above start (scanning up to
// start+200), falling back to start. Avoids the 8080 collision on a 2nd launch.
func FreePort(start int) int {
	for p := start; p < start+200; p++ {
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p))
		if err == nil {
			ln.Close()
			return p
		}
	}
	return start
}

// LaunchSpec describes a model server to start. Zero-valued optional fields are
// omitted from the command, so llama-server applies its own defaults.
type LaunchSpec struct {
	HF        string // repo:quant passed to llama-server -hf
	Port      int
	Ctx       int
	NGL       int
	CacheDir  string // host dir mounted at /root/.cache/huggingface
	FlashAttn bool

	// Device pins the container to a single GPU index (e.g. "0"); "" = all GPUs.
	Device string

	// Optional server tuning (mirrors the bash toolkit's flags).
	Host       string // bind address inside the container (default 0.0.0.0)
	Threads    int    // --threads (0 = server default)
	NCPUMoE    int    // --n-cpu-moe: offload N MoE layers to CPU (0 = none)
	UBatch     int    // --ubatch-size (0 = server default)
	Parallel   int    // --parallel request slots (0 = server default; scales hybrid RS state)
	CacheTypeK string // --cache-type-k (e.g. q8_0; "" = default f16)
	CacheTypeV string // --cache-type-v
	Jinja      bool   // --jinja (use the model's chat template)
	NoMmap     bool   // --no-mmap
	Mlock      bool   // --mlock

	// CPU launches with no GPU: omit --gpus so the container needs no NVIDIA
	// runtime. Pair with NGL: 0 (no offload).
	CPU bool
}

// TailLog returns the last n lines of a container's combined output (stdout +
// stderr), where llama-server prints model download and load progress. Empty on
// any error so callers can treat it as "nothing to show yet".
func TailLog(name string, n int) string {
	out, err := exec.Command("docker", "logs", "--tail", strconv.Itoa(n), name).CombinedOutput()
	if err != nil {
		return ""
	}
	return strings.TrimRight(string(out), "\n")
}

// SlugifyName builds a docker-safe container name from a model ref.
func SlugifyName(hf string) string {
	var b strings.Builder
	prev := false
	for _, r := range strings.ToLower(hf) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			prev = false
		} else if !prev {
			b.WriteByte('-')
			prev = true
		}
	}
	return "llama-" + strings.Trim(b.String(), "-")
}

// Launch starts a model server via `docker run`, mirroring the compose the bash
// toolkit generates (managed labels, GPU reservation, shared cache mount, 1:1
// port). Returns the container name. Self-contained — no compose file needed.
func Launch(s LaunchSpec) (string, error) {
	if s.Port == 0 {
		s.Port = FreePort(8080)
	}
	name := SlugifyName(s.HF)
	out, err := exec.Command("docker", launchArgs(name, s)...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("docker run: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return name, nil
}

// launchArgs builds the full `docker run` argument list for a spec. Pure (no
// side effects) so the flag-mapping can be unit-tested without Docker.
func launchArgs(name string, s LaunchSpec) []string {
	host := s.Host
	if host == "" {
		host = "0.0.0.0"
	}
	args := []string{
		"run", "-d", "--name", name,
		"--restart", "unless-stopped",
	}
	if !s.CPU {
		gpus := "all"
		if s.Device != "" {
			gpus = "device=" + s.Device
		}
		args = append(args, "--gpus", gpus)
	}
	args = append(args,
		"--label", ManagedLabel,
		"--label", "com.llamacpp.model=" + s.HF,
		"-p", fmt.Sprintf("%d:%d", s.Port, s.Port),
		"-v", s.CacheDir + ":/root/.cache/huggingface",
		// Override the image's hardcoded :8080 healthcheck to the ACTUAL port, or
		// Docker reports "unhealthy" forever while the server is fine. Long
		// start-period so a big model download/load shows "starting", not failing.
		"--health-cmd", fmt.Sprintf("curl -fsS http://localhost:%d/health || exit 1", s.Port),
		"--health-interval", "10s",
		"--health-timeout", "5s",
		"--health-start-period", "1200s",
		"--health-retries", "3",
	)
	if s.Mlock {
		// Docker's default RLIMIT_MEMLOCK is tiny, so --mlock inside the container
		// silently does nothing but spam "failed to mlock" warnings. Lift the limit.
		args = append(args, "--ulimit", "memlock=-1")
	}
	args = append(args,
		ImageTag,
		"-hf", s.HF,
		"--host", host,
		"--port", strconv.Itoa(s.Port),
		"--ctx-size", strconv.Itoa(s.Ctx),
		"-ngl", strconv.Itoa(s.NGL),
	)
	if s.FlashAttn {
		args = append(args, "--flash-attn", "on")
	}
	if s.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(s.Threads))
	}
	if s.NCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(s.NCPUMoE))
	}
	if s.UBatch > 0 {
		args = append(args, "--ubatch-size", strconv.Itoa(s.UBatch))
	}
	if s.Parallel > 0 {
		args = append(args, "--parallel", strconv.Itoa(s.Parallel))
	}
	if s.CacheTypeK != "" {
		args = append(args, "--cache-type-k", s.CacheTypeK)
	}
	if s.CacheTypeV != "" {
		args = append(args, "--cache-type-v", s.CacheTypeV)
	}
	if s.Jinja {
		args = append(args, "--jinja")
	}
	if s.NoMmap {
		args = append(args, "--no-mmap")
	}
	if s.Mlock {
		args = append(args, "--mlock")
	}
	return args
}
