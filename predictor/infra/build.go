package infra

// Image build / update orchestration for the Update tab. Building needs the repo
// (Dockerfile + llama.cpp source + docker-build.sh), so these functions locate
// the repo root and reuse the existing, tested build script for CUDA/arch
// detection — while adding the git-update and dev-fork logic on top.

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const defaultLlamaRepo = "https://github.com/ggml-org/llama.cpp.git"

// RepoRoot walks up from the current directory looking for the toolkit root
// (identified by docker-build.sh + Dockerfile). Returns an error if not found —
// build/update require it even though launch/monitor/fit do not.
func RepoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if exists(filepath.Join(dir, "docker-build.sh")) && exists(filepath.Join(dir, "Dockerfile")) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("toolkit root not found (need docker-build.sh + Dockerfile); run from the repo to build")
		}
		dir = parent
	}
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

// runIn runs a command in dir, returning combined output.
func runIn(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// UpdateLlamaCppSource refreshes the llama.cpp checkout. With forkURL set it
// re-clones from that fork (the Dev option); otherwise it pulls the existing
// checkout, or clones upstream if absent.
func UpdateLlamaCppSource(root, forkURL string) (string, error) {
	dst := filepath.Join(root, "llama.cpp")
	repo := forkURL
	if repo == "" {
		repo = defaultLlamaRepo
	}
	if forkURL != "" {
		// Dev fork: replace the checkout entirely.
		if err := os.RemoveAll(dst); err != nil {
			return "", err
		}
		return runIn(root, "git", "clone", "--depth", "1", repo, "llama.cpp")
	}
	if !exists(filepath.Join(dst, ".git")) {
		return runIn(root, "git", "clone", "--depth", "1", repo, "llama.cpp")
	}
	return runIn(dst, "git", "pull", "--ff-only")
}

// BuildImageStream runs the toolkit's build script (CUDA tag + GPU arch
// detection, clone-if-missing, docker build), streaming combined output to w.
func BuildImageStream(root string, w io.Writer) error {
	cmd := exec.Command("bash", "docker-build.sh")
	cmd.Dir = root
	cmd.Stdout = w
	cmd.Stderr = w
	return cmd.Run()
}

// RebuildStream performs the full Update flow — remove the old image, refresh
// the llama.cpp source (optionally from a fork), then rebuild — streaming
// progress to w as it goes.
func RebuildStream(root, forkURL string, w io.Writer) error {
	fmt.Fprintln(w, "› removing old image…")
	_ = RemoveImage(ImageTag) // ignore "no such image" on first build
	fmt.Fprintln(w, "› updating llama.cpp source…")
	out, err := UpdateLlamaCppSource(root, forkURL)
	if out != "" {
		fmt.Fprintln(w, strings.TrimRight(out, "\n"))
	}
	if err != nil {
		return fmt.Errorf("updating llama.cpp: %w", err)
	}
	fmt.Fprintln(w, "› building image…")
	return BuildImageStream(root, w)
}
