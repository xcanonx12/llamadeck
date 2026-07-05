#!/usr/bin/env sh
# llamadeck installer. Usage:
#   curl -fsSL https://raw.githubusercontent.com/xcanonx12/llamadeck/main/install.sh | sh
#
# Downloads the latest prebuilt binary for your OS/arch from GitHub Releases and
# installs it to ~/.local/bin (or /usr/local/bin if writable). Override the repo
# with LLAMADECK_REPO=owner/name.
set -eu

REPO="${LLAMADECK_REPO:-xcanonx12/llamadeck}"
BINARY="llamadeck"

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$arch" in
  x86_64 | amd64) arch="amd64" ;;
  aarch64 | arm64) arch="arm64" ;;
  *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
esac
case "$os" in
  linux | darwin) ;;
  *) echo "unsupported OS: $os" >&2; exit 1 ;;
esac

# Resolve the latest release tag via the GitHub API.
tag=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | grep -m1 '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "${tag:-}" ]; then
  echo "could not determine latest release of ${REPO}" >&2
  echo "(has a release been published yet?)" >&2
  exit 1
fi

asset="${BINARY}_${os}_${arch}.tar.gz"
url="https://github.com/${REPO}/releases/download/${tag}/${asset}"
echo "downloading ${BINARY} ${tag} (${os}/${arch})…"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp/$asset"
tar -xzf "$tmp/$asset" -C "$tmp"

# Pick an install dir: /usr/local/bin if writable, else ~/.local/bin.
if [ -w /usr/local/bin ] 2>/dev/null; then
  dest="/usr/local/bin"
else
  dest="$HOME/.local/bin"
  mkdir -p "$dest"
fi
install -m 0755 "$tmp/$BINARY" "$dest/$BINARY" 2>/dev/null || {
  cp "$tmp/$BINARY" "$dest/$BINARY" && chmod 0755 "$dest/$BINARY"
}

echo "installed ${BINARY} to ${dest}/${BINARY}"
case ":$PATH:" in
  *":$dest:"*) ;;
  *) echo "note: add ${dest} to your PATH" ;;
esac
