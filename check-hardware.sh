#!/usr/bin/env bash
set -euo pipefail

# ponytail: find lib/common.sh relative to this script, accounting for sourcing from any CWD
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_script_dir}/lib/common.sh"

# ensure_pkg <cmd> <apt-package> <human-name>: install via apt if missing (safe userland only)
ensure_pkg() {
  local cmd="$1" pkg="$2" name="$3"
  if command -v "$cmd" >/dev/null 2>&1; then log_info "$name present"; return 0; fi
  log_warn "$name missing — installing $pkg"
  sudo apt-get update -qq && sudo apt-get install -y "$pkg" \
    || die "Failed to install $pkg — install it manually and re-run"
  log_info "$name installed"
}

# check_driver: hard requirement. Detect + print instructions, never auto-install.
check_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_err "NVIDIA driver / nvidia-smi not found."
    cat <<'EOF'
  Install the NVIDIA driver for your GPU, then re-run this script:
    Ubuntu: sudo ubuntu-drivers autoinstall   (then reboot)
    Or follow https://www.nvidia.com/Download/index.aspx
  This script will NOT auto-install drivers (kernel modules / reboot risk).
EOF
    return 1
  fi
  log_info "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  log_info "Max CUDA supported by driver -> image tag $(detect_cuda_tag)"
  return 0
}

check_docker() {
  ensure_pkg docker docker.io "Docker Engine"
  local err
  err=$(docker info 2>&1 >/dev/null) && { log_info "Docker daemon reachable"; return 0; }
  case "$err" in
    *"permission denied"*) log_warn "Docker installed but this user can't reach the socket: sudo usermod -aG docker \"\$USER\" && newgrp docker" ;;
    *"Cannot connect"*|*"daemon running"*) log_warn "Docker installed but not running: sudo systemctl enable --now docker" ;;
    *) log_warn "Docker installed but 'docker info' failed: $(printf '%s' "$err" | head -n1)" ;;
  esac
}

# check_toolkit: NVIDIA Container Toolkit — safe userland install.
check_toolkit() {
  if command -v nvidia-ctk >/dev/null 2>&1 || docker info 2>/dev/null | grep -qi 'nvidia'; then
    log_info "NVIDIA Container Toolkit present"; return 0
  fi
  log_warn "NVIDIA Container Toolkit missing — installing"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit \
    || die "Failed to install nvidia-container-toolkit — see https://docs.nvidia.com/datacenter/cloud-native/"
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  log_info "NVIDIA Container Toolkit installed"
}

main() {
  header "hardware & dependency check"
  check_driver || die "Fix the NVIDIA driver above, then re-run."
  check_docker
  check_toolkit
  ensure_pkg jq jq "jq"
  ensure_pkg curl curl "curl"
  command -v fzf    >/dev/null 2>&1 && log_info "fzf present (nicer pickers)"  || log_info "fzf not installed (optional — falls back to a numbered menu)"
  command -v figlet >/dev/null 2>&1 && log_info "figlet present (fancy banner)" || true
  log_info "All prerequisites satisfied."
}

[[ -n "${CHECK_HW_LIB:-}" ]] || main "$@"
