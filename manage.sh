#!/usr/bin/env bash
set -euo pipefail
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cd "$(dirname "$0")"
fi
source lib/common.sh

LABEL="com.llamacpp.managed=true"

list_containers() { docker ps -a --filter "label=$LABEL" --format '{{.Names}}'; }

show_status() {
  header "managed llama.cpp servers"
  local names; names=$(list_containers)
  if [[ -z "$names" ]]; then log_warn "No managed containers found."; return 0; fi
  printf '%s\n' "$(bold 'NAME                                   STATE       HEALTH    PORT')"
  local name state health port hc
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo '?')
    health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}' "$name" 2>/dev/null || echo '?')
    port=$(docker inspect -f '{{range $p,$_ := .NetworkSettings.Ports}}{{$p}} {{end}}' "$name" 2>/dev/null | grep -oE '^[0-9]+' | head -n1 || true)
    hc="${_C_GREEN}"; [[ "$health" == "healthy" ]] || hc="${_C_YELLOW}"
    printf '%-38s %-11s %s%-9s%s %s\n' "$name" "$state" "$hc" "$health" "${_C_RESET}" "${port:-?}"
  done <<< "$names"
}

pick_container() { list_containers | fzf_select "container"; }

main() {
  local cmd="${1:-status}" c
  case "$cmd" in
    status|ls) show_status ;;
    stop) c="${2:-$(pick_container)}"; [[ -n "$c" ]] || die "no container selected"; docker stop "$c" >/dev/null && log_info "Stopped $c" ;;
    rm)   c="${2:-$(pick_container)}"; [[ -n "$c" ]] || die "no container selected"; docker rm -f "$c" >/dev/null && log_info "Removed $c" ;;
    logs) c="${2:-$(pick_container)}"; [[ -n "$c" ]] || die "no container selected"; docker logs -f "$c" ;;
    *) die "Usage: manage.sh [status|ls|stop|rm|logs] [container]" ;;
  esac
}

[[ -n "${MANAGE_LIB:-}" ]] || main "$@"
