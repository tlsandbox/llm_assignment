#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

UVICORN_BIN="${ROOT_DIR}/.venv/bin/uvicorn"
if [[ ! -x "${UVICORN_BIN}" ]]; then
  echo "Missing ${UVICORN_BIN}. Create the virtualenv and install dependencies first."
  exit 1
fi

REQUESTED_PORT="${PORT:-8001}"

port_is_busy() {
  local port="$1"
  # Treat any process-bound TCP state on the port as busy (LISTEN/CLOSED/etc)
  # to avoid uvicorn bind races with stale orphaned watcher processes.
  lsof -nP -iTCP:"${port}" >/dev/null 2>&1
}

find_free_port() {
  local port="$1"
  while port_is_busy "${port}"; do
    port=$((port + 1))
  done
  echo "${port}"
}

stale_pids=()
while IFS= read -r pid; do
  if [[ -n "${pid}" ]]; then
    stale_pids+=("${pid}")
  fi
done < <(ps -Ao pid=,command= | awk '/[.]venv\/bin\/uvicorn app[.]api_server:app/ {print $1}')

if [[ ${#stale_pids[@]} -gt 0 ]]; then
  echo "Stopping stale app server process(es): ${stale_pids[*]}"
  kill "${stale_pids[@]}" 2>/dev/null || true
  sleep 1

  stubborn_pids=()
  for pid in "${stale_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      stubborn_pids+=("${pid}")
    fi
  done

  if [[ ${#stubborn_pids[@]} -gt 0 ]]; then
    echo "Force stopping stubborn process(es): ${stubborn_pids[*]}"
    kill -9 "${stubborn_pids[@]}" 2>/dev/null || true
    sleep 1
  fi
fi

PORT_TO_USE="${REQUESTED_PORT}"
if port_is_busy "${PORT_TO_USE}"; then
  PORT_TO_USE="$(find_free_port $((REQUESTED_PORT + 1)))"
  echo "Port ${REQUESTED_PORT} is busy; using ${PORT_TO_USE} instead."
fi

echo "Starting RetailNext API on http://127.0.0.1:${PORT_TO_USE}"
exec "${UVICORN_BIN}" app.api_server:app \
  --app-dir "${ROOT_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT_TO_USE}" \
  --reload \
  --reload-dir app \
  --reload-dir src
