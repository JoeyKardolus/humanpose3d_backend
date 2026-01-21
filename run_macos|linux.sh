#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${TERM-}" ] || [ ! -t 1 ]; then
  script_path="$SCRIPT_DIR/run_macos|linux.sh"
  escaped_script="$(printf '%q' "$script_path")"
  escaped_args="$(printf ' %q' "$@")"
  cmd="cd $(printf '%q' "$SCRIPT_DIR"); $escaped_script$escaped_args"

  if command -v osascript >/dev/null 2>&1; then
    osascript -e "tell application \"Terminal\" to do script \"$cmd\"" \
      -e "tell application \"Terminal\" to activate"
    exit 0
  fi

  if command -v x-terminal-emulator >/dev/null 2>&1; then
    exec x-terminal-emulator -e bash -lc "$cmd"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
  else
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

uv python install 3.12
uv sync

if [ "$#" -eq 0 ]; then
  echo "Starting server at http://127.0.0.1:8000/"
  uv run python manage.py runserver
else
  uv run python manage.py "$@"
fi
