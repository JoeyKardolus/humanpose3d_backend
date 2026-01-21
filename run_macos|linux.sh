#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
