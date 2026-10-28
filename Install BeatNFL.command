#!/bin/bash
# One-time setup for macOS. Safe to run again (it just refreshes).
cd "$(dirname "$0")" || exit 1
set -e
echo "== BeatNFL setup =="
UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ -z "$UV" ]; then
  echo "Installing uv 0.11.32 (manages Python for this app only)…"
  # Version-pinned official installer; it verifies the downloaded uv archive's checksum.
  # Prefer Homebrew? Run 'brew install uv' first and this step is skipped.
  curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi
# A folder that was moved (e.g. out of Downloads) gets a fresh environment — scripts inside it hold the old path.
if [ -d .venv ] && [ "$(cat .venv/.beatnfl_path 2>/dev/null)" != "$PWD" ]; then
  echo "Folder moved — rebuilding the Python environment for the new location…"
  rm -rf .venv
fi
"$UV" venv --python 3.13 .venv --allow-existing
echo "$PWD" > .venv/.beatnfl_path
# Every package (47, including sub-dependencies) is locked to an exact version AND sha256 checksum;
# installation stops if any downloaded file does not match.
"$UV" pip install --python .venv --require-hashes -r requirements.txt
echo; echo "== Self-test =="
.venv/bin/python -m pytest -q tests
echo; echo "== Checking data sources from this computer =="
.venv/bin/python -m beatnfl doctor || true
bash make_mac_app.sh || true
echo; echo "Setup complete. Open BeatNFL from your Applications folder (or Launchpad) — drag it to the Dock to keep it handy."
