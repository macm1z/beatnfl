#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then echo "Run 'Install BeatNFL.command' first."; read -r; exit 1; fi
exec .venv/bin/python -m beatnfl app
