#!/bin/bash
# Starts BeatNFL in the background and opens it in your browser (same as the BeatNFL app icon).
cd "$(dirname "$0")" || exit 1
exec /bin/bash ./launch.sh
