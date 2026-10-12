#!/bin/bash
# Creates ~/Applications/BeatNFL.app, a double-click launcher for this folder's launch.sh.
# Built on your Mac, so macOS doesn't treat it as a downloaded app.
[ "$(uname)" = "Darwin" ] || exit 0
DIR="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$DIR/launch.sh"
mkdir -p "$HOME/Applications"
TMP="$(mktemp -t beatnfl).applescript"
cat > "$TMP" <<APPLESCRIPT
do shell script "/bin/bash " & quoted form of "$DIR/launch.sh"
APPLESCRIPT
rm -rf "$HOME/Applications/BeatNFL.app"
osacompile -o "$HOME/Applications/BeatNFL.app" "$TMP" && rm -f "$TMP"
echo "Created ~/Applications/BeatNFL.app — double-click it (or drag it to the Dock) to open BeatNFL."
