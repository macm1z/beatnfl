"""BeatNFL as an always-on service on the Mac (launchd), plus phone alerts and phone access.

  Always on    The app server starts when you log in and restarts itself if it ever stops — no window to
               open first, and the scheduled jobs and your phone always find it running.
  Phone access With "phone access" on, the server also listens on the Mac's network address, so a phone on
               your private Tailscale network can open it. A PIN is asked for any visit that isn't from the
               Mac itself.
  Phone alerts Alerts are also pushed to the free ntfy app (ntfy.sh) on a private topic name only you know.
"""

from __future__ import annotations

import os
import plistlib
import secrets
import socket
import subprocess
import sys
from pathlib import Path

import requests

from .config import HOME

APP_DIR = Path(__file__).resolve().parent.parent
AGENTS = Path.home() / "Library" / "LaunchAgents"
LABEL = "com.beatnfl.server"
PORT = 8501
NTFY = "https://ntfy.sh"


def supported() -> bool:
    return sys.platform == "darwin"


def _python() -> str:
    v = APP_DIR / ".venv" / "bin" / "python"
    return str(v if v.exists() else sys.executable)


def plist(remote: bool) -> dict:
    return {"Label": LABEL,
            "ProgramArguments": [_python(), "-m", "streamlit", "run", str(APP_DIR / "app.py"),
                                 "--server.headless=true", f"--server.port={PORT}",
                                 f"--server.address={'0.0.0.0' if remote else 'localhost'}",
                                 "--browser.gatherUsageStats=false"],
            "WorkingDirectory": str(APP_DIR), "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10,
            "StandardOutPath": str(HOME / "app.log"), "StandardErrorPath": str(HOME / "app.log")}


def installed() -> bool:
    return (AGENTS / f"{LABEL}.plist").exists()


def install(on: bool = True, remote: bool = False) -> str:
    """Turn the always-on service on/off. Any app started the old way is stopped so the service can take over
    (the page reconnects by itself a few seconds later)."""
    if not supported():
        return "The always-on service is available on a Mac only."
    from .schedule import blocked_location
    where = blocked_location()
    if on and where:
        return (f"Can't turn on yet: the app folder is in {where}, which macOS doesn't let background services open. "
                "Run Install BeatNFL.command — it moves the folder to your home folder as “BeatNFL App”.")
    AGENTS.mkdir(parents=True, exist_ok=True)
    p = AGENTS / f"{LABEL}.plist"
    subprocess.run(["launchctl", "unload", str(p)], capture_output=True)
    if not on:
        p.unlink(missing_ok=True)
        return "Always-on is off. Open BeatNFL from its icon when you need it."
    p.write_bytes(plistlib.dumps(plist(remote)))
    # stop an app started by the icon (same port), then let launchd start the service
    cmd = (f"sleep 1; pkill -f '[s]treamlit run {APP_DIR}/app.py'; sleep 2; launchctl load '{p}'")
    subprocess.Popen(["/bin/bash", "-c", cmd], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ("Always-on is on" + (" with phone access" if remote else "") +
            ". The page reconnects in a few seconds.")


def running(url: str = f"http://localhost:{PORT}") -> bool:
    try:
        return requests.get(f"{url}/_stcore/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def mac_name() -> str:
    """The Mac's network name (what Tailscale shows it as, lower-case, without .local)."""
    try:
        r = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return socket.gethostname().split(".")[0].lower()


# ---------------------------------------------------------------- phone alerts (ntfy)
def new_topic() -> str:
    """A private topic name — anyone who knows it can read the alerts, so it is long and random."""
    return "beatnfl-" + secrets.token_urlsafe(12).replace("_", "").replace("-", "").lower()[:16]


def push(topic: str, title: str, text: str, url: str | None = None, priority: str = "default") -> bool:
    """Send one alert to the ntfy app. Returns False if it couldn't be sent (never raises)."""
    if not topic:
        return False
    headers = {"Title": title.encode("utf-8").decode("latin-1", "ignore") or "BeatNFL", "Priority": priority,
               "Tags": "football"}
    if url:
        headers["Click"] = url
    try:
        r = requests.post(f"{NTFY}/{topic}", data=text.encode("utf-8"), headers=headers, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def is_local(ip: str | None) -> bool:
    """True for a visit from the Mac itself (Streamlit gives no address for localhost visits)."""
    if not ip:
        return True
    return ip in ("127.0.0.1", "::1", "localhost") or ip.startswith("::ffff:127.")


def check_pin(entered: str, pin: str) -> bool:
    return bool(pin) and secrets.compare_digest(str(entered).strip(), str(pin).strip())


def env_ok() -> bool:
    return os.access(APP_DIR, os.R_OK)
