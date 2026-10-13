"""Optional automatic checks on a Mac (launchd), so you don't have to remember them.

  Midweek check   Wednesday 7:00 PM — builds the props picks list, logs it, and shows a Mac notification.
  Game-day check  after the inactive lists post (Thu 7:00 PM, Sun 11:50 AM / 3:10 PM / 7:00 PM, Mon 7:00 PM)
                  — re-checks statuses and notifies you if a player in this week's saved lineups is ruled out.
  Closing prices  shortly before each kickoff window (Thu 6:30 PM, Sun 11:45 AM / 3:15 PM / 7:30 PM,
                  Mon 7:30 PM) — records the last prices for games that have logged picks, for grading.

Times are your Mac's local time. Runs only while the Mac is awake (a missed run happens at next wake).
Nothing here places a bet or opens a sportsbook. Turn it off any time from Settings.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

from .config import HOME

APP_DIR = Path(__file__).resolve().parent.parent
AGENTS = Path.home() / "Library" / "LaunchAgents"
JOBS = {
    "com.beatnfl.midweek": (["midweek", "--notify"], [{"Weekday": 3, "Hour": 19, "Minute": 0}]),
    "com.beatnfl.gamecheck": (["gamecheck", "--notify"], [{"Weekday": 4, "Hour": 19, "Minute": 0},
                                                          {"Weekday": 0, "Hour": 11, "Minute": 50},
                                                          {"Weekday": 0, "Hour": 15, "Minute": 10},
                                                          {"Weekday": 0, "Hour": 19, "Minute": 0},
                                                          {"Weekday": 1, "Hour": 19, "Minute": 0}]),
    "com.beatnfl.close": (["close", "--notify"], [{"Weekday": 4, "Hour": 18, "Minute": 30},
                                                  {"Weekday": 0, "Hour": 11, "Minute": 45},
                                                  {"Weekday": 0, "Hour": 15, "Minute": 15},
                                                  {"Weekday": 0, "Hour": 19, "Minute": 30},
                                                  {"Weekday": 1, "Hour": 19, "Minute": 30}]),
}


def supported() -> bool:
    return sys.platform == "darwin"


def _python() -> str:
    v = APP_DIR / ".venv" / "bin" / "python"
    return str(v if v.exists() else sys.executable)


def plist(label: str) -> dict:
    args, times = JOBS[label]
    return {"Label": label, "ProgramArguments": [_python(), "-m", "beatnfl", *args],
            "WorkingDirectory": str(APP_DIR), "StartCalendarInterval": times,
            "StandardOutPath": str(HOME / "schedule.log"), "StandardErrorPath": str(HOME / "schedule.log"),
            "RunAtLoad": False}


def installed() -> bool:
    return all((AGENTS / f"{k}.plist").exists() for k in JOBS)


def install(on: bool = True) -> str:
    if not supported():
        return "Automatic checks are available on a Mac only."
    AGENTS.mkdir(parents=True, exist_ok=True)
    for label in JOBS:
        p = AGENTS / f"{label}.plist"
        subprocess.run(["launchctl", "unload", str(p)], capture_output=True)
        if on:
            p.write_bytes(plistlib.dumps(plist(label)))
            r = subprocess.run(["launchctl", "load", str(p)], capture_output=True, text=True)
            if r.returncode != 0:
                return f"Couldn't schedule {label}: {r.stderr.strip()[:200]}"
        else:
            p.unlink(missing_ok=True)
    return ("On: midweek check Wednesday 7:00 PM; game-day lineup check after inactives post; closing prices before "
            "Thursday, Sunday and Monday games."
            if on else "Automatic checks are off.")


def notify(title: str, text: str):
    """Mac notification (no-op elsewhere)."""
    if not supported():
        return
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{esc(text)}" with title "{esc(title)}"'],
                   capture_output=True)
