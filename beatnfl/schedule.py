"""Optional automatic checks on a Mac (launchd), so you don't have to remember them.

  Midweek check   Wednesday 7:00 PM — builds the props picks list, logs it, and shows a Mac notification.
  Game-day check  after the inactive lists post (Thu 7:00 PM, Sun 11:50 AM / 3:10 PM / 7:00 PM, Mon 7:00 PM)
                  — re-checks statuses and notifies you if a player in this week's saved lineups is ruled out.
  BettingPros     every evening 9:15 PM — grows the BettingPros track record with the day's spare
                  allowance; Wed 8 PM and Sun 11:40 AM also save the week's board (pre-game ratings).
  Closing prices  shortly before each kickoff window (Thu 6:30 PM, Sun 11:45 AM / 3:15 PM / 7:30 PM,
                  Mon 7:30 PM) — records the last prices for games that have logged picks, for grading.

Times above are US Eastern (NFL kickoff time); they're converted to your Mac's time zone when the
schedule is installed, and re-converted automatically if you travel to another time zone. Runs only while the Mac is awake (a missed run happens at next wake).
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
    # BettingPros record: every evening after the daily allowance resets (midnight UTC)
    "com.beatnfl.bettingpros": (["bettingpros"], [{"Weekday": d, "Hour": 21, "Minute": 15} for d in range(7)]),
    "com.beatnfl.bpsnapshot": (["bettingpros", "--snapshot"], [{"Weekday": 3, "Hour": 20, "Minute": 0},
                                                              {"Weekday": 0, "Hour": 11, "Minute": 40}]),
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


def to_local(times: list[dict], tz: str | None = None) -> list[dict]:
    """Eastern weekday/hour/minute → the same moment in this Mac's time zone (launchd uses local time)."""
    import pandas as pd
    from .config import local_tz
    tz = tz or local_tz()
    base = pd.Timestamp("2026-10-18", tz="America/New_York")          # a Sunday (launchd Weekday 0)
    out = []
    for t in times:
        et = base + pd.Timedelta(days=t["Weekday"], hours=t["Hour"], minutes=t["Minute"])
        lt = et.tz_convert(tz)
        out.append({"Weekday": (lt.weekday() + 1) % 7, "Hour": lt.hour, "Minute": lt.minute})
    return out


def plist(label: str, tz: str | None = None) -> dict:
    args, times = JOBS[label]
    return {"Label": label, "ProgramArguments": [_python(), "-m", "beatnfl", *args],
            "WorkingDirectory": str(APP_DIR), "StartCalendarInterval": to_local(times, tz),
            "StandardOutPath": str(HOME / "schedule.log"), "StandardErrorPath": str(HOME / "schedule.log"),
            "RunAtLoad": False}


def installed() -> bool:
    return all((AGENTS / f"{k}.plist").exists() for k in JOBS)


TZ_MARK = HOME / "schedule_tz.txt"


def keep_in_sync() -> str | None:
    """If the checks are on and this Mac's time zone changed since they were scheduled (travel),
    reschedule them so they still fire at the right Eastern times. Returns a note when it did."""
    from .config import local_tz
    if not supported():
        return None
    some = any((AGENTS / f"{k}.plist").exists() for k in JOBS)
    if some and not installed():            # checks were on before a new one was added — add it
        install(True)
        return "Automatic checks updated (BettingPros record added)."
    if not installed():
        return None
    tz = local_tz()
    try:
        prev = TZ_MARK.read_text().strip()
    except OSError:
        prev = ""
    if prev == tz:
        return None
    install(True)
    return f"Automatic checks re-timed for your time zone ({tz})."


def install(on: bool = True) -> str:
    if not supported():
        return "Automatic checks are available on a Mac only."
    AGENTS.mkdir(parents=True, exist_ok=True)
    for label in JOBS:
        p = AGENTS / f"{label}.plist"
        subprocess.run(["launchctl", "unload", str(p)], capture_output=True)
        if on:
            p.write_bytes(plistlib.dumps(plist(label)))
            from .config import local_tz
            TZ_MARK.write_text(local_tz())
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
