"""Command line.

  python -m beatnfl app          open the app in your browser (same as Start BeatNFL)
  python -m beatnfl run          whole week headless: slate → projections → 20 lineups → exports
  python -m beatnfl doctor       test every data source from this computer and explain failures
  python -m beatnfl backtest     rerun the historical tests (writes ~/BeatNFL/evidence/)
  python -m beatnfl grade        grade frozen pregame projections and stored props against results
  python -m beatnfl midweek      build and log this week's props picks list (research only)
  python -m beatnfl close        record closing prices for logged picks whose games start soon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_run(a):
    from . import pipeline
    from .config import Settings
    from .optimizer import OptSettings
    s = Settings.load()
    r = pipeline.run_week(s, salary_file=Path(a.salary_file) if a.salary_file else None, draft_group_id=a.dg,
                          paid_file=Path(a.paid) if a.paid else None, use_model=not a.no_model, progress=print)
    print()
    for st in r.steps:
        print(st.as_text())
    if r.pool is None or r.pool.proj.notna().sum() == 0:
        sys.exit(1)
    o = OptSettings(n_lineups=a.n or s.n_lineups, min_salary=s.min_salary, max_exposure=s.max_exposure,
                    min_unique=s.min_unique, stack_min=s.stack_min, bring_back=s.bring_back, randomness=s.randomness,
                    no_offense_vs_dst=s.no_offense_vs_dst)
    out = pipeline.build_lineups(r, o, save=True, export_files=True)
    res = out["result"]
    print(f"\nLineups: requested {res.requested}, generated {len(res.lineups)} ({res.status}, {res.elapsed}s)")
    for d in res.diagnostics:
        print("  •", d)
    if out["set_id"]:
        print(f"Saved as lineup set #{out['set_id']}")
    for f in out["files"]:
        print("Wrote", f)


def cmd_doctor(a):
    from . import doctor
    ok = doctor.run(print)
    sys.exit(0 if ok else 1)


def cmd_backtest(a):
    from . import backtest
    o = backtest.run(progress=print)
    print("Evidence written to", o["path"])
    if a.sleeper:
        c = backtest.sleeper_compare(progress=print)
        print("Sleeper vs model:", json.dumps(c, indent=1))


def cmd_grade(a):
    from . import backtest, research
    g = backtest.grade_frozen()
    print("Frozen projection runs:" if not g.empty else "No pregame projection runs with final results yet.")
    if not g.empty:
        print(g.to_string(index=False))
    print("Props:", research.grade_props())
    w = backtest.suggest_weights(g)
    if w:
        print("Suggested blend weights from graded weeks:", w, "(set them in Settings)")


def cmd_midweek(a):
    from . import research, schedule, storage
    from .config import Settings
    from .errors import StepError
    try:
        picks, info = research.build_midweek(Settings.load(), progress=print)
    except StepError as e:
        print(e.as_text())
        storage.log_status("Midweek picks", "fail", e.what)
        if a.notify:
            schedule.notify("BeatNFL midweek check", "Didn't run: " + e.what)
        sys.exit(1)
    msg = (f"{info['n']} props flagged for {info['season']} week {info['week']} ({info['new']} new; {info['fp_note']})."
           if info["n"] else f"No props met the bar for week {info['week']} yet.")
    print(msg)
    if len(picks):
        print(research.picks_view(picks).head(15).to_string(index=False))
    storage.log_status("Midweek picks", "ok", msg)
    if a.notify:
        top = "; ".join(f"{r.player} {r.side} {r.line:g}" for r in picks.head(3).itertuples()) if len(picks) else ""
        schedule.notify("BeatNFL midweek check", msg + (f" Top: {top}" if top else "") + " Open BeatNFL → Picks.")


def cmd_close(a):
    from . import research, storage
    from .config import Settings
    from .errors import StepError
    s = Settings.load()
    try:
        r = research.capture_close(s.odds_api_key, within_h=a.hours, progress=print)
    except StepError as e:
        print(e.as_text())
        storage.log_status("Closing prices", "fail", e.what)
        sys.exit(1)
    msg = f"Closing prices recorded for {r['games']} game(s) ({r['credits']} Odds API credits)."
    print(msg)
    storage.log_status("Closing prices", "ok", msg)


def cmd_update(a):
    from . import update
    from .config import Settings
    s = Settings.load()
    res = update.apply(a.repo or s.update_repo, s.github_token, progress=print)
    print(res["note"])


def cmd_app(a):
    import subprocess
    # Skip Streamlit's one-time "enter your email" prompt (leave blank = no emails).
    cred = Path.home() / ".streamlit" / "credentials.toml"
    if not cred.exists():
        cred.parent.mkdir(parents=True, exist_ok=True)
        cred.write_text('[general]\nemail = ""\n')
    app = Path(__file__).resolve().parent.parent / "app.py"
    subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless=false",
                     "--browser.gatherUsageStats=false"])


def main():
    p = argparse.ArgumentParser(prog="beatnfl")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--salary-file")
    r.add_argument("--dg", help="DraftKings draft group id (default: main slate)")
    r.add_argument("--paid", help="paid projections CSV (default: newest in inbox)")
    r.add_argument("-n", type=int)
    r.add_argument("--no-model", action="store_true")
    sub.add_parser("doctor")
    b = sub.add_parser("backtest")
    b.add_argument("--sleeper", action="store_true", help="also compare against Sleeper's historical projections")
    sub.add_parser("grade")
    sub.add_parser("app")
    mw = sub.add_parser("midweek")
    mw.add_argument("--notify", action="store_true")
    cl = sub.add_parser("close")
    cl.add_argument("--notify", action="store_true")
    cl.add_argument("--hours", type=float, default=4.0, help="games starting within this many hours")
    u = sub.add_parser("update")
    u.add_argument("repo", nargs="?", help="owner/name (default: the one in Settings)")
    a = p.parse_args()
    {"run": cmd_run, "doctor": cmd_doctor, "backtest": cmd_backtest, "grade": cmd_grade, "app": cmd_app, "update": cmd_update,
     "midweek": cmd_midweek, "close": cmd_close}.get(
        a.cmd or "app", cmd_app)(a)


if __name__ == "__main__":
    main()
