"""Today — the one screen to open (on the phone or the Mac): what's next, your lineup and its status,
this week's prop picks still to play, and whether the automatic jobs are healthy.

Built from what is already saved (the morning refresh, your saved lineups, the logged picks), so it opens
instantly and needs no Refresh first."""

from __future__ import annotations

import pandas as pd

from . import export, research, storage
from .errors import StepError
from .sources import nflverse as nv

STATUS_TXT = {"Q": "Questionable", "D": "Doubtful", "O": "Out", "IR": "IR"}
JOBS = ["Morning refresh", "Midweek picks", "Game-day check", "Closing prices", "BettingPros record"]


def _week(offline=False):
    sc = nv.schedules(offline=offline)
    season, week = nv.upcoming_week(sc)
    g = sc[(sc.season == season) & (sc.week == week)]
    kick = research._kick_utc(g) if not g.empty else pd.Series(dtype="datetime64[ns, UTC]")
    now = pd.Timestamp.now(tz="UTC")
    nxt = kick[kick > now].min() if len(kick) else pd.NaT
    return season, week, nxt


def lineup(season: int, week: int) -> dict | None:
    """Lineup #1 of your newest saved set this week, with each player's latest status and projection."""
    sets = storage.list_lineup_sets()
    sets = sets[(sets.season == season) & (sets.week == week)] if not sets.empty else sets
    if sets.empty:
        return None
    sid = int(sets.id.max())
    d = storage.load_lineup_set(sid)
    if not d["lineups"]:
        return None
    _, saved = storage.get_projection_run(d["projection_run_id"])
    lr = storage.latest_run_for_week(season, week)
    cur = lr[1].drop_duplicates("key").set_index("key") if lr else saved.drop_duplicates("key").set_index("key")
    sk = saved.set_index("dk_id")
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    rows = []
    for slot, pid in zip(slots, d["lineups"][0]):
        r = sk.loc[pid]
        c = cur.loc[r.key] if r.key in cur.index else r
        st_ = str(c.get("status", "") or "")
        kick = pd.to_datetime(r.start_utc, utc=True, errors="coerce") if r.start_utc else pd.NaT
        rows.append({"Slot": slot, "Player": r["name"], "Team": r.team, "Opp": r.opp, "Salary": int(r.salary),
                     "Proj": round(float(c.get("proj", r.proj) if pd.notna(c.get("proj", None)) else r.proj), 1),
                     "Status": STATUS_TXT.get(st_, ""),
                     "Kickoff": kick.tz_convert("America/New_York").strftime("%a %-I:%M %p") if pd.notna(kick) else ""})
    t = pd.DataFrame(rows)
    try:
        csv = export.dk_csv_text([d["lineups"][0]], saved)
    except ValueError:
        csv = None
    return {"set_id": sid, "name": d["name"], "table": t, "proj": round(float(t.Proj.sum()), 1),
            "salary": int(t.Salary.sum()), "flags": t[t.Status != ""], "csv": csv,
            "updated": (lr[0].get("created_utc") if lr else None)}


def picks(season: int, week: int, limit: int = 8) -> pd.DataFrame:
    """This week's logged prop picks whose games haven't kicked off — one per prop, strongest first."""
    p = storage.picks()
    if p.empty:
        return p
    p = p[(p.season == season) & (p.week == week)]
    p = p[~research.started(p.commence).values] if "commence" in p else p
    if p.empty:
        return p
    p = p.assign(edge=(p.model_p - p.market_p)).sort_values("edge", ascending=False).drop_duplicates(["player", "stat"])
    kick = pd.to_datetime(p.commence, utc=True, errors="coerce").dt.tz_convert("America/New_York")
    return pd.DataFrame({
        "Player": p.player, "Bet": [f"{sd} {l:g} {research.STAT_NAMES.get(s, s).lower()}" for sd, l, s in zip(p.side, p.line, p.stat)],
        "Odds": p.price.round(), "Book": p.book.map(research.book_name),
        "Model %": (p.model_p * 100).round(0), "Market %": (p.market_p * 100).round(0),
        "Kickoff": [k.strftime("%a %-I:%M %p") if pd.notna(k) else "" for k in kick]}).head(limit)


def health(hours: float = 48) -> list[str]:
    """Problems worth telling you about: jobs that failed recently."""
    ds = storage.data_status()
    if ds.empty:
        return []
    out = []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    for r in ds[ds.source.isin(JOBS)].itertuples():
        t = pd.to_datetime(r.last_attempt_utc, utc=True, errors="coerce")
        if r.status in ("fail", "failed") and pd.notna(t) and t >= cutoff:
            out.append(f"{r.source} failed {t.tz_convert('America/New_York'):%a %-I:%M %p}: {str(r.message)[:140]}")
    return out


def summary(offline=False) -> dict:
    try:
        season, week, nxt = _week(offline)
    except StepError as e:
        return {"error": e.as_text()}
    return {"season": season, "week": week, "next_kick": nxt, "lineup": lineup(season, week),
            "picks": picks(season, week), "health": health()}
