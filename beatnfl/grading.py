"""Grade a saved lineup set against what actually happened (after the games are final).

For every lineup: projected vs actual DraftKings points. For every player used: projection vs
actual and how many lineups he was in. For context: the best lineup that was possible on that slate
in hindsight (same salary cap and positions), and how the lineups would have ranked by projection.
Actual points come from nflverse (players) and DK DST scoring (defenses); a player with no stat line
did not play and scores 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimizer, storage
from .errors import StepError
from .names import match_key
from .sources import nflverse as nv


def actual_points(pool: pd.DataFrame, season: int, week: int, offline=False) -> pd.Series:
    """DK points scored by each slate player (index dk_id). NaN = that game isn't final yet."""
    stats = nv.weekly_stats([season], offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    wk = stats[stats.week == week]
    off = wk[wk.position.isin(["QB", "RB", "WR", "TE"])]
    pts = {match_key(n, p, t): float(v) for n, p, t, v in zip(off.player_display_name, off.position, off.team, off.dk_pts)}
    d = nv.dst_weekly(stats, tg)
    d = d[(d.season == season) & (d.week == week)]
    pts.update({f"DST|{t}": float(v) for t, v in zip(d.team, d.dk_pts)})
    games = sched[(sched.season == season) & (sched.week == week)]
    final = set(games[games.home_score.notna()].home_team) | set(games[games.home_score.notna()].away_team)
    out = []
    for r in pool.itertuples():
        if r.team not in final:
            out.append(np.nan)
        else:
            out.append(pts.get(r.key, 0.0))      # final game, no stat line → didn't play → 0
    return pd.Series(out, index=pool.dk_id.values, dtype=float)


def grade_set(set_id: int, offline=False) -> dict:
    d = storage.load_lineup_set(int(set_id))
    _, pool = storage.get_projection_run(d["projection_run_id"])
    slate, _ = storage.get_slate(d["slate_id"])
    season, week = slate.get("season"), slate.get("week")
    if not season or not week:
        raise StepError("Grade lineups", "This set's slate has no season/week recorded.", "", "Grade a set saved from a weekly Refresh.")
    act = actual_points(pool, int(season), int(week), offline=offline)
    if act.isna().all():
        raise StepError("Grade lineups", f"No final results yet for {season} week {week}.",
                        "nflverse posts player stats a day or two after the games.", "Try again on Tuesday or Wednesday.")
    idx = pool.set_index("dk_id")
    rows = []
    for i, lu in enumerate(d["lineups"], 1):
        a = act.reindex(lu)
        rows.append({"lineup": i, "proj": round(float(idx.loc[lu].proj.sum()), 1),
                     "actual": round(float(a.sum()), 1) if a.notna().all() else np.nan,
                     "players": [(idx.at[x, "name"], idx.at[x, "pos"], round(float(idx.at[x, "proj"]), 1),
                                  None if pd.isna(a[x]) else round(float(a[x]), 1)) for x in lu]})
    L = pd.DataFrame(rows)
    # players used across the set
    from collections import Counter
    cnt = Counter(x for lu in d["lineups"] for x in lu)
    P = pd.DataFrame([{"player": idx.at[x, "name"], "pos": idx.at[x, "pos"], "team": idx.at[x, "team"],
                       "salary": int(idx.at[x, "salary"]), "proj": round(float(idx.at[x, "proj"]), 1),
                       "actual": act.get(x), "lineups": n} for x, n in cnt.items()])
    P["diff"] = (P.actual - P.proj).round(1)
    # hindsight: best possible lineup on this slate with the real scores (same cap and positions)
    best = None
    hp = pool.copy()
    hp["proj"] = act.reindex(hp.dk_id).values
    hp["out"] = False
    hp = hp[hp.proj.notna()]
    try:
        r = optimizer.build(hp, optimizer.OptSettings(n_lineups=1, randomness=0, stack_min=0, min_salary=0,
                                                      max_exposure=1.0, no_offense_vs_dst=False))
        if r.lineups:
            lu = r.lineups[0]
            hi = hp.set_index("dk_id")
            best = {"actual": round(float(hi.loc[lu].proj.sum()), 1),
                    "players": [(hi.at[x, "name"], hi.at[x, "pos"], round(float(hi.at[x, "proj"]), 1)) for x in lu]}
    except Exception:  # noqa: BLE001
        best = None
    done = L.actual.notna()
    s = {"season": int(season), "week": int(week), "name": d["name"], "n": len(L), "graded": int(done.sum()),
         "avg_actual": round(float(L.actual[done].mean()), 1) if done.any() else None,
         "avg_proj": round(float(L.proj.mean()), 1),
         "best": round(float(L.actual.max()), 1) if done.any() else None,
         "worst": round(float(L.actual.min()), 1) if done.any() else None,
         "beat_proj": int((L.actual[done] > L.proj[done]).sum()),
         "proj_rank_corr": round(float(L[done][["proj", "actual"]].corr(method="spearman").iloc[0, 1]), 2) if done.sum() > 3 else None,
         "hindsight_best": best["actual"] if best else None}
    return {"summary": s, "lineups": L, "players": P.sort_values("lineups", ascending=False), "hindsight": best}
