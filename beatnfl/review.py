"""Weekly review — how everything BeatNFL produced for one week landed, in one place.

  DFS     every saved lineup set for the week: projected vs actual, best/worst, and which projection
          source would have built the best lineups (same settings, real results).
  Props   the logged picks: record and ROI, one per prop (alternate lines counted once), by stat, with CLV.
  Games   the game model vs the closing line: average miss of the margin and total, and how its
          disagreements with the market did against the spread and the total.
  BettingPros  their pre-game ratings (the app's snapshots) graded on the box scores, by star rating.

One week is mostly luck. The review shows where things landed; changes to the model are made only when
the multi-season tests on the Evidence tab agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import grading, research, storage
from .errors import StepError
from .sources import nflverse as nv


def _dfs(season: int, week: int, offline=False) -> dict:
    sets = storage.list_lineup_sets()
    out = {"sets": pd.DataFrame(), "sources": pd.DataFrame(), "note": ""}
    if sets.empty:
        out["note"] = "No saved lineup sets."
        return out
    rows, last = [], None
    for sid in sets.id.astype(int):
        try:
            d = storage.load_lineup_set(int(sid))
            sl, _ = storage.get_slate(d["slate_id"])
        except Exception:  # noqa: BLE001
            continue
        if (sl.get("season"), sl.get("week")) != (season, week):
            continue
        try:
            g = grading.grade_set(int(sid), offline=offline)
        except StepError as e:
            out["note"] = e.what
            continue
        s = g["summary"]
        rows.append({"set": int(sid), "name": s["name"], "lineups": s["n"], "avg projected": s["avg_proj"],
                     "avg actual": s["avg_actual"], "best": s["best"], "worst": s["worst"],
                     "beat projection": s["beat_proj"], "best possible": s["hindsight_best"]})
        last = last or int(sid)
    out["sets"] = pd.DataFrame(rows)
    if last:
        try:
            out["sources"] = grading.compare_sources(last, offline=offline)
        except Exception:  # noqa: BLE001
            pass
    if not rows and not out["note"]:
        out["note"] = f"No lineup sets saved for {season} week {week}."
    return out


def _picks(season: int, week: int, offline=False) -> dict:
    g, _ = research.grade_picks(offline=offline)
    if g.empty:
        return {"summary": {}, "by_stat": pd.DataFrame()}
    g = g[(g.season == season) & (g.week == week) & g.result.isin(["won", "lost", "push"])]
    if g.empty:
        return {"summary": {}, "by_stat": pd.DataFrame()}
    one = g.sort_values("created_utc").drop_duplicates(["player", "stat"])

    def s(x):
        n = len(x)
        return {"picks": n, "won %": round(float(100 * (x.result == "won").mean()), 1), "units": round(float(x.units.sum()), 1),
                "ROI %": round(float(100 * x.units.mean()), 1),
                "± (luck)": round(float(100 * x.units.std(ddof=1) / np.sqrt(n)), 1) if n > 1 else None,
                "avg CLV": round(float(x.clv.mean()), 2) if x.clv.notna().any() else None}
    return {"summary": {"all logged": s(g), "one per prop": s(one)},
            "by_stat": pd.DataFrame([{"stat": research.STAT_NAMES.get(k, k), **s(x)} for k, x in one.groupby("stat")])}


def _games(season: int, week: int, offline=False) -> dict:
    g = research.games_data(season, week, None, offline=offline)          # schedule lines = closing consensus
    sc = nv.schedules(offline=offline)
    s = sc[(sc.season == season) & (sc.week == week)][["home_team", "away_team", "home_score", "away_score"]]
    g = g.drop(columns=[c for c in ("home_score", "away_score", "home_team", "away_team") if c in g])
    g = g.merge(s, left_on=["home", "away"], right_on=["home_team", "away_team"], how="left")
    g = g[g.home_score.notna()]
    if g.empty:
        return {"n": 0}
    margin, tot = g.home_score - g.away_score, g.home_score + g.away_score
    d_sp, d_to = g.model_spread - g.spread_nv, g.model_total - g.total
    ats = np.sign(d_sp) == np.sign(margin - g.spread_nv)
    ou = np.sign(d_to) == np.sign(tot - g.total)
    big_s, big_t = d_sp.abs() >= 3, d_to.abs() >= 3
    return {"n": int(len(g)),
            "margin miss: market": round(float((margin - g.spread_nv).abs().mean()), 2),
            "margin miss: model": round(float((margin - g.model_spread).abs().mean()), 2),
            "total miss: market": round(float((tot - g.total).abs().mean()), 2),
            "total miss: model": round(float((tot - g.model_total).abs().mean()), 2),
            "model side vs spread": f"{int(ats.sum())}-{int((~ats).sum())}",
            "model 3+ pts off spread": f"{int((ats & big_s).sum())}-{int((~ats & big_s).sum())}",
            "model side vs total": f"{int(ou.sum())}-{int((~ou).sum())}",
            "model 3+ pts off total": f"{int((ou & big_t).sum())}-{int((~ou & big_t).sum())}",
            "overs": f"{int((tot > g.total).sum())} of {len(g)}"}


def _bettingpros(season: int, week: int, offline=False) -> dict:
    from . import bp_record
    fs = sorted(bp_record.DIR.glob("snapshots_*.csv"))
    if not fs:
        return {"verdict": "No BettingPros snapshots saved.", "by_stars": pd.DataFrame()}
    S = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    S = S[(S.season == season) & (S.week == week)]
    if S.empty:
        return {"verdict": f"No BettingPros snapshots for week {week}.", "by_stars": pd.DataFrame()}
    S = S.sort_values("fetched_utc").drop_duplicates(["bp_player_id", "market_id", "event_id"], keep="last")
    g = bp_record.grade(bp_record.actuals_from_stats(S, offline=offline))
    return bp_record.summarize(g)


def week_review(season: int, week: int, offline=False, progress=lambda m: None) -> dict:
    out = {"season": season, "week": week}
    for name, fn in (("dfs", _dfs), ("picks", _picks), ("games", _games), ("bettingpros", _bettingpros)):
        progress(f"Reviewing {name}…")
        try:
            out[name] = fn(season, week, offline=offline)
        except StepError as e:
            out[name] = {"error": e.as_text()}
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    sc = nv.schedules(offline=offline)
    wk = sc[(sc.season == season) & (sc.week == week)]
    out["games_final"] = f"{int(wk.home_score.notna().sum())} of {len(wk)} games final"
    return out


def as_text(r: dict) -> str:
    """Plain-text version to copy or paste into a chat."""
    lines = [f"BeatNFL review — {r['season']} week {r['week']} ({r.get('games_final', '')})", ""]
    d = r.get("dfs", {})
    lines.append("DFS")
    if isinstance(d.get("sets"), pd.DataFrame) and not d["sets"].empty:
        lines.append(d["sets"].to_string(index=False))
        if isinstance(d.get("sources"), pd.DataFrame) and not d["sources"].empty:
            lines.append(d["sources"].to_string(index=False))
    else:
        lines.append(d.get("note") or d.get("error", ""))
    p = r.get("picks", {})
    lines += ["", "PROPS PICKS"]
    for k, v in (p.get("summary") or {}).items():
        lines.append(f"{k}: {v}")
    if isinstance(p.get("by_stat"), pd.DataFrame) and not p["by_stat"].empty:
        lines.append(p["by_stat"].to_string(index=False))
    lines += ["", "GAMES (model vs closing line)"]
    lines += [f"{k}: {v}" for k, v in r.get("games", {}).items()]
    b = r.get("bettingpros", {})
    lines += ["", "BETTINGPROS", str(b.get("verdict", b.get("error", "")))]
    if isinstance(b.get("by_stars"), pd.DataFrame) and not b["by_stars"].empty:
        lines.append(b["by_stars"].to_string(index=False))
    return "\n".join(lines)
