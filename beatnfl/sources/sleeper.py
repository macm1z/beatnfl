"""Sleeper public API: current week, player status/injuries/depth charts, projections.

Free, no key. Not an official contract — every response is schema-checked and a
failure here degrades the run (warning) rather than stopping it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..errors import StepError
from ..names import match_key, loose_key, pos as norm_pos, team as norm_team
from ..net import fetch_json
from ..scoring import dk_dst_from_stat_line, dk_from_stat_line

BASE = "https://api.sleeper.app"
INJ_STEP, PROJ_STEP = "Injuries & roles (Sleeper)", "Projections (Sleeper)"
INJURY_MAP = {"Out": "O", "IR": "O", "PUP": "O", "Sus": "O", "NA": "O", "COV": "O", "DNR": "O",
              "Doubtful": "D", "Questionable": "Q", "Probable": "P"}


def state(offline=False) -> dict:
    data, _ = fetch_json(f"{BASE}/v1/state/nfl", step="Season/week", max_age_h=2, offline=offline)
    if "week" not in data:
        raise StepError("Season/week", "Could not read the current NFL week.", "Sleeper's state feed changed.",
                        "Set the week manually in the app.")
    return data


def players(offline=False) -> tuple[pd.DataFrame, str]:
    data, meta = fetch_json(f"{BASE}/v1/players/nfl", step=INJ_STEP, max_age_h=3, timeout=60, offline=offline)
    if not isinstance(data, dict) or len(data) < 1000:
        raise StepError(INJ_STEP, "Sleeper's player list was incomplete.", "Feed changed or truncated download.",
                        "Run again; injuries from DraftKings' own status field are still used.")
    rows = []
    for pid, p in data.items():
        position = norm_pos(p.get("position"))
        if position not in {"QB", "RB", "WR", "TE", "DST"}:
            continue
        tm = norm_team(p.get("team") or (pid if position == "DST" else ""))
        if not tm:
            continue
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        news = p.get("news_updated")
        rows.append({
            "sleeper_id": pid, "gsis_id": p.get("gsis_id"), "name": name, "pos": position, "team": tm,
            "injury": INJURY_MAP.get(p.get("injury_status") or "", "") if p.get("injury_status") else "",
            "injury_raw": p.get("injury_status") or "", "injury_body": p.get("injury_body_part") or "",
            "status": p.get("status") or "", "depth_pos": p.get("depth_chart_position") or "",
            "depth_order": p.get("depth_chart_order"),
            "news_utc": datetime.fromtimestamp(news / 1000, tz=timezone.utc).isoformat() if news else None,
        })
    df = pd.DataFrame(rows)
    df["key"] = [match_key(n, p, t) for n, p, t in zip(df.name, df.pos, df.team)]
    df["lkey"] = [loose_key(n, p) for n, p in zip(df.name, df.pos)]
    note = ("STALE " + meta.note) if meta.stale else ("cached" if meta.from_cache else "live")
    return df, note


def projections(season: int, week: int, offline=False) -> tuple[pd.DataFrame, str]:
    """Sleeper weekly projections converted to DK points with our scoring code."""
    url = f"{BASE}/projections/nfl/{season}/{week}"
    data, meta = fetch_json(url, params={"season_type": "regular", "position[]": ["QB", "RB", "WR", "TE", "DEF"]},
                            step=PROJ_STEP, max_age_h=2, offline=offline)
    if not isinstance(data, list):
        raise StepError(PROJ_STEP, "Sleeper projections were not in the expected format.", "Feed changed.",
                        "The run continues with the other projection sources.")
    rows = []
    for r in data:
        pl = r.get("player") or {}
        position = norm_pos(pl.get("position"))
        if position not in {"QB", "RB", "WR", "TE", "DST"}:
            continue
        st = r.get("stats") or {}
        if not st:
            continue
        tm = norm_team(r.get("team") or pl.get("team"))
        name = f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip()
        pts = dk_dst_from_stat_line(st) if position == "DST" else dk_from_stat_line(st)
        rows.append({"sleeper_id": str(r.get("player_id")), "name": name, "pos": position, "team": tm,
                     "proj_sleeper": round(pts, 2), "opp": norm_team(r.get("opponent")),
                     **{f"s_{k}": st.get(k) for k in ("pass_yd", "pass_td", "pass_att", "pass_cmp", "rush_yd", "rush_att",
                                                     "rec", "rec_yd", "rec_tgt", "rush_td", "rec_td")}})
    df = pd.DataFrame(rows)
    if df.empty:
        raise StepError(PROJ_STEP, f"Sleeper has no projections for {season} week {week} yet.",
                        "They usually appear early in the week.", "The run continues with the other sources.")
    df["key"] = [match_key(n, p, t) for n, p, t in zip(df.name, df.pos, df.team)]
    df["lkey"] = [loose_key(n, p) for n, p in zip(df.name, df.pos)]
    note = ("STALE " + meta.note) if meta.stale else ("cached" if meta.from_cache else "live")
    return df, note
