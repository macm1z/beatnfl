"""FantasyPros API (public v2) — weekly projections, read-only.

Needs a production key (FantasyPros MVP/HOF members activate one at
secure.fantasypros.com/api-keys/request). A free key returns only sample rows; that is
detected from the response's own `tier` / `public_api_limited` / `count` fields and
reported, never silently used as if complete.

FantasyPros publishes stat lines plus Standard/Half/PPR points — not DraftKings points —
so DK points are computed here from the projected stats with BeatNFL's DK scoring
(full PPR, -1 INT, -1 fumble, probability-weighted 300/100-yard bonuses, DST tiers).
"""

from __future__ import annotations

import pandas as pd

from ..errors import StepError
from ..names import loose_key, match_key, team as norm_team
from ..net import fetch_json
from ..scoring import dk_dst_from_stat_line, dk_from_stat_line

BASE = "https://api.fantasypros.com/public/v2/json"
STEP = "FantasyPros projections"
POSITIONS = ("QB", "RB", "WR", "TE", "DST")

# FantasyPros stat keys → the Sleeper-style keys BeatNFL's scoring uses. Several spellings are
# accepted for each so a renamed key degrades gracefully instead of zeroing a stat.
OFF_KEYS = {
    "pass_yd": ("pass_yds", "pass_yd", "passing_yds"),
    "pass_td": ("pass_tds", "pass_td", "passing_tds"),
    "pass_int": ("pass_ints", "pass_int", "ints", "interceptions"),
    "rush_yd": ("rush_yds", "rush_yd"),
    "rush_td": ("rush_tds", "rush_td"),
    "rec": ("rec_rec", "rec", "receptions"),
    "rec_yd": ("rec_yds", "rec_yd"),
    "rec_td": ("rec_tds", "rec_td"),
    "fum_lost": ("fumbles", "fumbles_lost", "fum_lost"),
    "rush_2pt": ("2pt_tds", "two_pts"),
    "ret_td": ("ret_tds",),
}
# Stat lines in BeatNFL's own stat names (used to compare with / blend into the model's stat forecasts)
STAT_KEYS = {
    "pass_att": ("pass_att", "pass_attempts"), "pass_cmp": ("pass_cmp", "pass_completions"),
    "pass_yds": ("pass_yds", "pass_yd"), "pass_td": ("pass_tds", "pass_td"),
    "rush_att": ("rush_att", "rush_attempts"), "rush_yds": ("rush_yds", "rush_yd"),
    "rec": ("rec_rec", "rec", "receptions"), "rec_yds": ("rec_yds", "rec_yd"),
}
DST_KEYS = {"sack": ("def_sack",), "int": ("def_int",), "fum_rec": ("def_fr",), "def_td": ("def_td",),
            "def_st_td": ("def_retd", "def_st_td"), "safe": ("def_safety",), "pts_allow": ("def_pa",)}


def _pick(stats: dict, names) -> float:
    for n in names:
        v = stats.get(n)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def dk_points(pos: str, stats: dict) -> float | None:
    """DraftKings points from one FantasyPros projected stat line."""
    if pos == "DST":
        line = {k: _pick(stats, v) for k, v in DST_KEYS.items()}
        return round(dk_dst_from_stat_line(line), 2) if any(line.values()) else None
    line = {k: _pick(stats, v) for k, v in OFF_KEYS.items()}
    needed = ("pass_yd",) if pos == "QB" else ("rush_yd", "rec", "rec_yd")
    if not any(line[k] for k in needed):
        # no usable stat line — fall back to their PPR total (close to DK, minus yardage bonuses)
        ppr = stats.get("points_ppr")
        return round(float(ppr), 2) if ppr is not None else None
    return round(dk_from_stat_line(line) + 6 * line["ret_td"], 2)


def _need_key(key: str):
    if not key:
        raise StepError(STEP, "No FantasyPros API key set.", "", "Paste your key in Settings (sidebar) and click Save settings.")


def projections(key: str, season: int, week: int, offline=False) -> tuple[pd.DataFrame, dict]:
    """One row per projected player for the week, with DK points. Returns (df, info)."""
    _need_key(key)
    rows, info = [], {"tier": None, "limited": False, "short": []}
    for pos in POSITIONS:
        try:
            data, meta = fetch_json(f"{BASE}/nfl/{season}/projections", params={"position": pos, "week": week},
                                    headers={"x-api-key": key}, step=STEP, max_age_h=3, offline=offline)
        except StepError as e:
            if "HTTP 401" in (e.detail or "") or "HTTP 403" in (e.detail or ""):
                raise StepError(STEP, "FantasyPros refused the API key.",
                                "The key is wrong, expired, or not activated for production use.",
                                "Check the key at secure.fantasypros.com/api-keys, paste it again in Settings, then Refresh.",
                                detail=e.detail) from None
            raise
        info["tier"] = data.get("tier") or info["tier"]
        info["limited"] = info["limited"] or bool(data.get("public_api_limited")) and data.get("tier") != "premium"
        players = data.get("players") or []
        try:
            if int(data.get("count") or 0) > len(players):
                info["short"].append(f"{pos} {len(players)}/{int(data['count'])}")
        except (TypeError, ValueError):
            pass
        for p in players:
            st = p.get("stats") or {}
            row = {"name": p.get("name", ""), "pos": p.get("position_id") or pos,
                   "team": norm_team(p.get("team_id") or ""), "fpid": p.get("fpid"),
                   "proj_paid": dk_points(pos, st), "points_ppr": st.get("points_ppr")}
            if pos != "DST":
                for k, names in STAT_KEYS.items():
                    v = next((st[n] for n in names if st.get(n) is not None), None)
                    row[f"fp_{k}"] = float(v) if v is not None else None
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise StepError(STEP, f"FantasyPros returned no projections for {season} week {week}.",
                        "Projections usually appear early in the week.", "Try again later; other sources still cover the slate.")
    df = df[df.proj_paid.notna()].copy()
    if not df.empty:
        dst = df.pos == "DST"
        # DST names come back as "Denver Broncos" — match DraftKings' DST naming by team instead
        df.loc[dst, "name"] = df.loc[dst, "team"]
    df["key"] = [match_key(n, p, t) for n, p, t in zip(df.name, df.pos, df.team)]
    df["lkey"] = [loose_key(n, p) for n, p in zip(df.name, df.pos)]
    return df.reset_index(drop=True), info
