"""nflverse open data (GitHub releases): weekly player stats, schedules with
closing lines and results, official injury reports, snap counts.

This is the backbone for the in-house model and all historical testing.
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd

from ..errors import StepError
from ..net import fetch
from ..scoring import dk_dst, dk_offense

REL = "https://github.com/nflverse/nflverse-data/releases/download"
STEP = "Historical stats (nflverse)"


def current_season(today: datetime | None = None) -> int:
    t = today or datetime.now()
    return t.year if t.month >= 6 else t.year - 1


def _age(season: int) -> float:
    return 6 if season >= current_season() else 24 * 30


def _parquet(url: str, season: int, offline: bool) -> pd.DataFrame:
    body, _ = fetch(url, step=STEP, max_age_h=_age(season), timeout=90, offline=offline)
    return pd.read_parquet(io.BytesIO(body))


def weekly_stats(seasons, offline=False) -> pd.DataFrame:
    out = []
    for s in seasons:
        try:
            df = _parquet(f"{REL}/stats_player/stats_player_week_{s}.parquet", s, offline)
        except StepError:
            if s >= current_season():
                raise
            df = _parquet(f"{REL}/player_stats/player_stats_{s}.parquet", s, offline)
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    df = df[df.season_type == "REG"].copy()
    df["position"] = df["position"].replace({"FB": "RB"})
    df["dk_pts"] = dk_offense(df)
    return df


def schedules(offline=False) -> pd.DataFrame:
    body, _ = fetch(f"{REL}/schedules/games.csv", step=STEP, max_age_h=3, timeout=60, offline=offline)
    g = pd.read_csv(io.BytesIO(body), low_memory=False)
    g["away_team"] = g["away_team"].replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
    g["home_team"] = g["home_team"].replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
    return g


def injuries(season: int, offline=False) -> pd.DataFrame:
    return _parquet(f"{REL}/injuries/injuries_{season}.parquet", season, offline)


def snaps(seasons, offline=False) -> pd.DataFrame:
    return pd.concat([_parquet(f"{REL}/snap_counts/snap_counts_{s}.parquet", s, offline) for s in seasons],
                     ignore_index=True)


def team_games(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with closing spread/total and implied team total.

    nflverse spread_line is the home team's expected margin (positive = home favoured).
    """
    g = sched[sched.game_type == "REG"].copy()
    home = pd.DataFrame({"game_id": g.game_id, "season": g.season, "week": g.week, "team": g.home_team,
                         "opp": g.away_team, "home": 1, "pf": g.home_score, "pa": g.away_score,
                         "spread": g.spread_line, "total": g.total_line, "gameday": g.gameday, "gametime": g.gametime})
    away = pd.DataFrame({"game_id": g.game_id, "season": g.season, "week": g.week, "team": g.away_team,
                         "opp": g.home_team, "home": 0, "pf": g.away_score, "pa": g.home_score,
                         "spread": -g.spread_line, "total": g.total_line, "gameday": g.gameday, "gametime": g.gametime})
    t = pd.concat([home, away], ignore_index=True)
    t["implied"] = (t.total + t.spread) / 2
    return t


def dst_weekly(stats: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """DK DST points per team-week, built from defensive player stats + points allowed."""
    c = lambda n: stats[n].fillna(0) if n in stats else 0
    d = stats.assign(
        sk=c("def_sacks"), it=c("def_interceptions"), fr=c("fumble_recovery_opp"), dtd=c("def_tds"),
        sttd=c("special_teams_tds"), sf=c("def_safeties"),
        blk=c("def_fg_blocks") + c("def_punt_blocks"),
    ).groupby(["season", "week", "team"], as_index=False)[["sk", "it", "fr", "dtd", "sttd", "sf", "blk"]].sum()
    d = d.merge(tg[["season", "week", "team", "opp", "pa", "implied", "spread", "total", "game_id"]],
                on=["season", "week", "team"], how="inner")
    d = d[d.pa.notna()]
    # Points allowed for DK excludes points the defense/ST gave up on its own offense's turnovers returned —
    # not available here; opponent final score is the standard approximation.
    d["dk_pts"] = dk_dst(d.sk, d.it, d.fr, d.dtd, d.sttd, d.sf, d.blk, d.pa)
    d["position"] = "DST"
    d["player_id"] = "DST_" + d.team
    d["player_display_name"] = d.team + " DST"
    return d


# ------------------------------------------------------------------ play-by-play team metrics
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "success", "pass", "rush",
            "yards_gained", "wp", "qtr", "pass_oe", "qb_dropback", "sack", "interception", "fumble_lost", "fixed_drive",
            "fixed_drive_result", "yardline_100", "qb_kneel", "qb_spike", "two_point_attempt"]
TEAM_FIX = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA"}


def _team_metrics_one(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game offense and defense efficiency from one season of play-by-play."""
    p = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["pass", "run"]) & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)
            & (pbp.two_point_attempt != 1) & pbp.epa.notna()].copy()
    for c in ("posteam", "defteam"):
        p[c] = p[c].replace(TEAM_FIX)
    p["explosive"] = ((p.rush == 1) & (p.yards_gained >= 10)) | ((p["pass"] == 1) & (p.yards_gained >= 20))
    p["neutral"] = p.wp.between(0.2, 0.8) & (p.qtr <= 3)
    p["turnover"] = (p.interception.fillna(0) + p.fumble_lost.fillna(0)) > 0
    p["is_pass"] = p["pass"] == 1
    p["is_rush"] = p.rush == 1
    p["pass_epa"] = p.epa.where(p.is_pass)
    p["rush_epa"] = p.epa.where(p.is_rush)
    p["neutral_pass"] = p.is_pass.where(p.neutral)
    p["proe"] = p.pass_oe.where(p.neutral)
    p["sack_db"] = p.sack.where(p.qb_dropback == 1)
    # red zone: share of drives reaching the 20 that end in a touchdown
    rz = p[p.yardline_100 <= 20].drop_duplicates(["game_id", "posteam", "fixed_drive"])
    rz = rz.assign(rz_td=(rz.fixed_drive_result == "Touchdown").astype(float))
    rzg = rz.groupby(["game_id", "posteam"]).agg(rz_trips=("rz_td", "size"), rz_tds=("rz_td", "sum")).reset_index()
    agg = dict(plays=("epa", "size"), epa=("epa", "mean"), sr=("success", "mean"), pass_epa=("pass_epa", "mean"),
               rush_epa=("rush_epa", "mean"), explosive=("explosive", "mean"), neutral_pass=("neutral_pass", "mean"),
               proe=("proe", "mean"), sack_rate=("sack_db", "mean"), to_rate=("turnover", "mean"))
    off = p.groupby(["game_id", "season", "week", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    off = off.merge(rzg.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
    de = p.groupby(["game_id", "season", "week", "defteam"]).agg(**agg).reset_index().rename(columns={"defteam": "team"})
    de = de.merge(rzg.rename(columns={"posteam": "opp_off"}), left_on="game_id", right_on="game_id", how="left")
    de = de[de.opp_off != de.team].drop(columns="opp_off")
    off = off.rename(columns={c: f"off_{c}" for c in off.columns if c not in ("game_id", "season", "week", "team")})
    de = de.rename(columns={c: f"def_{c}" for c in de.columns if c not in ("game_id", "season", "week", "team")})
    return off.merge(de, on=["game_id", "season", "week", "team"], how="outer")


def team_metrics(seasons, offline=False, progress=lambda m: None) -> pd.DataFrame:
    """Team-game efficiency table for the given seasons. Each season is computed once and cached."""
    from ..config import CACHE
    out = []
    for s in seasons:
        cache = CACHE / f"team_metrics_{s}.parquet"
        fresh_needed = s >= current_season()
        src_age = None
        if cache.exists() and not fresh_needed:
            out.append(pd.read_parquet(cache))
            continue
        progress(f"Play-by-play {s}…")
        body, meta = fetch(f"{REL}/pbp/play_by_play_{s}.parquet", step=STEP, max_age_h=_age(s), timeout=180, offline=offline)
        import pyarrow.parquet as pq
        tbl = pq.read_table(io.BytesIO(body), columns=PBP_COLS).to_pandas()
        tm = _team_metrics_one(tbl)
        tm.to_parquet(cache, index=False)
        out.append(tm)
    return pd.concat(out, ignore_index=True)
