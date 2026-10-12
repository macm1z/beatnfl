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
