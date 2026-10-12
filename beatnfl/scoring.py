"""DraftKings NFL Classic scoring (rule version dk_nfl_classic_2024).

Player stats are forecast first; fantasy points are computed here, once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE_VERSION = "dk_nfl_classic_2024"


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].fillna(0) if name in df else pd.Series(0.0, index=df.index)


def dk_offense(df: pd.DataFrame) -> pd.Series:
    """DK points from nflverse-style weekly player stat columns."""
    py, ry, rcy = _col(df, "passing_yards"), _col(df, "rushing_yards"), _col(df, "receiving_yards")
    pts = (
        0.04 * py + 4 * _col(df, "passing_tds") - 1 * _col(df, "passing_interceptions")
        + 3 * (py >= 300)
        + 0.1 * ry + 6 * _col(df, "rushing_tds") + 3 * (ry >= 100)
        + 1 * _col(df, "receptions") + 0.1 * rcy + 6 * _col(df, "receiving_tds") + 3 * (rcy >= 100)
        + 6 * _col(df, "special_teams_tds") + 6 * _col(df, "fumble_recovery_tds")
        - 1 * (_col(df, "rushing_fumbles_lost") + _col(df, "receiving_fumbles_lost") + _col(df, "sack_fumbles_lost"))
        + 2 * (_col(df, "passing_2pt_conversions") + _col(df, "rushing_2pt_conversions") + _col(df, "receiving_2pt_conversions"))
    )
    return pts.astype(float)


def pa_points(points_allowed) -> np.ndarray:
    pa = np.asarray(points_allowed, dtype=float)
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10, 7, 4, 1, 0, -1], -4).astype(float)


def dk_dst(sacks, ints, fum_rec, def_tds, st_tds, safeties, blocks, points_allowed) -> np.ndarray:
    return (np.asarray(sacks) * 1 + np.asarray(ints) * 2 + np.asarray(fum_rec) * 2
            + (np.asarray(def_tds) + np.asarray(st_tds)) * 6 + np.asarray(safeties) * 2
            + np.asarray(blocks) * 2 + pa_points(points_allowed))


def dk_from_stat_line(s: dict) -> float:
    """DK points from a projected stat line using Sleeper stat keys.

    Yardage bonuses are applied to projections as probability-weighted values
    when a mean is given (P(yds>=threshold) under a normal approx), not as hard steps.
    """
    from math import erf, sqrt

    def p_over(mean, sd, thr):
        if mean <= 0:
            return 0.0
        return 0.5 * (1 - erf((thr - mean) / (sd * sqrt(2))))

    g = lambda k: float(s.get(k) or 0)
    py, ry, rcy = g("pass_yd"), g("rush_yd"), g("rec_yd")
    return (0.04 * py + 4 * g("pass_td") - g("pass_int") + 3 * p_over(py, 70, 300)
            + 0.1 * ry + 6 * g("rush_td") + 3 * p_over(ry, 30, 100)
            + g("rec") + 0.1 * rcy + 6 * g("rec_td") + 3 * p_over(rcy, 28, 100)
            - g("fum_lost") + 2 * (g("pass_2pt") + g("rush_2pt") + g("rec_2pt")))


def dk_dst_from_stat_line(s: dict) -> float:
    g = lambda k: float(s.get(k) or 0)
    pa = g("pts_allow")
    # Expected PA tier points: integrate over a normal around the mean (sd≈9.5 pts).
    grid = np.arange(0, 60)
    w = np.exp(-0.5 * ((grid - pa) / 9.5) ** 2)
    w /= w.sum()
    pa_exp = float((pa_points(grid) * w).sum()) if pa > 0 else 0.0
    return (g("sack") + 2 * g("int") + 2 * g("fum_rec") + 6 * (g("def_td") + g("def_st_td"))
            + 2 * g("safe") + 2 * g("blk_kick") + pa_exp)
