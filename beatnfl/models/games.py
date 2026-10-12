"""Game model: margin and total forecasts from opponent-adjusted team ratings.

Ratings are updated game by game (Elo-style on points), so every forecast uses only
earlier results. Two outputs are produced for each game:
  model_spread / model_total          the model on its own
  blend_spread / blend_total          market line nudged toward the model by BLEND_W
The backtest reports both against the closing line so you can see whether the model
adds anything the market does not already know.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

HFA = 1.6          # home-field points (league-wide, recent seasons)
K_MARGIN = 0.10    # rating step per point of margin error
K_TOTAL = 0.08
REGRESS = 0.33     # offseason pull toward mean
BLEND_W = 0.25


def run(sched: pd.DataFrame) -> pd.DataFrame:
    """Return every regular/post-season game with pre-game model numbers."""
    g = sched[sched.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])].sort_values(["season", "week", "gameday", "gametime"])
    rating = defaultdict(float)           # power (points vs average)
    off, de = defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)  # scoring above/below league
    lg_pts = 22.0
    season = None
    out = []
    for r in g.itertuples(index=False):
        if r.season != season:
            for d in (rating, off, de):
                for k in list(d):
                    d[k] *= (1 - REGRESS)
            season = r.season
        h, a = r.home_team, r.away_team
        neutral = str(getattr(r, "location", "Home")) == "Neutral"
        hfa = 0 if neutral else HFA
        m_spread = rating[h] - rating[a] + hfa  # home expected margin (nflverse sign convention)
        exp_h = lg_pts + off[h] + de[a] + hfa / 2
        exp_a = lg_pts + off[a] + de[h] - hfa / 2
        m_total = exp_h + exp_a
        out.append((r.game_id, r.season, r.week, r.game_type, h, a, r.spread_line, r.total_line,
                    r.home_score, r.away_score, round(m_spread, 2), round(m_total, 2)))
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = r.home_score - r.away_score
            err = np.clip(margin - m_spread, -21, 21)
            rating[h] += K_MARGIN * err
            rating[a] -= K_MARGIN * err
            eh, ea = r.home_score - exp_h, r.away_score - exp_a
            off[h] += K_TOTAL * eh
            de[a] += K_TOTAL * eh
            off[a] += K_TOTAL * ea
            de[h] += K_TOTAL * ea
            lg_pts += 0.002 * ((r.home_score + r.away_score) / 2 - lg_pts)
    df = pd.DataFrame(out, columns=["game_id", "season", "week", "game_type", "home", "away", "spread_line",
                                    "total_line", "home_score", "away_score", "model_spread", "model_total"])
    df["blend_spread"] = df.spread_line + BLEND_W * (df.model_spread - df.spread_line)
    df["blend_total"] = df.total_line + BLEND_W * (df.model_total - df.total_line)
    df["margin"] = df.home_score - df.away_score
    df["points"] = df.home_score + df.away_score
    return df


def cover_prob(model_margin: float, line: float, sd: float = 13.4) -> float:
    """P(home margin > line) assuming normal error around the model margin."""
    from math import erf, sqrt
    return 0.5 * (1 - erf((line - model_margin) / (sd * sqrt(2))))
