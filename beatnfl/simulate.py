"""Game simulation for single-entry and cash lineups: pick the lineup most likely to cash, not just the one
with the highest average.

1. Each player's score gets a full distribution: the projection is its average, its spread comes from the
   model's floor/ceiling (15th/85th percentile) where available, otherwise a typical spread for the position.
   Scores are skewed (lognormal) — a 15-point projection can't go far below 0 but can go to 40.
2. Players move together the way football does, through shared factors:
     game factor    a shootout lifts both offenses (and sinks both defenses)
     team factor    a QB and his receivers rise and fall together; RBs less so
     DST            moves against the opposing offense
3. Thousands of slates are simulated. Many good candidate lineups are built (the optimizer with some
   randomness); in every simulated slate each candidate's score is compared with the others'. Because the
   candidates are what a sharp field looks like, "beat the middle of them" ≈ cash in a 50/50 or double-up,
   and "finish in their top 10%" ≈ the money in a single-entry tournament.

The output ranks the candidates by that chance and shows each one's average, bad day (10th percentile) and
big day (90th). One week is mostly luck, so the Weekly review tracks whether the simulated pick beats the
highest-average lineup over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimizer

# typical spread of DK points as a share of the projection, by position (when no floor/ceiling is known)
SPREAD = {"QB": 0.42, "RB": 0.58, "WR": 0.66, "TE": 0.72, "DST": 0.85}
TEAM_LOAD = {"QB": 0.70, "WR": 0.52, "TE": 0.45, "RB": 0.38, "DST": 0.0}
GAME_LOAD = 0.30            # offense ↔ game environment
DST_VS_OPP = -0.45          # defense ↔ opposing offense
TARGET = {"cash": 0.50, "se": 0.90}   # cash: beat the middle of the field · SE: reach its top 10%


def marginals(pool: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread for each player."""
    p = pool[["dk_id", "pos", "team", "opp", "game", "proj"]].copy()
    m = p.proj.astype(float).clip(lower=0.1)
    sd = m * p.pos.map(SPREAD).fillna(0.65)
    if {"p_dk_floor", "p_dk_ceiling"} <= set(pool.columns):
        span = (pool.p_dk_ceiling - pool.p_dk_floor).astype(float) / 2.07       # 15th–85th pct = ±1.04 sd
        # the model's range is for its own projection; scale it to the blended projection
        rel = (span / pool.proj_model.astype(float).clip(lower=0.5)) if "proj_model" in pool else span / m
        sd = np.where(span.notna() & (span > 0), (rel * m).clip(lower=0.25 * m, upper=1.2 * m), sd)
    p["mean"], p["sd"] = m.values, np.asarray(sd, dtype=float)
    return p.reset_index(drop=True)


def draws(pool: pd.DataFrame, n: int = 4000, seed: int = 7) -> tuple[np.ndarray, pd.DataFrame]:
    """Simulated DK points: array [n sims × players] (columns follow the returned marginals' order)."""
    mg = marginals(pool)
    rng = np.random.default_rng(seed)
    games = {g: i for i, g in enumerate(sorted(mg.game.unique()))}
    teams = {t: i for i, t in enumerate(sorted(set(mg.team) | set(mg.opp)))}
    zg = rng.standard_normal((n, len(games)))
    zt = rng.standard_normal((n, len(teams)))
    gi = mg.game.map(games).to_numpy()
    ti = mg.team.map(teams).to_numpy()
    oi = mg.opp.map(teams).fillna(0).astype(int).to_numpy()
    is_dst = (mg.pos == "DST").to_numpy()
    a_t = mg.pos.map(TEAM_LOAD).fillna(0.4).to_numpy()
    a_g = np.where(is_dst, -GAME_LOAD * 0.6, GAME_LOAD)
    a_o = np.where(is_dst, DST_VS_OPP, 0.0)
    resid = np.sqrt(np.clip(1 - a_t ** 2 - a_g ** 2 - a_o ** 2, 0.05, 1))
    L = zg[:, gi] * a_g + zt[:, ti] * a_t + zt[:, oi] * a_o + rng.standard_normal((n, len(mg))) * resid
    m, s = mg["mean"].to_numpy(), mg["sd"].to_numpy()
    sig2 = np.log1p((s / m) ** 2)
    mu = np.log(m) - sig2 / 2
    X = np.exp(mu + np.sqrt(sig2) * L)                                  # lognormal: mean m, sd s
    X[:, is_dst] = (m + s * L)[:, is_dst]                               # defenses can go negative
    return X, mg


def candidates(pool: pd.DataFrame, contest: str, k: int = 40, min_salary: int = 49000, locks=None, excludes=None,
               seed: int = 11) -> list[list[str]]:
    """Many good, different lineups for the contest (the best few by projection plus randomized variety)."""
    base = optimizer.preset(contest if contest in ("cash", "se") else "se")
    top = optimizer.build(pool, optimizer.OptSettings(**{**base, "n_lineups": 5}, min_salary=min_salary,
                                                      locks=locks or [], excludes=excludes or []))
    var = optimizer.build(pool, optimizer.OptSettings(**{**base, "n_lineups": k, "randomness": 0.18, "min_unique": 2,
                                                         "max_exposure": 1.0}, min_salary=min_salary, seed=seed,
                                                      locks=locks or [], excludes=excludes or []))
    seen, out = set(), []
    for lu in top.lineups + var.lineups:
        key = frozenset(lu)
        if key not in seen:
            seen.add(key)
            out.append(lu)
    return out


def rank(pool: pd.DataFrame, lineups: list[list[str]], contest: str = "cash", n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Each lineup's chance to cash (beat the middle of the candidates) and to reach their top 10%, plus its
    average and its 10th/90th percentile score. Sorted best first for the contest."""
    if not lineups:
        return pd.DataFrame()
    X, mg = draws(pool, n, seed)
    col = {d: i for i, d in enumerate(mg.dk_id)}
    S = np.stack([X[:, [col[x] for x in lu if x in col]].sum(axis=1) for lu in lineups], axis=1)   # sims × lineups
    mid = np.quantile(S, 0.50, axis=1, keepdims=True)
    top = np.quantile(S, 0.90, axis=1, keepdims=True)
    idx = pool.set_index("dk_id")
    out = pd.DataFrame({
        "lineup": list(range(len(lineups))),
        "projected": [round(float(idx.loc[lu].proj.sum()), 1) for lu in lineups],
        "sim_avg": S.mean(axis=0).round(1),
        "bad_day": np.quantile(S, 0.10, axis=0).round(1),
        "big_day": np.quantile(S, 0.90, axis=0).round(1),
        "p_cash": (100 * (S > mid).mean(axis=0)).round(1),
        "p_top10": (100 * (S >= top).mean(axis=0)).round(1)})
    key = "p_cash" if contest == "cash" else "p_top10"
    out = out.sort_values([key, "projected"], ascending=False).reset_index(drop=True)
    out["ids"] = [lineups[i] for i in out.lineup]
    return out


def best(pool: pd.DataFrame, contest: str, k: int = 40, min_salary: int = 49000, n: int = 4000, locks=None,
         excludes=None) -> tuple[pd.DataFrame, list[list[str]]]:
    """Build candidates, simulate, rank. Returns (ranking table, lineups in ranked order)."""
    lus = candidates(pool, contest, k, min_salary, locks, excludes)
    r = rank(pool, lus, contest, n)
    return r, list(r.ids)
