"""DST (team defense) DK-points model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DST_FEATURES = ["e_dk", "e_dk_long", "e_sk", "e_to", "e_pa", "opp_e_pf", "opp_e_sk_taken", "opp_implied", "spread",
                "total", "home"]


def build(dst: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """dst: dst_weekly() rows (+ future rows with NaN outcomes). Prior-only features."""
    d = dst.sort_values(["season", "week"]).reset_index(drop=True).copy()
    d["to"] = d.it.fillna(0) + d.fr.fillna(0) if "it" in d else np.nan
    g = d.groupby("team", sort=False)
    sh = lambda c, h=4: g[c].transform(lambda x: x.shift(1).ewm(halflife=h, min_periods=1).mean())
    d["e_dk"], d["e_dk_long"] = sh("dk_pts"), sh("dk_pts", 12)
    d["e_sk"], d["e_to"], d["e_pa"] = sh("sk"), sh("to"), sh("pa")
    # opponent offense (points scored, sacks taken) prior-only
    off = tg[["team", "season", "week", "pf"]].sort_values(["season", "week"]).reset_index(drop=True)
    off["opp_e_pf"] = off.groupby("team")["pf"].transform(lambda x: x.shift(1).ewm(halflife=4, min_periods=1).mean())
    sk_taken = d[["opp", "season", "week", "sk"]].rename(columns={"opp": "team"}).sort_values(["season", "week"])
    sk_taken["opp_e_sk_taken"] = sk_taken.groupby("team")["sk"].transform(lambda x: x.shift(1).ewm(halflife=6, min_periods=1).mean())
    d = d.merge(off[["team", "season", "week", "opp_e_pf"]].rename(columns={"team": "opp"}), on=["opp", "season", "week"], how="left")
    d = d.merge(sk_taken[["team", "season", "week", "opp_e_sk_taken"]].rename(columns={"team": "opp"}),
                on=["opp", "season", "week"], how="left")
    if "home" not in d:
        d = d.merge(tg[["team", "season", "week", "home"]], on=["team", "season", "week"], how="left")
    d["opp_implied"] = d.total - d.implied
    return d


class DSTModel:
    def __init__(self):
        self.m = None
        self.resid_sd = 6.0

    def fit(self, f: pd.DataFrame) -> "DSTModel":
        f = f[f.dk_pts.notna() & f.e_dk.notna()]
        self.m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_leaf_nodes=12,
                                               min_samples_leaf=80, random_state=7).fit(f[DST_FEATURES], f.dk_pts)
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return np.clip(self.m.predict(f[DST_FEATURES]), -2, None)
