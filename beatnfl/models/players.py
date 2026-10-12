"""In-house player model: gradient-boosted forecasts of DK points and of each stat
used by player props, plus out-of-sample residual pools that turn a point forecast
into an over/under probability.

MODEL_VERSION changes whenever features or targets change so frozen forecasts stay
attributable.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .features import FEATURES, PBP_FEATURES

MODEL_VERSION = "pm-1.2"

# target -> positions trained on
TARGETS = {
    "dk_pts": ["QB", "RB", "WR", "TE"],
    "pass_yds": ["QB"], "pass_att": ["QB"], "pass_cmp": ["QB"], "pass_td": ["QB"],
    "rush_yds": ["QB", "RB", "WR"], "rush_att": ["QB", "RB"],
    "rec": ["RB", "WR", "TE"], "rec_yds": ["RB", "WR", "TE"],
}
CLASS_TARGETS = {"any_td": ["QB", "RB", "WR", "TE"]}
GBM = dict(max_iter=350, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=60, l2_regularization=1.0,
           early_stopping=False, random_state=7)


@dataclass
class PlayerModel:
    regs: dict = field(default_factory=dict)
    clfs: dict = field(default_factory=dict)
    resid: dict = field(default_factory=dict)  # target -> (bucket_edges, list[np.ndarray residuals])
    calib: dict = field(default_factory=dict)  # target -> (a, b) logistic recalibration of P(over)
    trained_through: str = ""
    features: list = field(default_factory=lambda: list(FEATURES))
    use_pbp: bool = False
    version: str = MODEL_VERSION

    def fit(self, feat: pd.DataFrame) -> "PlayerModel":
        # Play-by-play matchup features are opt-in: tested 2022–2026 they made forecasts slightly worse
        # (betting-line implied totals already carry that information). See evidence/matchup_ab_test.json.
        if getattr(self, "use_pbp", False) and all(c in feat for c in PBP_FEATURES):
            self.features = list(FEATURES) + list(PBP_FEATURES)
        else:
            self.features = list(FEATURES)
        FEATURES_ = self.features
        f = feat[feat.n_prior >= 1]
        for t, poss in TARGETS.items():
            d = f[f.position.isin(poss) & f[t].notna()]
            self.regs[t] = HistGradientBoostingRegressor(loss="squared_error", **GBM).fit(d[FEATURES_], d[t])
        for t, poss in CLASS_TARGETS.items():
            d = f[f.position.isin(poss) & f[t].notna()]
            self.clfs[t] = HistGradientBoostingClassifier(**GBM).fit(d[FEATURES_], d[t].astype(int))
        last = f.sort_values(["season", "week"]).iloc[-1]
        self.trained_through = f"{int(last.season)} wk {int(last.week)}"
        return self

    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=feat.index)
        for t, poss in TARGETS.items():
            m = feat.position.isin(poss)
            out[f"p_{t}"] = np.nan
            if m.any():
                out.loc[m, f"p_{t}"] = np.clip(self.regs[t].predict(feat.loc[m, self._cols(feat)]), 0, None)
        for t, poss in CLASS_TARGETS.items():
            m = feat.position.isin(poss)
            out[f"p_{t}"] = np.nan
            if m.any():
                out.loc[m, f"p_{t}"] = self.clfs[t].predict_proba(feat.loc[m, self._cols(feat)])[:, 1]
        return out

    def _cols(self, feat: pd.DataFrame) -> list:
        cols = getattr(self, "features", None) or list(FEATURES)
        missing = [c for c in cols if c not in feat]
        if missing:
            raise ValueError(f"Model needs features that are missing: {missing[:5]}")
        return cols

    # ---- uncertainty
    def set_residuals(self, preds: pd.DataFrame, actual: pd.DataFrame, n_buckets: int = 8) -> None:
        """Store out-of-sample residuals bucketed by predicted value."""
        for t in TARGETS:
            if f"p_{t}" not in preds or t not in actual:
                continue
            p, a = preds[f"p_{t}"], actual[t]
            m = p.notna() & a.notna()
            if m.sum() < 200:
                continue
            edges = np.unique(np.quantile(p[m], np.linspace(0, 1, n_buckets + 1)))
            b = np.clip(np.searchsorted(edges, p[m], side="right") - 1, 0, len(edges) - 2)
            r = (a[m] - p[m]).to_numpy()
            self.resid[t] = (edges, [r[b == i] for i in range(len(edges) - 1)])

    def prob_over(self, t: str, pred: float, line: float, calibrated: bool = True) -> float | None:
        if t not in self.resid or pred is None or np.isnan(pred) or line is None or np.isnan(line):
            return None
        edges, pools = self.resid[t]
        i = int(np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(pools) - 1))
        sims = np.clip(pred + pools[i], 0, None)
        # half-point lines never push; whole-number lines: count pushes as half
        p = float((sims > line).mean() + 0.5 * (sims == line).mean())
        if calibrated and t in getattr(self, "calib", {}):
            a, b = self.calib[t]
            z = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
            p = float(1 / (1 + np.exp(-(a + b * z))))
        return p

    def fit_calibration(self, preds: pd.DataFrame, actual: pd.DataFrame, line_from: pd.DataFrame) -> None:
        """Recalibrate P(over) against lines that disagree with the model.

        A forecast that disagrees with a line is partly noise, so raw probabilities
        are overconfident. Fit logit(P_true) = a + b·logit(P_raw) on a season the
        residual pools did not see, using a baseline-derived line (EWM rounded to .5)."""
        from sklearn.linear_model import LogisticRegression
        for t in TARGETS:
            if t == "dk_pts" or t not in self.resid or f"e_{t}" not in line_from:
                continue
            m = preds[f"p_{t}"].notna() & actual[t].notna() & line_from[f"e_{t}"].notna() & (line_from[f"e_{t}"] > 0.5)
            if m.sum() < 300:
                continue
            line = np.floor(line_from.loc[m, f"e_{t}"]) + 0.5
            raw = np.array([self.prob_over(t, pr, ln, calibrated=False) for pr, ln in zip(preds.loc[m, f"p_{t}"], line)])
            z = np.log(np.clip(raw, 1e-3, 1 - 1e-3) / (1 - np.clip(raw, 1e-3, 1 - 1e-3))).reshape(-1, 1)
            y = (actual.loc[m, t].to_numpy() > line.to_numpy()).astype(int)
            # slope only: shrinks overconfident probabilities toward 50% without a directional tilt.
            # (An intercept fitted against baseline-derived lines would encode how those synthetic lines
            # were built — mean-based lines on skewed stats — and would bias every real book line toward the under.)
            lr = LogisticRegression(C=10.0, fit_intercept=False).fit(z, y)
            self.calib[t] = (0.0, float(lr.coef_[0][0]))

    def quantiles(self, t: str, pred: float, qs=(0.1, 0.5, 0.9)):
        if t not in self.resid or pred is None or np.isnan(pred):
            return [None] * len(qs)
        edges, pools = self.resid[t]
        i = int(np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(pools) - 1))
        return list(np.quantile(np.clip(pred + pools[i], 0, None), qs))

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "PlayerModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
