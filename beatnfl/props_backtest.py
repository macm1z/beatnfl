"""Props backtest on real sportsbook lines (The Odds API historical data — paid plans).

For one past season:
  1. The model is rebuilt exactly as the live app builds it, but cut off before that season
     (trained on earlier seasons only; error pools and calibration from the two seasons before).
  2. For every game: the props as they stood on the Wednesday evening before the week ("midweek")
     and 5 minutes before kickoff ("close"), from every US book.
  3. The Picks rule is replayed: props where the model differs from the midweek market by 4+ points,
     taken at the best midweek price across books, then graded on the result and on closing-line value.

Cost: 10 credits per market per snapshot per game (empty responses are free), + 1 per weekly event list.
Every response is cached for good, so a run stopped by the credit budget resumes later at no extra cost.
Information only.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import HOME
from .errors import StepError
from .models import features as F
from .models.players import PlayerModel
from .names import norm_name
from .net import odds_quota
from .sources import nflverse as nv
from .sources import odds

EVID = HOME / "evidence"
BT_DIR = HOME / "backtest"
BT_MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds"]
THRESHOLDS = (4, 6, 8, 10, 15)


def load_result(path) -> dict:
    """Stored result, re-summarised from the saved per-prop file when present (so improvements to the
    analysis apply to past runs without spending credits again)."""
    from pathlib import Path
    r = json.loads(Path(path).read_text())
    csv = BT_DIR / f"props_{r.get('season')}.csv"
    if csv.exists():
        try:
            D = pd.read_csv(csv)
            r.update(summarize(D))
            r["props_matched"] = int(len(main_lines(D)))
        except Exception:  # noqa: BLE001
            pass
    return r


def estimate(season: int, markets=None, offline=False) -> dict:
    sched = nv.schedules(offline=offline)
    g = sched[(sched.season == season) & (sched.game_type == "REG")]
    n_m = len(markets or BT_MARKETS)
    weeks = g.week.nunique()
    return {"games": int(len(g)), "weeks": int(weeks), "credits_max": int(len(g) * 2 * n_m * 10 + weeks)}


# ------------------------------------------------------------------ retrospective model
def season_model(season: int, offline=False, progress=lambda m: None) -> tuple[PlayerModel, pd.DataFrame]:
    """Model as the app would have had it before `season`, plus that season's player-game features."""
    BT_DIR.mkdir(parents=True, exist_ok=True)
    stats = nv.weekly_stats(range(2018, season + 1), offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    feat = F.build(F.tidy_stats(stats), tg)
    from .models.players import MODEL_VERSION
    path = BT_DIR / f"model_{season}_{MODEL_VERSION}.pkl"
    if path.exists():
        pm = pickle.loads(path.read_bytes())
    else:
        progress(f"Training the model as it stood before {season} (a few minutes)…")
        hold = PlayerModel().fit(feat[feat.season < season - 2])
        h1 = feat[(feat.season == season - 2) & (feat.n_prior >= 1)]
        h2 = feat[(feat.season == season - 1) & (feat.n_prior >= 1)]
        pm = PlayerModel().fit(feat[feat.season < season])
        pm.set_residuals(hold.predict(h1), h1)
        pm.fit_calibration(hold.predict(h2), h2, h2)
        path.write_bytes(pickle.dumps(pm))
    te = feat[(feat.season == season) & (feat.n_prior >= 2)].copy()
    pr = pm.predict(te)
    for c in pr.columns:
        te[c] = pr[c].values
    return pm, te


# ------------------------------------------------------------------ snapshots
def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def week_windows(sched: pd.DataFrame, season: int) -> list[dict]:
    g = sched[(sched.season == season) & (sched.game_type == "REG")].copy()
    g["kick"] = pd.to_datetime(g.gameday.astype(str) + " " + g.gametime.astype(str)).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    out = []
    for w, d in g.groupby("week"):
        first = d.kick.min()
        wed = (first - pd.Timedelta(days=(first.weekday() - 2) % 7 or 7)).normalize() + pd.Timedelta(hours=19)
        if wed >= first:
            wed -= pd.Timedelta(days=7)
        out.append({"week": int(w), "midweek": wed, "games": d})
    return out


class Budget:
    def __init__(self, reserve: float):
        self.reserve, self.spent = reserve, 0.0

    def ok(self, cost: float) -> bool:
        rem = odds_quota().get("remaining")
        return rem is None or rem - cost >= self.reserve


def _consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Per player/market/line: no-vig P(over) median across books + best price for each side."""
    if df.empty:
        return pd.DataFrame()
    ov = df[df.side == "Over"].drop_duplicates(["book", "market", "player", "line"])
    un = df[df.side == "Under"].drop_duplicates(["book", "market", "player", "line"])
    m = ov.merge(un[["book", "market", "player", "line", "price"]], on=["book", "market", "player", "line"],
                 how="inner", suffixes=("", "_u"))
    if m.empty:
        return pd.DataFrame()
    m["p_over"] = [odds.no_vig(odds.implied_prob(a), odds.implied_prob(b)) for a, b in zip(m.price, m.price_u)]
    m["dec_o"] = m.price.map(_dec)
    m["dec_u"] = m.price_u.map(_dec)
    rows = []
    for (mk, pl, ln), g in m.groupby(["market", "player", "line"]):
        io, iu = g.dec_o.idxmax(), g.dec_u.idxmax()
        rows.append({"market": mk, "stat": g.stat.iloc[0], "player": pl, "line": float(ln), "p_over": float(g.p_over.median()),
                     "books": int(g.book.nunique()), "best_over": float(g.price[io]), "book_over": g.book[io],
                     "best_under": float(g.price_u[iu]), "book_under": g.book[iu]})
    return pd.DataFrame(rows)


def _dec(a) -> float:
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


# ------------------------------------------------------------------ run
def run(key: str, season: int = 2025, markets=None, reserve: float = 4000, offline=False,
        progress=lambda m: None) -> dict:
    markets = list(markets or BT_MARKETS)
    if not key:
        raise StepError("Props backtest", "No Odds API key set.", "", "Paste your Odds API key in Settings.")
    sched = nv.schedules(offline=offline)
    pm, te = season_model(season, offline=offline, progress=progress)
    te["nn"] = te.name.map(norm_name)
    budget = Budget(reserve)
    status, rows, weeks_done = "complete", [], []
    for wk in week_windows(sched, season):
        if not budget.ok(1):
            status = "paused"
            break
        progress(f"{season} week {wk['week']}: midweek and closing prices…")
        try:
            evs = odds.hist_events(key, _iso(wk["midweek"]), offline=offline)
        except StepError as e:
            if "HTTP 401" in (e.detail or "") or "HTTP 403" in (e.detail or ""):
                raise StepError("Props backtest", "The Odds API refused the historical request.",
                                "Historical odds need a paid plan, and the key must be the one on that plan.",
                                "Check the key in Settings.", detail=e.detail) from None
            raise
        games = wk["games"]
        stopped = False
        for ev in evs:
            h = odds.FULL_TO_ABBR.get(ev.get("home_team"), ev.get("home_team"))
            a = odds.FULL_TO_ABBR.get(ev.get("away_team"), ev.get("away_team"))
            g = games[(games.home_team == h) & (games.away_team == a)]
            if g.empty:
                continue
            kick = pd.Timestamp(ev["commence_time"])
            cost = 2 * len(markets) * 10
            if not budget.ok(cost):
                status, stopped = "paused", True
                break
            mid, _ = odds.hist_props(key, ev["id"], _iso(wk["midweek"]), markets, offline=offline)
            close, _ = odds.hist_props(key, ev["id"], _iso(kick - pd.Timedelta(minutes=5)), markets, offline=offline)
            cm, cc = _consensus(mid), _consensus(close)
            if cm.empty:
                continue
            j = cm.merge(cc[["market", "player", "line", "p_over"]].rename(columns={"p_over": "close_p_over"}),
                         on=["market", "player", "line"], how="left")
            j = j.assign(season=season, week=wk["week"], game=f"{a} @ {h}", event_id=ev["id"], commence=ev["commence_time"],
                         home=h, away=a)
            rows.append(j)
        if stopped:
            break
        weeks_done.append(wk["week"])
    if not rows:
        return {"season": season, "status": status, "n": 0,
                "note": "No historical props were returned (check the plan includes historical odds)."}
    P = pd.concat(rows, ignore_index=True)
    # attach the model forecast and the actual result
    P["nn"] = P.player.map(norm_name)
    out = []
    for (wkn, home, away), g in P.groupby(["week", "home", "away"]):
        t = te[(te.week == wkn) & te.team.isin([home, away])].drop_duplicates("nn", keep=False).set_index("nn")
        for r in g.itertuples():
            if r.nn not in t.index:
                continue
            pred = t.at[r.nn, f"p_{r.stat}"] if f"p_{r.stat}" in t else np.nan
            actual = t.at[r.nn, r.stat] if r.stat in t else np.nan
            if pd.isna(pred) or pd.isna(actual):
                continue
            mp = pm.prob_over(r.stat, float(pred), float(r.line))
            if mp is None:
                continue
            out.append({**r._asdict(), "model_pred": float(pred), "model_p_over": float(mp), "actual": float(actual)})
    D = pd.DataFrame(out).drop(columns=["Index"], errors="ignore")
    BT_DIR.mkdir(parents=True, exist_ok=True)
    D.to_csv(BT_DIR / f"props_{season}.csv", index=False)
    res = summarize(D)
    res.update({"season": season, "status": status, "weeks": weeks_done, "markets": markets,
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "props_seen": int(len(P)), "props_matched": int(len(D))})
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / f"props_backtest_{season}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main_lines(D: pd.DataFrame) -> pd.DataFrame:
    """One prop per player-game: the line the most books posted (ties → closest to the median line).
    Alternate lines for the same player are near-copies of one bet, so counting them all overstates the sample."""
    D = D.copy()
    keys = ["event_id", "player", "stat"]
    if not set(keys + ["books", "line"]) <= set(D.columns):
        return D
    D["_med"] = D.groupby(keys).line.transform("median")
    D["_dist"] = (D.line - D._med).abs()
    return D.sort_values(["books", "_dist"], ascending=[False, True]).drop_duplicates(keys).drop(columns=["_med", "_dist"])


def summarize(D: pd.DataFrame) -> dict:
    """Calibration on every prop + the Picks rule replayed at several thresholds (main line per player-game)."""
    if D.empty:
        return {"n": 0}
    D = main_lines(D)
    D["over"] = np.where(D.actual > D.line, 1.0, np.where(D.actual < D.line, 0.0, np.nan))
    d = D[D.over.notna()]
    brier = {"model": float(((d.model_p_over - d.over) ** 2).mean()),
             "market_midweek": float(((d.p_over - d.over) ** 2).mean())}
    dc = d[d.close_p_over.notna()]
    if len(dc):
        brier["market_close"] = float(((dc.close_p_over - dc.over) ** 2).mean())
    picks = []
    D["diff"] = (D.model_p_over - D.p_over) * 100
    for thr in THRESHOLDS:
        s = D[D["diff"].abs() >= thr].copy()
        if s.empty:
            picks.append({"threshold": thr, "n": 0})
            continue
        over = s["diff"] > 0
        s["price"] = np.where(over, s.best_over, s.best_under)
        s["mkt_side"] = np.where(over, s.p_over, 1 - s.p_over)
        s["close_side"] = np.where(over, s.close_p_over, 1 - s.close_p_over)
        s["taken_p"] = s.price.map(odds.implied_prob)
        s["clv"] = (s.close_side - s.taken_p) * 100
        s["moved"] = (s.close_side - s.mkt_side) * 100           # did the market move toward the pick?
        won = np.where(s.actual == s.line, np.nan, np.where(over, s.actual > s.line, s.actual < s.line).astype(float))
        s["units"] = np.where(np.isnan(won), 0.0, np.where(won == 1, s.price.map(_dec) - 1, -1.0))
        s["won"] = won
        g = s[~np.isnan(won)]
        c = s[s.clv.notna()]
        side = {}
        for nm, msk in (("over", over), ("under", ~over)):
            gg = s[msk & ~np.isnan(won)]
            if len(gg):
                side[nm] = {"n": int(len(gg)), "win_rate": round(float(gg.won.mean()) * 100, 1),
                            "roi": round(float(gg.units.mean()) * 100, 1)}
        picks.append({"threshold": thr, "n": int(len(s)), "won": int(np.nansum(won)), "lost": int((won == 0).sum()),
                      "win_rate": round(float(np.nanmean(won)) * 100, 1) if len(g) else None,
                      "units": round(float(s.units.sum()), 1), "roi": round(float(g.units.mean()) * 100, 1) if len(g) else None,
                      "roi_se": round(float(g.units.std() / np.sqrt(len(g))) * 100, 1) if len(g) > 1 else None,
                      "clv_n": int(len(c)), "clv_avg": round(float(c.clv.mean()), 2) if len(c) else None,
                      "moved_toward": round(float((c.moved > 0).mean()) * 100, 1) if len(c) else None,
                      "moved_away": round(float((c.moved < 0).mean()) * 100, 1) if len(c) else None,
                      "beat_close": round(float((c.clv > 0).mean()) * 100, 1) if len(c) else None,
                      "sides": side,
                      "by_stat": {k: {"n": int(len(v)), "roi": round(float(v.units.mean()) * 100, 1)}
                                  for k, v in s.groupby("stat")}})
    # does the model add anything the market doesn't already know? logistic fit of the result on both
    edge = model_information(d)
    return {"n": int(len(D)), "brier": {k: round(v, 4) for k, v in brier.items()}, "picks": picks, "model_info": edge}


def model_information(d: pd.DataFrame) -> dict:
    """Weight of the model's view once the market's price is known (z ≥ 2 ≈ real information)."""
    if len(d) < 200:
        return {}
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    X = np.column_stack([np.ones(len(d)), lg(d.p_over.to_numpy()), lg(d.model_p_over.to_numpy())])
    y = d.over.to_numpy()
    b = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H + 1e-9 * np.eye(3), X.T @ (y - p))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"coef": round(float(b[2]), 3), "se": round(float(se[2]), 3), "z": round(float(b[2] / se[2]), 2), "n": int(len(d))}
