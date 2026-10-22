"""Historical testing — the evidence behind every number the app shows.

Retrospective tests (reconstructed forecasts; clearly labeled as such):
  players   DK-point forecasts vs two simple baselines, by position and season
  picks     decision test: top-N per position each week chosen by each method, actual points scored
  dst       DST forecasts vs baseline
  props     stat forecasts vs baseline + over/under probability calibration
  games     model and market-blend vs the closing line, and against-the-spread record

Prospective grading (true pregame forecasts frozen by the app before kickoff):
  grade_frozen  each frozen projection run vs final nflverse stats, per source

Leakage rule: the model for season S is trained only on seasons < S, and every
feature uses games strictly before the forecast game.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .config import HOME
from .models import dst as dstm
from .models import features as F
from .models import games as gm
from .models.players import PlayerModel
from .sources import nflverse as nv

EVID = HOME / "evidence"
TOPN = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
PROP_STATS = {"pass_yds": ["QB"], "pass_att": ["QB"], "pass_cmp": ["QB"], "rush_yds": ["RB"], "rush_att": ["RB"],
              "rec": ["WR", "TE", "RB"], "rec_yds": ["WR", "TE", "RB"]}


def _spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def run(test_seasons=(2022, 2023, 2024, 2025, 2026), first=2018, offline=False, progress=print) -> dict:
    progress("Loading history…")
    cur = nv.current_season()
    stats = nv.weekly_stats(range(first, cur + 1), offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    progress("Building features…")
    feat = F.build(F.tidy_stats(stats), tg)
    feat = feat[feat.n_prior >= 1]
    preds = []
    for S in test_seasons:
        progress(f"Season {S}: training on {first}–{S - 1}, forecasting {S}…")
        m = PlayerModel().fit(feat[feat.season < S])
        te = feat[feat.season == S]
        p = m.predict(te)
        preds.append(pd.concat([te.reset_index(drop=True), p.reset_index(drop=True)], axis=1))
    P = pd.concat(preds, ignore_index=True)
    P["naive"] = P.std_mean.fillna(P.prev_season_mean).fillna(P.e_dk_pts)
    P["ewm"] = P.e_dk_pts
    P["model"] = P.p_dk_pts

    out = {"generated": datetime.now().isoformat(timespec="minutes"), "label": "retrospective reconstruction",
           "train_rule": "season S forecast with a model trained on seasons before S; features use prior games only",
           "seasons": list(test_seasons)}
    out["players"] = _player_metrics(P)
    out["picks"] = _picks(P)
    out["props"] = _props(P)
    progress("DST…")
    out["dst"] = _dst(stats, tg, test_seasons)
    progress("Games vs closing line…")
    out["games"] = _games(sched, min(test_seasons) - 7)
    EVID.mkdir(parents=True, exist_ok=True)
    path = EVID / f"backtest_{datetime.now():%Y%m%d_%H%M}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    out["path"] = str(path)
    return out


def _player_metrics(P: pd.DataFrame) -> list[dict]:
    rows = []
    for (s, pos), d in P.groupby(["season", "position"]):
        r = {"season": int(s), "pos": pos, "n": len(d)}
        for c in ("model", "naive", "ewm"):
            e = d[c] - d.dk_pts
            r[f"mae_{c}"] = round(float(e.abs().mean()), 3)
            r[f"rmse_{c}"] = round(float(np.sqrt((e ** 2).mean())), 3)
            r[f"bias_{c}"] = round(float(e.mean()), 3)
        rows.append(r)
    for pos, d in P.groupby("position"):
        r = {"season": "all", "pos": pos, "n": len(d)}
        for c in ("model", "naive", "ewm"):
            e = d[c] - d.dk_pts
            r[f"mae_{c}"] = round(float(e.abs().mean()), 3)
            r[f"rmse_{c}"] = round(float(np.sqrt((e ** 2).mean())), 3)
            r[f"bias_{c}"] = round(float(e.mean()), 3)
        rows.append(r)
    return rows


def _picks(P: pd.DataFrame) -> list[dict]:
    """Each week, take the top-N players per position by each method; compare actual DK points.
    Also rank correlation within the fantasy-relevant pool (top 2N by any method)."""
    rows = []
    for pos, n in TOPN.items():
        d = P[P.position == pos]
        acc = {c: [] for c in ("model", "naive", "ewm", "oracle")}
        rho = {c: [] for c in ("model", "naive", "ewm")}
        for (s, w), g in d.groupby(["season", "week"]):
            if len(g) < 2 * n:
                continue
            for c in ("model", "naive", "ewm"):
                acc[c].append(g.nlargest(n, c).dk_pts.mean())
            acc["oracle"].append(g.nlargest(n, "dk_pts").dk_pts.mean())
            pool = g[g.index.isin(set(g.nlargest(2 * n, "model").index) | set(g.nlargest(2 * n, "naive").index))]
            for c in rho:
                rho[c].append(_spearman(pool[c], pool.dk_pts))
        rows.append({"pos": pos, "top_n": n, "weeks": len(acc["model"]),
                     **{f"avg_pts_{c}": round(float(np.mean(v)), 2) for c, v in acc.items()},
                     **{f"rank_corr_{c}": round(float(np.nanmean(v)), 3) for c, v in rho.items()},
                     "model_beats_naive_weeks": f"{np.mean(np.array(acc['model']) > np.array(acc['naive'])):.0%}"})
    return rows


def _props(P: pd.DataFrame) -> list[dict]:
    """Stat forecasts vs EWM baseline, and calibration of P(over) at a baseline-derived line.

    No historical sportsbook prop lines are used here (they are not freely available);
    the 'line' is the baseline rounded to .5. This shows whether the probabilities are
    honest, not whether they beat a book. Real lines are graded prospectively."""
    rows = []
    # out-of-sample residual pools from the first test season, applied to later seasons
    seasons = sorted(P.season.unique())
    if len(seasons) < 2:
        return rows
    if len(seasons) < 3:
        return rows
    # residual pools from the first test season, calibration fit on the second, evaluated on the rest
    first, second, rest = P[P.season == seasons[0]], P[P.season == seasons[1]], P[P.season > seasons[1]]
    pm = PlayerModel()
    pm.set_residuals(first[[c for c in first if c.startswith("p_")]], first)
    pm.fit_calibration(second, second, second)
    for stat, poss in PROP_STATS.items():
        d = rest[rest.position.isin(poss) & rest[f"p_{stat}"].notna() & rest[f"e_{stat}"].notna()]
        d = d[d[f"e_{stat}"] >= {"pass_yds": 150, "pass_att": 20, "pass_cmp": 12, "rush_yds": 25, "rush_att": 6,
                                 "rec": 2, "rec_yds": 25}[stat]]
        if d.empty:
            continue
        line = (np.floor(d[f"e_{stat}"]) + 0.5).to_numpy()
        pred = d[f"p_{stat}"].to_numpy()
        act = d[stat].to_numpy()
        p_over = np.array([pm.prob_over(stat, a, b) for a, b in zip(pred, line)], dtype=float)
        p_raw = np.array([pm.prob_over(stat, a, b, calibrated=False) for a, b in zip(pred, line)], dtype=float)
        over = (act > line).astype(float)
        brier = float(np.mean((p_over - over) ** 2))
        brier_raw = float(np.mean((p_raw - over) ** 2))
        brier_base = float(np.mean((over.mean() - over) ** 2))
        conf = np.abs(p_over - 0.5) >= 0.08
        hit = float(np.mean((p_over[conf] > 0.5) == (over[conf] == 1))) if conf.any() else np.nan
        bins = pd.cut(p_over, [0, .3, .4, .45, .5, .55, .6, .7, 1])
        cal = pd.DataFrame({"b": bins, "o": over, "p": p_over}).groupby("b", observed=True).agg(
            pred=("p", "mean"), actual=("o", "mean"), n=("o", "size")).reset_index()
        rows.append({"stat": stat, "n": len(d),
                     "mae_model": round(float(np.mean(np.abs(pred - act))), 2),
                     "mae_baseline": round(float(np.mean(np.abs(d[f"e_{stat}"] - act))), 2),
                     "brier_model": round(brier, 4), "brier_uncalibrated": round(brier_raw, 4), "brier_coinflip": round(brier_base, 4),
                     "confident_calls": int(conf.sum()), "confident_hit_rate": round(hit, 3),
                     "calibration": [{"bin": str(r.b), "pred": round(r.pred, 3), "actual": round(r.actual, 3), "n": int(r.n)}
                                     for r in cal.itertuples()]})
    # anytime TD (classifier)
    d = rest[rest.p_any_td.notna()]
    base = d.e_any_td.fillna(d.any_td.mean())
    rows.append({"stat": "any_td", "n": len(d),
                 "brier_model": round(float(np.mean((d.p_any_td - d.any_td) ** 2)), 4),
                 "brier_baseline": round(float(np.mean((base - d.any_td) ** 2)), 4)})
    return rows


def _dst(stats, tg, seasons) -> list[dict]:
    f = dstm.build(nv.dst_weekly(stats, tg), tg)
    rows = []
    for S in seasons:
        tr, te = f[f.season < S], f[(f.season == S) & f.dk_pts.notna() & f.e_dk.notna()]
        if te.empty:
            continue
        m = dstm.DSTModel().fit(tr)
        p = m.predict(te)
        rows.append({"season": int(S), "n": len(te), "mae_model": round(float(np.mean(np.abs(p - te.dk_pts))), 3),
                     "mae_ewm": round(float(np.mean(np.abs(te.e_dk - te.dk_pts))), 3),
                     "rank_corr_model": round(float(te.assign(p=p).groupby("week").apply(
                         lambda g: _spearman(g.p, g.dk_pts), include_groups=False).mean()), 3),
                     "rank_corr_ewm": round(float(te.groupby("week").apply(
                         lambda g: _spearman(g.e_dk, g.dk_pts), include_groups=False).mean()), 3)})
    return rows


def _games(sched, since) -> dict:
    d = gm.run(sched)
    t = d[(d.season >= since) & d.margin.notna() & d.spread_line.notna() & d.total_line.notna()]
    out = {"seasons": f"{since}–{int(t.season.max())}", "n_games": len(t),
           "mae_margin": {c: round(float(np.mean(np.abs(t.margin - t[c]))), 3) for c in ("spread_line", "model_spread", "blend_spread")},
           "mae_total": {c: round(float(np.mean(np.abs(t.points - t[c]))), 3) for c in ("total_line", "model_total", "blend_total")},
           "ats": [], "totals": [], "breakeven_at_-110": 0.524}
    edge = t.model_spread - t.spread_line
    for th in (1, 2, 3, 4, 5):
        s = t[edge.abs() >= th]
        home = edge[edge.abs() >= th] > 0
        win = np.where(home, s.margin > s.spread_line, s.margin < s.spread_line)
        push = (s.margin == s.spread_line).to_numpy()
        out["ats"].append({"edge_pts": th, "n": int((~push).sum()), "win_rate": round(float(win[~push].mean()), 3)})
    e2 = t.model_total - t.total_line
    for th in (1, 2, 3, 4, 5):
        s = t[e2.abs() >= th]
        over = e2[e2.abs() >= th] > 0
        win = np.where(over, s.points > s.total_line, s.points < s.total_line)
        push = (s.points == s.total_line).to_numpy()
        out["totals"].append({"edge_pts": th, "n": int((~push).sum()), "win_rate": round(float(win[~push].mean()), 3)})
    return out


# ------------------------------------------------------------------ prospective grading
def grade_frozen(offline=False) -> pd.DataFrame:
    """Grade every pregame-frozen projection run against final stats (per source)."""
    from . import storage
    from .names import match_key
    runs = storage.projection_runs_for_grading()
    if runs.empty:
        return pd.DataFrame()
    seasons = sorted(set(runs.season.dropna().astype(int)))
    stats = nv.weekly_stats(seasons, offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    act = stats[stats.position.isin(["QB", "RB", "WR", "TE"])]
    act = act.assign(key=[match_key(n, p, t) for n, p, t in zip(act.player_display_name, act.position, act.team)])
    dstw = nv.dst_weekly(stats, tg).assign(key=lambda d: "DST|" + d.team)
    rows = []
    for r in runs.itertuples():
        _, pool = storage.get_projection_run(r.id)
        wk_games = sched[(sched.season == r.season) & (sched.week == r.week)]
        if wk_games.home_score.isna().any():
            continue  # keep pending weeks ungraded
        a = pd.concat([act[(act.season == r.season) & (act.week == r.week)][["key", "dk_pts"]],
                       dstw[(dstw.season == r.season) & (dstw.week == r.week)][["key", "dk_pts"]]])
        g = pool[~pool.out].merge(a, on="key", how="left")
        g["played"] = g.dk_pts.notna()
        for src in ("proj_paid", "proj_sleeper", "proj_model", "proj_raw"):
            if src not in g:
                continue
            m = g[g.played & g[src].notna()]
            if len(m):
                rows.append({"run": r.id, "season": r.season, "week": r.week, "source": src.replace("proj_", ""),
                             "n": len(m), "mae": round(float((m[src] - m.dk_pts).abs().mean()), 3),
                             "bias": round(float((m[src] - m.dk_pts).mean()), 3)})
        rows.append({"run": r.id, "season": r.season, "week": r.week, "source": "did-not-play",
                     "n": int((~g.played).sum()), "mae": None, "bias": None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Sleeper vs model vs blends (runs on your computer)
def sleeper_compare(seasons=(2024, 2025), offline=False, progress=print) -> dict:
    """Retrospective head-to-head: Sleeper's own weekly projections (fetched for past weeks)
    vs the in-house model vs blends, on the same player-games. Picks the blend weight
    with the lowest error — evidence for the Settings weights instead of a guess."""
    from .sources import sleeper
    cur = nv.current_season()
    first = 2018
    stats = nv.weekly_stats(range(first, cur + 1), offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    feat = F.build(F.tidy_stats(stats), tg)
    feat = feat[feat.n_prior >= 2]
    spl, _ = sleeper.players(offline=offline)
    gs = spl.dropna(subset=["gsis_id"]).drop_duplicates("sleeper_id").set_index("sleeper_id").gsis_id
    rows = []
    for S in seasons:
        progress(f"{S}: training model on seasons before {S}…")
        m = PlayerModel().fit(feat[feat.season < S])
        te = feat[feat.season == S].copy()
        te["model"] = m.predict(te).p_dk_pts.values
        for w in sorted(te.week.unique()):
            try:
                sp, _ = sleeper.projections(int(S), int(w), offline=offline)
            except Exception as e:  # noqa: BLE001
                progress(f"  week {w}: Sleeper projections unavailable ({e})")
                continue
            sp["player_id"] = sp.sleeper_id.map(gs)
            j = te[te.week == w].merge(sp[["player_id", "proj_sleeper"]].dropna(), on="player_id")
            rows.append(j[["season", "week", "position", "dk_pts", "model", "proj_sleeper"]])
    if not rows:
        return {"n": 0, "note": "Sleeper projections could not be downloaded from this network."}
    J = pd.concat(rows, ignore_index=True)
    out = {"n": len(J), "seasons": list(seasons), "label": "retrospective; Sleeper numbers as published for each week"}
    grid = {}
    for wm in np.linspace(0, 1, 11):
        b = wm * J.model + (1 - wm) * J.proj_sleeper
        grid[round(float(wm), 1)] = round(float((b - J.dk_pts).abs().mean()), 4)
    out["mae_by_model_weight"] = grid
    out["best_model_weight"] = min(grid, key=grid.get)
    out["by_pos"] = [{"pos": p, "n": len(d), "mae_model": round(float((d.model - d.dk_pts).abs().mean()), 3),
                      "mae_sleeper": round(float((d.proj_sleeper - d.dk_pts).abs().mean()), 3)}
                     for p, d in J.groupby("position")]
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / f"sleeper_compare_{datetime.now():%Y%m%d_%H%M}.json").write_text(json.dumps(out, indent=1))
    return out


def suggest_weights(graded: pd.DataFrame) -> dict | None:
    """Inverse-error weights from prospective grading once there are ≥3 graded weeks per source."""
    if graded is None or graded.empty:
        return None
    g = graded[graded.source.isin(["paid", "sleeper", "model"])]
    weeks = g.groupby("source").week.nunique()
    ok = weeks[weeks >= 3].index
    if len(ok) < 2:
        return None
    mae = g[g.source.isin(ok)].groupby("source").apply(lambda d: np.average(d.mae, weights=d.n), include_groups=False)
    inv = (1 / mae ** 2)
    return (inv / inv.sum()).round(2).to_dict()


# ------------------------------------------------------------------ FantasyPros vs Sleeper vs model (runs on your computer)
SRC_COLS = {"paid": "FantasyPros", "sleeper": "Sleeper", "model": "Model"}
STAT_BLEND = {"pass_yds": ["QB"], "pass_att": ["QB"], "pass_cmp": ["QB"], "pass_td": ["QB"], "rush_yds": ["QB", "RB", "WR"],
              "rush_att": ["QB", "RB"], "rec": ["RB", "WR", "TE"], "rec_yds": ["RB", "WR", "TE"]}


def best_weights(df: pd.DataFrame, cols: list[str], target: str = "dk_pts", step: float = 0.05) -> tuple[dict, float]:
    """Blend weights (summing to 1) with the lowest average miss, searched on a grid. Rows must have every col."""
    import itertools
    d = df.dropna(subset=cols + [target])
    if d.empty:
        return {}, float("nan")
    X, y = d[cols].to_numpy(float), d[target].to_numpy(float)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    best, best_w = np.inf, None
    for w in itertools.product(grid, repeat=len(cols) - 1):
        last = 1 - sum(w)
        if last < -1e-9:
            continue
        wv = np.array(list(w) + [max(last, 0.0)])
        mae = float(np.abs(X @ wv - y).mean())
        if mae < best - 1e-9:
            best, best_w = mae, wv
    return {c: round(float(v), 2) for c, v in zip(cols, best_w)}, round(best, 3)


def _weights_report(J: pd.DataFrame, current: dict) -> dict:
    """Per-position comparison and best weights for whichever sources are present in J."""
    have = [c for c in ("paid", "sleeper", "model") if c in J and J[c].notna().sum() >= 50]
    out = {"sources": [SRC_COLS[c] for c in have], "by_pos": [], "weights": {}}
    for pos, d in J.groupby("position"):
        d = d.dropna(subset=have + ["dk_pts"])
        if len(d) < 100:
            continue
        w, mae = best_weights(d, have)
        cur_w = np.array([current.get(c, 0) for c in have], float)
        cur_mae = float(np.abs(d[have].to_numpy(float) @ (cur_w / cur_w.sum()) - d.dk_pts).mean()) if cur_w.sum() else np.nan
        row = {"pos": pos, "n": len(d), **{f"mae_{c}": round(float((d[c] - d.dk_pts).abs().mean()), 3) for c in have},
               "mae_current_blend": round(cur_mae, 3), "mae_best_blend": mae, **{f"w_{c}": w[c] for c in have}}
        out["by_pos"].append(row)
        out["weights"][pos] = {c: w[c] for c in have}
    return out


def source_compare(fp_key: str, seasons=None, current_weights=None, offline=False, progress=print) -> dict:
    """Retrospective test on past weeks: FantasyPros' published weekly projections (your API key),
    Sleeper's, and the in-house model (trained only on earlier seasons) vs what actually happened.
    Picks blend weights per position, and per stat for the props forecasts. Past-week FantasyPros
    projections may not be available on every key; the result says so and the test still runs on
    the sources that are."""
    from .names import match_key
    from .sources import fantasypros, sleeper
    cur = nv.current_season()
    seasons = list(seasons or (cur - 1, cur))
    stats = nv.weekly_stats(range(2012, cur + 1), offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    feat = F.build(F.tidy_stats(stats), tg)
    feat = feat[feat.n_prior >= 2]
    try:
        spl, _ = sleeper.players(offline=offline)
        gs = spl.dropna(subset=["gsis_id"]).drop_duplicates("sleeper_id").set_index("sleeper_id").gsis_id
    except Exception:  # noqa: BLE001
        gs = None
    notes, rows, fp_weeks, sl_weeks = [], [], 0, 0
    for S in seasons:
        te = feat[(feat.season == S) & (feat.played == 1)].copy()
        if te.empty:
            continue
        progress(f"{S}: training model on seasons before {S}…")
        m = PlayerModel().fit(feat[feat.season < S])
        pr = m.predict(te)
        te["model"] = pr.p_dk_pts.values
        for t in STAT_BLEND:
            te[f"m_{t}"] = pr[f"p_{t}"].values
        te["key"] = [match_key(n, p, tm) for n, p, tm in zip(te.name, te.position, te.team)]
        for w in sorted(te.week.unique()):
            j = te[te.week == w].copy()
            progress(f"{S} week {w}: FantasyPros + Sleeper projections…")
            if fp_key:
                try:
                    fp, info = fantasypros.projections(fp_key, int(S), int(w), offline=offline)
                    fp = fp[fp.pos != "DST"].drop_duplicates("key")
                    if len(fp) >= 50 and not info.get("limited"):
                        keep = ["key", "proj_paid"] + [c for c in fp if c.startswith("fp_")]
                        j = j.merge(fp[keep].rename(columns={"proj_paid": "paid"}), on="key", how="left")
                        fp_weeks += 1
                except Exception as e:  # noqa: BLE001
                    if not any("FantasyPros" in n for n in notes):
                        notes.append(f"FantasyPros past weeks unavailable ({getattr(e, 'what', str(e))[:90]}).")
            if gs is not None:
                try:
                    sp, _ = sleeper.projections(int(S), int(w), offline=offline)
                    sp["player_id"] = sp.sleeper_id.map(gs)
                    j = j.merge(sp[["player_id", "proj_sleeper"]].dropna().drop_duplicates("player_id")
                                .rename(columns={"proj_sleeper": "sleeper"}), on="player_id", how="left")
                    sl_weeks += 1
                except Exception:  # noqa: BLE001
                    pass
            rows.append(j)
    if not rows:
        return {"n": 0, "note": "No completed weeks to test."}
    J = pd.concat(rows, ignore_index=True)
    for c in ("paid", "sleeper"):
        if c not in J:
            J[c] = np.nan
    if fp_key and fp_weeks == 0 and not notes:
        notes.append("FantasyPros returned no projections for past weeks on this key — it will be graded live "
                     "from now on (each Refresh freezes its numbers; graded after the games).")
    cw = current_weights or {}
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "seasons": seasons, "n": int(len(J)),
           "weeks_fantasypros": fp_weeks, "weeks_sleeper": sl_weeks, "notes": notes,
           "label": "retrospective; FantasyPros and Sleeper numbers as published for each week",
           **_weights_report(J, cw)}
    # props: should FantasyPros' stat lines be blended into the model's stat forecasts?
    stat_rows, stat_w = [], {}
    if fp_weeks:
        for t, poss in STAT_BLEND.items():
            d = J[J.position.isin(poss)].dropna(subset=[f"m_{t}", f"fp_{t}", t])
            if len(d) < 200:
                continue
            w, mae = best_weights(pd.DataFrame({"model": d[f"m_{t}"], "fp": d[f"fp_{t}"], t: d[t]}), ["model", "fp"], target=t)
            stat_w[t] = w["fp"]
            stat_rows.append({"stat": t, "n": len(d), "mae_model": round(float((d[f"m_{t}"] - d[t]).abs().mean()), 2),
                              "mae_fantasypros": round(float((d[f"fp_{t}"] - d[t]).abs().mean()), 2),
                              "mae_blend": round(mae, 2), "w_fantasypros": w["fp"]})
    out["stats"], out["stat_weights"] = stat_rows, stat_w
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / f"source_compare_{datetime.now():%Y%m%d_%H%M}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def live_weights(min_weeks: int = 3, offline=False) -> dict:
    """Per-position weights from the app's own frozen pregame projections once ≥ min_weeks are graded —
    the prospective check that keeps the retrospective weights honest."""
    from . import storage
    from .names import match_key
    runs = storage.projection_runs_for_grading()
    if runs.empty:
        return {"weeks": 0}
    stats = nv.weekly_stats(sorted(set(runs.season.dropna().astype(int))), offline=offline)
    sched = nv.schedules(offline=offline)
    act = stats[stats.position.isin(["QB", "RB", "WR", "TE"])]
    act = act.assign(key=[match_key(n, p, t) for n, p, t in zip(act.player_display_name, act.position, act.team)])
    parts = []
    for r in runs.itertuples():
        if sched[(sched.season == r.season) & (sched.week == r.week)].home_score.isna().any():
            continue
        _, pool = storage.get_projection_run(r.id)
        a = act[(act.season == r.season) & (act.week == r.week)][["key", "dk_pts"]]
        g = pool[~pool.out & (pool.pos != "DST")].merge(a, on="key", how="inner")
        g = g.rename(columns={"proj_paid": "paid", "proj_sleeper": "sleeper", "proj_model": "model", "pos": "position"})
        parts.append(g.assign(season=r.season, week=r.week))
    if not parts:
        return {"weeks": 0}
    J = pd.concat(parts, ignore_index=True)
    weeks = int(J.groupby(["season", "week"]).ngroups)
    out = {"weeks": weeks, "n": int(len(J))}
    if weeks >= min_weeks:
        out.update(_weights_report(J, {}))
    return out
