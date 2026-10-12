"""Training (cached weekly) and live forecasting for a slate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HOME
from ..errors import StepError
from ..names import loose_key, match_key, norm_name
from ..sources import nflverse as nv
from . import dst as dstm
from . import features as F
from .players import MODEL_VERSION, PlayerModel

MODEL_DIR = HOME / "models"
FIRST_SEASON = 2018


def _hist(offline=False, first=FIRST_SEASON):
    cur = nv.current_season()
    stats = nv.weekly_stats(range(first, cur + 1), offline=offline)
    sched = nv.schedules(offline=offline)
    tg = nv.team_games(sched)
    return stats, sched, tg


def get_models(offline=False, force=False, progress=lambda m: None):
    """Return (PlayerModel, DSTModel, meta). Retrains when a newer completed week exists."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stats, sched, tg = _hist(offline)
    last = stats.sort_values(["season", "week"]).iloc[-1]
    tag = f"{MODEL_VERSION}_{int(last.season)}w{int(last.week):02d}"
    pm_p, dm_p, meta_p = MODEL_DIR / f"{tag}_players.pkl", MODEL_DIR / f"{tag}_dst.pkl", MODEL_DIR / f"{tag}.json"
    if pm_p.exists() and dm_p.exists() and not force:
        import pickle
        return PlayerModel.load(pm_p), pickle.loads(dm_p.read_bytes()), json.loads(meta_p.read_text())
    t0 = time.time()
    progress("Building historical features (first run of the week takes a few minutes)…")
    feat = F.build(F.tidy_stats(stats), tg)
    cur = int(last.season)
    # out-of-sample residuals: fit on seasons before the most recent full season, predict after it
    progress("Measuring forecast error on unseen games…")
    # residual pools from season cur-2, probability calibration from cur-1 and cur (all unseen by the hold-out model)
    pm_hold = PlayerModel().fit(feat[feat.season < cur - 2])
    h1 = feat[(feat.season == cur - 2) & (feat.n_prior >= 1)]
    h2 = feat[(feat.season >= cur - 1) & (feat.n_prior >= 1)]
    progress("Training final player model…")
    pm = PlayerModel().fit(feat)
    pm.set_residuals(pm_hold.predict(h1), h1)
    pm.fit_calibration(pm_hold.predict(h2), h2, h2)
    progress("Training DST model…")
    dfeat = dstm.build(nv.dst_weekly(stats, tg), tg)
    dm = dstm.DSTModel().fit(dfeat)
    import pickle
    pm.save(pm_p)
    dm_p.write_bytes(pickle.dumps(dm))
    meta = {"tag": tag, "version": MODEL_VERSION, "trained_through": pm.trained_through,
            "rows": int(len(feat)), "seconds": round(time.time() - t0, 1)}
    meta_p.write_text(json.dumps(meta))
    return pm, dm, meta


# ------------------------------------------------------------------ live slate forecast
def locate_week(slate: pd.DataFrame, sched: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """Find the nflverse season/week for the slate's games. Refuses to guess across weeks."""
    from ..names import team as nt
    slate_games = {tuple(sorted((nt(a), nt(b)))) for a, b in
                   (g.split("@") for g in slate.game.unique() if "@" in g)}
    s = sched[sched.home_score.isna() & (sched.game_type == "REG")].copy()
    s["pair"] = [tuple(sorted((a, h))) for a, h in zip(s.away_team, s.home_team)]
    s = s[s.pair.isin(slate_games)]
    if s.empty:
        raise StepError("Match slate to schedule", "None of the slate's games are on the upcoming NFL schedule.",
                        "The slate may already be over, or it is not a regular-season slate.",
                        "Load the current week's slate.")
    # the earliest unplayed week that contains the most slate games (same matchup can recur later in the season)
    wk = s.groupby(["season", "week"]).size().reset_index(name="n").sort_values(["n", "season", "week"],
                                                                                 ascending=[False, True, True])
    season, week = int(wk.iloc[0].season), int(wk.iloc[0].week)
    games = sched[(sched.season == season) & (sched.week == week)]
    sched_games = {tuple(sorted((r.away_team, r.home_team))) for r in games.itertuples()}
    missing = slate_games - sched_games
    if missing:
        raise StepError("Match slate to schedule", f"{len(missing)} slate game(s) are not in {season} week {week}: "
                        f"{', '.join('@'.join(m) for m in missing)}.",
                        "The slate mixes weeks or the schedule feed is behind.",
                        "Reload in a few hours; if it persists, report the game in Data Status.")
    return season, week, games


def forecast(slate: pd.DataFrame, sleeper_players: pd.DataFrame | None, pm: PlayerModel, dm,
             lines: pd.DataFrame | None = None, offline=False) -> tuple[pd.DataFrame, list[str], dict]:
    """In-house model forecast for every slate player. Returns (df[dk_id, proj_model, model stats…], warnings, info)."""
    warns = []
    cur = nv.current_season()
    stats, sched, tg = _hist(offline, first=cur - 3)
    season, week, games = locate_week(slate, sched)
    tg = apply_lines(tg, lines, season, week)  # fresher market lines override schedule lines
    wk = tg[(tg.season == season) & (tg.week == week)]
    if wk.total.isna().any():
        warns.append(f"{int(wk.total.isna().sum() / 2)} game(s) have no betting line yet — model runs without game totals for them.")

    hist = F.tidy_stats(stats)
    hist = hist[(hist.season < season) | (hist.week < week)]

    # identity: slate → nflverse player_id (via Sleeper gsis_id, else name/team/pos)
    sl = slate[slate.pos != "DST"].copy()
    if sleeper_players is not None and not sleeper_players.empty:
        sp = sleeper_players.dropna(subset=["gsis_id"]).drop_duplicates("key")
        sl = sl.merge(sp[["key", "gsis_id"]], on="key", how="left")
        miss = sl.gsis_id.isna()
        if miss.any():
            sp2 = sleeper_players.dropna(subset=["gsis_id"]).assign(lk=lambda d: d.lkey).drop_duplicates("lk", keep=False)
            sl.loc[miss, "gsis_id"] = sl.loc[miss].assign(lk=[loose_key(n, p) for n, p in zip(sl.loc[miss, "name"], sl.loc[miss, "pos"])]).merge(
                sp2[["lk", "gsis_id"]], on="lk", how="left")["gsis_id"].values
    else:
        sl["gsis_id"] = np.nan
    latest = hist.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    latest["key"] = [match_key(n, p, t) for n, p, t in zip(latest.name, latest.position, latest.team)]
    latest["lk"] = [loose_key(n, p) for n, p in zip(latest.name, latest.position)]
    by_key = latest.drop_duplicates("key", keep=False).set_index("key").player_id
    by_lk = latest.drop_duplicates("lk", keep=False).set_index("lk").player_id
    known = set(hist.player_id)
    pid = []
    for r in sl.itertuples():
        g = r.gsis_id if isinstance(r.gsis_id, str) and r.gsis_id in known else None
        g = g or by_key.get(r.key) or by_lk.get(loose_key(r.name, r.pos))
        pid.append(g)
    sl["player_id"] = pid
    unmatched = sl.player_id.isna().sum()
    if unmatched:
        warns.append(f"{int(unmatched)} slate players have no NFL game history (rookies/new signings) — "
                     "the in-house model skips them; other sources still cover them.")

    # expected actives: slate players not ruled out
    out_status = {"O", "IR", "D", "OUT", "Out"}
    active = sl[sl.player_id.notna() & ~sl.status.isin(out_status)]
    fut = active.merge(wk[["team", "opp", "game_id"]], on=["team", "opp"], how="left")
    fut = pd.DataFrame({"player_id": fut.player_id, "name": fut.name, "position": fut.pos, "season": season,
                        "week": week, "game_id": fut.game_id, "team": fut.team, "opp": fut.opp, "dk_pts": np.nan,
                        **{c: np.nan for c in F.STAT_COLS.values()}, "any_td": np.nan, "played": 1})
    # teammates in history who are not on the slate but are not ruled out still "occupy" volume only if they
    # appear on the slate; players missing from the slate entirely are treated as inactive (vacated)
    feat = F.build(pd.concat([hist, fut], ignore_index=True), tg)
    fx = feat[(feat.season == season) & (feat.week == week)]
    pr = pm.predict(fx)
    fx = pd.concat([fx[["player_id", "team"]].reset_index(drop=True), pr.reset_index(drop=True),
                    fx[["vac_tgt", "vac_rush", "implied", "e_tgt_share", "e_rush_share", "n_prior"]].reset_index(drop=True)], axis=1)
    res = sl[["dk_id", "player_id"]].merge(fx, on="player_id", how="left").drop_duplicates("dk_id")
    res = res.rename(columns={"p_dk_pts": "proj_model"})
    # thin-history players: model output only when ≥2 prior games
    res.loc[res.n_prior.fillna(0) < 2, "proj_model"] = np.nan

    # DST
    dst_rows = slate[slate.pos == "DST"]
    if len(dst_rows):
        dhist = nv.dst_weekly(stats, tg)
        dhist = dhist[(dhist.season < season) | (dhist.week < week)]
        dfut = pd.DataFrame({"season": season, "week": week, "team": dst_rows.team.values, "opp": dst_rows.opp.values,
                             "sk": np.nan, "it": np.nan, "fr": np.nan, "pa": np.nan, "dk_pts": np.nan})
        dfut = dfut.merge(wk[["team", "implied", "spread", "total", "game_id"]], on="team", how="left")
        df_ = dstm.build(pd.concat([dhist, dfut], ignore_index=True), tg)
        fx_d = df_[(df_.season == season) & (df_.week == week)]
        dproj = pd.DataFrame({"team": fx_d.team.values, "proj_model": dm.predict(fx_d)})
        dres = dst_rows[["dk_id", "team"]].merge(dproj, on="team", how="left")[["dk_id", "proj_model"]]
        res = pd.concat([res, dres], ignore_index=True)
    info = {"season": season, "week": week, "games": len(wk) // 2}
    return res, warns, info


def apply_lines(tg: pd.DataFrame, lines: pd.DataFrame | None, season: int, week: int) -> pd.DataFrame:
    tg = tg.copy()
    if lines is not None and not lines.empty:
        m = (tg.season == season) & (tg.week == week)
        for r in lines.itertuples():
            hm, aw = m & (tg.team == r.home) & (tg.opp == r.away), m & (tg.team == r.away) & (tg.opp == r.home)
            tg.loc[hm, ["spread", "total"]] = [r.spread_nv, r.total]
            tg.loc[aw, ["spread", "total"]] = [-r.spread_nv, r.total]
        tg["implied"] = (tg.total + tg.spread) / 2
    return tg


def forecast_week(season: int, week: int, sleeper_players: pd.DataFrame | None, pm: PlayerModel,
                  lines: pd.DataFrame | None = None, offline=False) -> tuple[pd.DataFrame, list[str]]:
    """Stat forecasts for every likely-active skill player in a week's games (used by Research).

    Likely active = played for the team in one of its last 3 games and not ruled out by Sleeper."""
    warns = []
    stats, sched, tg = _hist(offline, first=season - 3)
    tg = apply_lines(tg, lines, season, week)
    wk = tg[(tg.season == season) & (tg.week == week)]
    if wk.empty:
        raise StepError("Research forecast", f"No games found for {season} week {week}.", "", "Pick another week.")
    hist = F.tidy_stats(stats)
    hist = hist[(hist.season < season) | (hist.week < week)]
    tw = hist[["team", "season", "week"]].drop_duplicates().sort_values(["season", "week"])
    recent = tw.groupby("team").tail(3)
    act = hist.merge(recent, on=["team", "season", "week"]).sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    # current team / status from Sleeper when available
    if sleeper_players is not None and not sleeper_players.empty:
        sp = sleeper_players.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
        act["team"] = [sp.at[p, "team"] if p in sp.index else t for p, t in zip(act.player_id, act.team)]
        out = {p for p in sp.index[sp.injury.isin(["O", "D"])]}
        act = act[~act.player_id.isin(out)]
        act["injury"] = [sp.at[p, "injury"] if p in sp.index else "" for p in act.player_id]
    else:
        warns.append("Injury status unavailable — ruled-out players may still appear.")
        act["injury"] = ""
    act = act.merge(wk[["team", "opp", "game_id"]], on="team", how="inner", suffixes=("_old", ""))
    fut = pd.DataFrame({"player_id": act.player_id, "name": act.name, "position": act.position, "season": season,
                        "week": week, "game_id": act.game_id, "team": act.team, "opp": act["opp"], "dk_pts": np.nan,
                        **{c: np.nan for c in F.STAT_COLS.values()}, "any_td": np.nan, "played": 1})
    feat = F.build(pd.concat([hist, fut], ignore_index=True), tg)
    fx = feat[(feat.season == season) & (feat.week == week)].reset_index(drop=True)
    pr = pm.predict(fx)
    res = pd.concat([fx[["player_id", "name", "position", "team", "opp", "game_id", "n_prior", "implied", "vac_tgt",
                         "vac_rush"]], pr.reset_index(drop=True)], axis=1)
    res = res.merge(act[["player_id", "injury"]], on="player_id", how="left")
    return res[res.n_prior >= 2].reset_index(drop=True), warns
