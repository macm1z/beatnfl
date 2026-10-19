"""Feature construction with a strict information cutoff.

Every feature for a player-game uses only games *before* that game (shift(1) before
any rolling/EWM). Game lines are the closing spread/total, i.e. what is known at
kickoff. "Vacated" role features use who actually played, which at forecast time is
known from inactives/injury designations — the live pipeline feeds expected-out
players into the same function so training and live forecasting use identical code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF = 4  # EWM half-life in games

STAT_COLS = {
    "attempts": "pass_att", "completions": "pass_cmp", "passing_yards": "pass_yds", "passing_tds": "pass_td",
    "passing_interceptions": "pass_int", "carries": "rush_att", "rushing_yards": "rush_yds", "rushing_tds": "rush_td",
    "targets": "tgt", "receptions": "rec", "receiving_yards": "rec_yds", "receiving_tds": "rec_td",
    "passing_air_yards": "pass_air", "receiving_air_yards": "rec_air",
}
OFF_POS = ["QB", "RB", "WR", "TE"]


def tidy_stats(stats: pd.DataFrame) -> pd.DataFrame:
    s = stats[stats.position.isin(OFF_POS)].copy()
    s = s.rename(columns=STAT_COLS)
    keep = ["player_id", "player_display_name", "position", "season", "week", "game_id", "team", "opponent_team",
            "dk_pts"] + list(STAT_COLS.values())
    s = s[[c for c in keep if c in s.columns]].rename(columns={"player_display_name": "name", "opponent_team": "opp"})
    for c in STAT_COLS.values():
        s[c] = s[c].fillna(0.0) if c in s else 0.0
    s["any_td"] = ((s.rush_td + s.rec_td) > 0).astype(float)
    s["played"] = 1
    return attach_snaps(s)


def attach_snaps(s: pd.DataFrame) -> pd.DataFrame:
    """Share of the team's offensive snaps each player was on the field for (nflverse snap counts).
    Missing data stays NaN — the model handles gaps. Tested 2022–2025: lower miss every season (Evidence)."""
    s = s.copy()
    s["snap_pct"] = np.nan
    try:
        from ..names import norm_name
        from ..sources import nflverse as nv
        seasons = sorted(set(s.season.dropna().astype(int)))
        try:
            sn = nv.snaps(seasons)
        except Exception:  # noqa: BLE001
            sn = nv.snaps(seasons, offline=True)
        sn = sn[sn.game_type.eq("REG") | sn.game_type.isna()] if "game_type" in sn else sn
        sn = sn.assign(nn=sn.player.map(norm_name)).drop_duplicates(["season", "week", "team", "nn"])
        nn = s.name.map(norm_name)
        m = s[["season", "week", "team"]].assign(nn=nn.values).merge(
            sn[["season", "week", "team", "nn", "offense_pct"]], on=["season", "week", "team", "nn"], how="left")
        s["snap_pct"] = m.offense_pct.values
    except Exception:  # noqa: BLE001
        pass
    return s


def _ewm_prior(df: pd.DataFrame, by: str, cols: list[str], half=HALF, prefix="e_") -> pd.DataFrame:
    """EWM of each column over a group's *previous* rows (row order must be chronological)."""
    key = df[by]
    shifted = df[cols].groupby(key, sort=False).shift(1)
    e = shifted.groupby(key, sort=False).ewm(halflife=half, min_periods=1).mean()
    e = e.reset_index(level=0, drop=True).reindex(df.index)
    e.columns = [prefix + c for c in cols]
    return e


def build(rows: pd.DataFrame, tg: pd.DataFrame, tm: pd.DataFrame | None = None) -> pd.DataFrame:
    """rows: tidy player-games (historical, played) plus optional future rows with NaN outcomes
    and played=1 for expected-active players. tg: team_games() table (needs rows for future games).
    Returns one feature row per input row."""
    df = rows.copy().sort_values(["season", "week"]).reset_index(drop=True)
    df["order"] = df.season * 100 + df.week

    # ---- team volume per game (from players who played; future rows contribute NaN)
    team = df.groupby(["team", "season", "week"], as_index=False).agg(
        t_pass_att=("pass_att", lambda x: x.sum(min_count=1)), t_rush_att=("rush_att", lambda x: x.sum(min_count=1)),
        t_tgt=("tgt", lambda x: x.sum(min_count=1)), t_pass_yds=("pass_yds", lambda x: x.sum(min_count=1)),
        t_rush_yds=("rush_yds", lambda x: x.sum(min_count=1)), t_dk=("dk_pts", lambda x: x.sum(min_count=1)))
    team = team.merge(tg[["team", "season", "week", "opp", "pf", "pa", "spread", "total", "implied", "home"]],
                      on=["team", "season", "week"], how="left").sort_values(["season", "week"]).reset_index(drop=True)
    team[["e_t_pass_att", "e_t_rush_att", "e_t_pf", "e_t_dk"]] = _ewm_prior(
        team, "team", ["t_pass_att", "t_rush_att", "pf", "t_dk"]).values
    # defense allowed (opponent's offense output), prior-only, keyed by defending team
    allowed = team[["opp", "season", "week", "t_pass_yds", "t_rush_yds", "t_pass_att", "t_rush_att", "pf"]].rename(
        columns={"opp": "def_team"}).sort_values(["season", "week"]).reset_index(drop=True)
    allowed["ypa_allowed"] = allowed.t_pass_yds / allowed.t_pass_att.replace(0, np.nan)
    allowed["ypc_allowed"] = allowed.t_rush_yds / allowed.t_rush_att.replace(0, np.nan)
    allowed[["e_ypa_allowed", "e_ypc_allowed", "e_pts_allowed"]] = _ewm_prior(
        allowed, "def_team", ["ypa_allowed", "ypc_allowed", "pf"], half=6).values

    # ---- player shares and EWMs
    df = df.merge(team[["team", "season", "week", "t_pass_att", "t_rush_att", "t_tgt"]], on=["team", "season", "week"], how="left")
    df["tgt_share"] = df.tgt / df.t_tgt.replace(0, np.nan)
    df["rush_share"] = df.rush_att / df.t_rush_att.replace(0, np.nan)
    df["ypt"] = df.rec_yds / df.tgt.replace(0, np.nan)
    df["ypc"] = df.rush_yds / df.rush_att.replace(0, np.nan)
    df["ypa"] = df.pass_yds / df.pass_att.replace(0, np.nan)
    df["adot"] = df.rec_air / df.tgt.replace(0, np.nan)
    ecols = ["dk_pts", "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "rush_att", "rush_yds", "tgt", "rec",
             "rec_yds", "rush_td", "rec_td", "tgt_share", "rush_share", "ypt", "ypc", "ypa", "adot", "any_td"]
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df = pd.concat([df, _ewm_prior(df, "player_id", ecols)], axis=1)
    g = df.groupby("player_id", sort=False)
    df["n_prior"] = g.cumcount()
    df["e_dk_long"] = _ewm_prior(df, "player_id", ["dk_pts"], half=12)["e_dk_pts"]
    sh = g["dk_pts"].shift(1)
    gsh = sh.groupby(df.player_id, sort=False)
    df["dk_max8"] = gsh.rolling(8, min_periods=1).max().reset_index(level=0, drop=True).reindex(df.index)
    df["dk_sd8"] = gsh.rolling(8, min_periods=2).std().reset_index(level=0, drop=True).reindex(df.index)
    # season-to-date mean (the naive baseline)
    ps = df.player_id + "_" + df.season.astype(str)
    sh2 = df.groupby(ps, sort=False)["dk_pts"].shift(1)
    df["std_mean"] = sh2.groupby(ps, sort=False).expanding().mean().reset_index(level=0, drop=True).reindex(df.index)
    last_season = df.groupby(["player_id", "season"])["dk_pts"].mean().rename("prev_season_mean").reset_index()
    last_season["season"] += 1
    df = df.merge(last_season, on=["player_id", "season"], how="left")

    # ---- vacated opportunity: recent teammates absent from this game
    df = df.merge(_vacated(df), on=["team", "season", "week"], how="left")
    for c in ("vac_tgt", "vac_rush", "vac_tgt_pos", "vac_rush_pos"):
        if c not in df:
            df[c] = 0.0
    df["vac_tgt"] = df.vac_tgt.fillna(0)
    df["vac_rush"] = df.vac_rush.fillna(0)

    # ---- context joins
    df = df.merge(team[["team", "season", "week", "spread", "total", "implied", "home", "e_t_pass_att",
                        "e_t_rush_att", "e_t_pf", "e_t_dk"]], on=["team", "season", "week"], how="left")
    df = df.merge(allowed[["def_team", "season", "week", "e_ypa_allowed", "e_ypc_allowed", "e_pts_allowed"]].rename(
        columns={"def_team": "opp"}), on=["opp", "season", "week"], how="left")
    dvp = _dvp(df)
    df = df.merge(dvp, on=["opp", "position", "season", "week"], how="left")

    # ---- structural expectations (interpretable stat forecasts the GBM can lean on)
    pa_mult = 1 - 0.012 * df.spread.fillna(0)
    df["x_tgt"] = df.e_tgt_share * df.e_t_pass_att * pa_mult * (1 + df.vac_tgt.clip(0, 0.6) * 0.8)
    df["x_rush"] = df.e_rush_share * df.e_t_rush_att * (1 + 0.012 * df.spread.fillna(0)) * (1 + df.vac_rush.clip(0, 0.7) * 0.8)
    df["implied_ratio"] = df.implied / df.e_t_pf
    df["pos_code"] = df.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
    # ---- starting quarterback (listed starter; tested 2022–2025: lower miss every season, see Evidence)
    df = df.merge(_qb_context(df, tg), on=["team", "season", "week"], how="left")
    df["is_starter"] = np.where(df.position == "QB", (df.player_id == df.qb_id).astype(float), 0.0)
    df.loc[(df.position == "QB") & df.qb_id.isna(), "is_starter"] = np.nan
    df = df.drop(columns=["qb_id"])
    # ---- snap share, prior games only (this game's snaps are unknown at forecast time)
    if "snap_pct" not in df:
        df["snap_pct"] = np.nan
    prev = df.groupby("player_id", sort=False).snap_pct.shift(1)
    key = df.player_id
    df["snap_last"] = prev
    df["snap_e"] = prev.groupby(key, sort=False).ewm(halflife=3, ignore_na=True).mean().reset_index(level=0, drop=True).reindex(df.index)
    df["snap_last2"] = prev.groupby(key, sort=False).rolling(2, min_periods=1).mean().reset_index(level=0, drop=True).reindex(df.index)
    df["snap_trend"] = df.snap_last2 - df.snap_e
    if tm is not None and not tm.empty:
        df = add_team_metrics(df, tm)
    return df


def _vacated(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of recent shares of teammates who played in either of the team's previous
    two games but not in this one (vectorised)."""
    tw = df[["team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    tw["k"] = tw.groupby("team").cumcount()
    app = df[df.played == 1].merge(tw, on=["team", "season", "week"])
    app = app.assign(ts=app.tgt_share.fillna(0) * 0.5 + app.e_tgt_share.fillna(app.tgt_share.fillna(0)) * 0.5,
                     rs=app.rush_share.fillna(0) * 0.5 + app.e_rush_share.fillna(app.rush_share.fillna(0)) * 0.5)
    base = app[["team", "k", "player_id", "ts", "rs"]]
    cand = pd.concat([base.assign(kt=base.k + 1), base.assign(kt=base.k + 2)])
    cand = cand.sort_values("k").drop_duplicates(["team", "kt", "player_id"], keep="last")
    present = base[["team", "k", "player_id"]].rename(columns={"k": "kt"}).assign(here=1)
    cand = cand.merge(present, on=["team", "kt", "player_id"], how="left")
    gone = cand[cand.here.isna()]
    v = gone.groupby(["team", "kt"], as_index=False)[["ts", "rs"]].sum().rename(
        columns={"kt": "k", "ts": "vac_tgt", "rs": "vac_rush"})
    v = v.merge(tw, on=["team", "k"])  # drops k beyond the last known game
    return v[["team", "season", "week", "vac_tgt", "vac_rush"]]


def _dvp(df: pd.DataFrame) -> pd.DataFrame:
    """DK points allowed by each defense to each position, relative to league, prior-only."""
    a = df[df.played == 1].groupby(["opp", "position", "season", "week"], as_index=False)["dk_pts"].sum(min_count=1)
    a = a.sort_values(["season", "week"]).reset_index(drop=True)
    a["op"] = a.opp + "|" + a.position
    a["e"] = _ewm_prior(a, "op", ["dk_pts"], half=6)["e_dk_pts"]
    lg = a.groupby(["position", "season", "week"])["dk_pts"].mean().rename("lg").reset_index()
    lg = lg.sort_values(["season", "week"])
    lg = lg.reset_index(drop=True)
    lg["lg_e"] = _ewm_prior(lg, "position", ["lg"], half=10)["e_lg"]
    a = a.merge(lg[["position", "season", "week", "lg_e"]], on=["position", "season", "week"], how="left")
    a["dvp"] = a.e / a.lg_e
    # future rows: carry last known value
    return a[["opp", "position", "season", "week", "dvp"]]


def _qb_context(df: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: who starts at QB, whether that's a change from the usual starter, and the
    starter's track record going in (prior games only)."""
    cols = ["team", "season", "week", "qb_id", "qb_new", "qb_e_ypa", "qb_e_dk", "qb_starts", "qb_dk_drop"]
    if "qb_id" not in tg:
        return pd.DataFrame(columns=cols)
    st = tg[["team", "season", "week", "qb_id"]].drop_duplicates(["team", "season", "week"]).copy()
    st["order"] = (st.season * 100 + st.week).astype("int64")
    st = st.sort_values(["team", "order"])
    prev = [st.groupby("team").qb_id.shift(k) for k in (1, 2, 3)]
    usual = []
    for a, b, c in zip(*prev):
        v = [x for x in (a, b, c) if isinstance(x, str)]
        usual.append(max(v, key=v.count) if v else None)
    st["usual"] = usual
    st["qb_new"] = ((st.usual.notna()) & st.qb_id.notna() & (st.qb_id != st.usual)).astype(float)
    # each QB's record after every game he played, then looked up as of the game before
    q = df[(df.position == "QB") & df.dk_pts.notna()][["player_id", "season", "week", "pass_yds", "pass_att", "dk_pts"]].copy()
    q["order"] = (q.season * 100 + q.week).astype("int64")
    q = q.sort_values("order")
    q["ypa"] = q.pass_yds / q.pass_att.replace(0, np.nan)
    g = q.groupby("player_id", sort=False)
    q["e_ypa"] = g.ypa.transform(lambda x: x.ewm(halflife=6, ignore_na=True).mean())
    q["e_dk"] = g.dk_pts.transform(lambda x: x.ewm(halflife=6, ignore_na=True).mean())
    q["n"] = g.cumcount() + 1
    rec = q[["player_id", "order", "e_ypa", "e_dk", "n"]].astype({"player_id": str}).sort_values("order")

    def asof(ids: pd.Series) -> pd.DataFrame:
        ok = ids.notna().values
        left = pd.DataFrame({"player_id": ids.astype(str).values, "order": st.order.values, "_i": np.arange(len(st))})
        m = pd.merge_asof(left[ok].sort_values("order"), rec, on="order", by="player_id", allow_exact_matches=False)
        return m.set_index("_i").reindex(np.arange(len(st)))
    a, u = asof(st.qb_id), asof(st.usual)
    st["qb_e_ypa"], st["qb_e_dk"], st["qb_starts"] = a.e_ypa.values, a.e_dk.values, a.n.fillna(0).values
    st["qb_dk_drop"] = np.where(st.qb_new == 1, u.e_dk.values - a.e_dk.values, 0.0)
    return st[cols]


FEATURES = ["pos_code", "e_dk_pts", "e_dk_long", "dk_max8", "dk_sd8", "std_mean", "prev_season_mean", "n_prior",
            "e_pass_att", "e_pass_cmp", "e_pass_yds", "e_pass_td", "e_pass_int", "e_rush_att", "e_rush_yds", "e_tgt",
            "e_rec", "e_rec_yds", "e_rush_td", "e_rec_td", "e_tgt_share", "e_rush_share", "e_ypt", "e_ypc", "e_ypa",
            "e_adot", "e_any_td", "vac_tgt", "vac_rush", "spread", "total", "implied", "home", "e_t_pass_att",
            "e_t_rush_att", "e_t_pf", "e_t_dk", "e_ypa_allowed", "e_ypc_allowed", "e_pts_allowed", "dvp", "x_tgt",
            "x_rush", "implied_ratio", "snap_last", "snap_e", "snap_last2", "snap_trend",
            "qb_new", "qb_e_ypa", "qb_e_dk", "qb_starts", "qb_dk_drop", "is_starter"]


# ------------------------------------------------------------------ play-by-play matchup features
OFF_M = ["off_epa", "off_sr", "off_pass_epa", "off_rush_epa", "off_explosive", "off_neutral_pass", "off_proe",
         "off_sack_rate", "off_to_rate", "off_plays"]
DEF_M = ["def_epa", "def_sr", "def_pass_epa", "def_rush_epa", "def_explosive", "def_sack_rate", "def_to_rate"]
PBP_FEATURES = [f"t_{c}" for c in OFF_M] + ["t_rz_td", "o_rz_td"] + [f"o_{c}" for c in DEF_M]


def team_metric_priors(tm: pd.DataFrame, half: float = 6) -> pd.DataFrame:
    """Prior-only EWM of each team's efficiency (one row per team-game, chronological)."""
    t = tm.sort_values(["season", "week"]).reset_index(drop=True).copy()
    t["off_rz_rate"] = t.off_rz_tds / t.off_rz_trips.replace(0, np.nan)
    t["def_rz_rate"] = t.def_rz_tds / t.def_rz_trips.replace(0, np.nan)
    cols = OFF_M + DEF_M + ["off_rz_rate", "def_rz_rate"]
    e = _ewm_prior(t, "team", cols, half=half, prefix="e_")
    return pd.concat([t[["team", "season", "week"]], e], axis=1)


def add_team_metrics(df: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Join the player's team offense and the opponent's defense (priors) onto each row.
    Future weeks (no game played yet) get the latest prior via an extra row per team-week."""
    keys = df[["team", "season", "week"]].drop_duplicates()
    base = tm[["team", "season", "week"]]
    missing = keys.merge(base, how="left", indicator=True).query('_merge == "left_only"').drop(columns="_merge")
    t = pd.concat([tm, missing], ignore_index=True)  # NaN outcomes for future weeks → priors only
    pr = team_metric_priors(t)
    off = pr[["team", "season", "week"] + [f"e_{c}" for c in OFF_M] + ["e_off_rz_rate"]].rename(
        columns={**{f"e_{c}": f"t_{c}" for c in OFF_M}, "e_off_rz_rate": "t_rz_td"})
    de = pr[["team", "season", "week"] + [f"e_{c}" for c in DEF_M] + ["e_def_rz_rate"]].rename(
        columns={**{f"e_{c}": f"o_{c}" for c in DEF_M}, "e_def_rz_rate": "o_rz_td", "team": "opp"})
    df = df.merge(off, on=["team", "season", "week"], how="left")
    df = df.merge(de, on=["opp", "season", "week"], how="left")
    return df
