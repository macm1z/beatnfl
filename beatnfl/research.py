"""Research area: games and player props. Information only — nothing here places,
recommends sizing for, or automates a wager.

Every prop/line pulled is stored with the book's timestamp and the model's frozen
prediction, so grade_props() can later show how the model did against real lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .errors import StepError
from .models import games as gm
from .names import norm_name
from .sources import nflverse as nv
from .sources import odds

CORE_PROP_MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions",
                     "player_anytime_td"]


def games_table(season: int, week: int, lines: pd.DataFrame | None = None, offline=False) -> pd.DataFrame:
    sched = nv.schedules(offline=offline)
    g = gm.run(sched)
    g = g[(g.season == season) & (g.week == week)].copy()
    if lines is not None and not lines.empty:
        c = lines.rename(columns={"spread_nv": "mkt_spread", "total": "mkt_total"})[["home", "away", "mkt_spread", "mkt_total"]]
        g = g.merge(c, on=["home", "away"], how="left")
        g["line_source"] = np.where(g.mkt_spread.notna(), "Odds API (median of books)", "nflverse schedule")
        g["mkt_spread"] = g.mkt_spread.fillna(g.spread_line)
        g["mkt_total"] = g.mkt_total.fillna(g.total_line)
    else:
        g["mkt_spread"], g["mkt_total"], g["line_source"] = g.spread_line, g.total_line, "nflverse schedule"
    g["blend_spread"] = g.mkt_spread + gm.BLEND_W * (g.model_spread - g.mkt_spread)
    g["blend_total"] = g.mkt_total + gm.BLEND_W * (g.model_total - g.mkt_total)
    g["p_home_cover"] = [gm.cover_prob(m, l) if pd.notna(l) else np.nan for m, l in zip(g.blend_spread, g.mkt_spread)]

    def fav(r):
        if pd.isna(r.mkt_spread):
            return "no line yet"
        t = r.home if r.mkt_spread > 0 else r.away
        return f"{t} -{abs(r.mkt_spread):g}"
    out = pd.DataFrame({
        "Game": g.away + " @ " + g.home, "Market": [fav(r) for r in g.itertuples()],
        "Market total": g.mkt_total, "Model margin (home)": g.model_spread.round(1),
        "Model total": g.model_total.round(1), "Blend margin": g.blend_spread.round(1),
        "Blend total": g.blend_total.round(1), "P(home covers)": g.p_home_cover.round(3), "Lines from": g.line_source,
    })
    return out


def match_props(props: pd.DataFrame, fc: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Attach model forecasts to prop quotes (player names matched within the two teams)."""
    if props.empty:
        return props
    f = fc[fc.team.isin([home, away])].assign(nn=lambda d: d.name.map(norm_name))
    f = f.drop_duplicates("nn", keep=False).set_index("nn")
    props = props.copy()
    props["nn"] = props.player.map(norm_name)
    props["model_pred"] = np.nan
    props["model_name"] = ""
    for i, r in props.iterrows():
        if r.nn in f.index:
            col = f"p_{r.stat}"
            props.at[i, "model_name"] = f.at[r.nn, "name"]
            if col in f.columns:
                props.at[i, "model_pred"] = f.at[r.nn, col]
    return props


def props_board(key: str, season: int, week: int, fc: pd.DataFrame, pm, markets=None, event_ids=None,
                offline=False, progress=lambda m: None) -> tuple[pd.DataFrame, dict]:
    """One row per player/market/book with model vs market. Costs ~len(markets) credits per game."""
    evs = odds.events(key, offline=offline)
    if event_ids:
        evs = [e for e in evs if e["id"] in event_ids]
    markets = markets or CORE_PROP_MARKETS
    rows, info = [], {"games": 0, "credits_est": 0}
    for e in evs:
        h = odds.FULL_TO_ABBR.get(e["home_team"], e["home_team"])
        a = odds.FULL_TO_ABBR.get(e["away_team"], e["away_team"])
        if not ((fc.team == h).any() or (fc.team == a).any()):
            continue
        progress(f"Props: {a} @ {h}")
        p = odds.props(key, e["id"], markets, offline=offline)
        info["games"] += 1
        info["credits_est"] += len(markets)
        if p.empty:
            continue
        p = match_props(p, fc, h, a)
        p["game"] = f"{a} @ {h}"
        rows.append(p)
    if not rows:
        raise StepError("Props", "No prop markets were returned for this week's games yet.",
                        "Books usually post most player props 2–4 days before kickoff.", "Try again closer to the games.")
    P = pd.concat(rows, ignore_index=True)
    # pair over/under per book+player+market+line
    ov = P[P.side == "Over"]
    un = P[P.side == "Under"][["book", "market", "player", "line", "price"]].rename(columns={"price": "under_price"})
    yes = P[(P.market == "player_anytime_td") & (P.side.isin(["Yes", "Over"]))]
    b = ov.merge(un, on=["book", "market", "player", "line"], how="left")
    b["book_p_over"] = [odds.no_vig(odds.implied_prob(o), odds.implied_prob(u)) if pd.notna(u) else odds.implied_prob(o)
                        for o, u in zip(b.price, b.under_price)]
    b["model_p_over"] = [pm.prob_over(s, pr, ln) if s != "any_td" else pr
                         for s, pr, ln in zip(b.stat, b.model_pred, b.line)]
    if not yes.empty and "player_anytime_td" in markets:
        y = yes.assign(book_p_over=yes.price.map(odds.implied_prob), model_p_over=yes.model_pred, line=0.5, under_price=np.nan)
        b = pd.concat([b[b.market != "player_anytime_td"], y], ignore_index=True)
    b["edge"] = (b.model_p_over - b.book_p_over).round(3)
    b["model_lean"] = np.where(b.edge > 0, "Over", np.where(b.edge < 0, "Under", ""))
    storage.save_quotes(b.rename(columns={}), "prop", season, week, model_version=getattr(pm, "version", ""))
    cols = ["game", "player", "stat", "book", "line", "price", "under_price", "model_pred", "book_p_over",
            "model_p_over", "edge", "model_lean", "source_ts"]
    return b[cols].sort_values(["game", "player", "stat", "book"]).reset_index(drop=True), info


def grade_props(offline=False) -> dict:
    """Prospective grading of every stored prop quote whose game is final.

    Only quotes retrieved before kickoff with a frozen model probability count.
    Reports Brier score for the model vs the book's no-vig probability and the hit
    rate of the side the model leaned to. Paper results only."""
    q = storage.quotes("prop")
    if q.empty:
        return {"n": 0, "note": "No prop quotes stored yet — open the Props board on game weeks to start building history."}
    q = q[q.model_p_over.notna()]
    seasons = sorted(set(q.season.astype(int)))
    st = nv.weekly_stats(seasons, offline=offline)
    from .models.features import tidy_stats
    t = tidy_stats(st).assign(nn=lambda d: d.name.map(norm_name))
    q = q.assign(nn=q.player.map(norm_name))
    m = q.merge(t[["season", "week", "nn", "pass_yds", "pass_att", "pass_cmp", "pass_td", "rush_yds", "rush_att", "rec",
                   "rec_yds", "any_td"]], on=["season", "week", "nn"], how="inner")
    if m.empty:
        return {"n": 0, "note": "Stored quotes have no final results yet."}
    m["actual"] = [getattr(r, r.stat) if r.stat in m.columns else np.nan for r in m.itertuples()]
    m = m[m.actual.notna()]
    m["over"] = (m.actual > m.line).astype(float)
    m = m.drop_duplicates(["season", "week", "nn", "stat", "book", "line"])
    from .sources.odds import implied_prob
    m["book_p"] = m.price.map(implied_prob)
    lean_over = m.model_p_over > m.book_p
    hit = np.where(lean_over, m.over == 1, m.over == 0)
    return {"n": len(m), "weeks": int(m.groupby(["season", "week"]).ngroups),
            "brier_model": round(float(((m.model_p_over - m.over) ** 2).mean()), 4),
            "brier_book": round(float(((m.book_p - m.over) ** 2).mean()), 4),
            "lean_hit_rate": round(float(hit.mean()), 3),
            "note": "Book probability here is the single-side implied price (includes vig) when the opposite side was not stored."}
