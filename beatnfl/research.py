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


def matchups(season: int, week: int, offline=False, progress=lambda m: None) -> pd.DataFrame:
    """Team efficiency going into the week (prior games only, recent games weighted more),
    offense vs the opponent's defense. From free NFL play-by-play (nflverse)."""
    from .models.features import team_metric_priors
    sched = nv.schedules(offline=offline)
    wk = sched[(sched.season == season) & (sched.week == week) & (sched.game_type == "REG")]
    if wk.empty:
        raise StepError("Matchups", f"No games for {season} week {week}.", "", "Pick another week.")
    tm = nv.team_metrics(range(season - 1, season + 1), offline=offline, progress=progress)
    have = set(tm[(tm.season == season) & (tm.week == week)].team)
    fut = pd.DataFrame([{"team": t, "season": season, "week": week}
                        for t in (set(wk.home_team) | set(wk.away_team)) - have])
    pr = team_metric_priors(pd.concat([tm, fut], ignore_index=True))
    pr = pr[(pr.season == season) & (pr.week == week)].drop_duplicates("team").set_index("team")
    rows = []
    for g in wk.itertuples():
        for t, o, side in ((g.away_team, g.home_team, "@"), (g.home_team, g.away_team, "vs")):
            if t not in pr.index or o not in pr.index:
                continue
            a, d = pr.loc[t], pr.loc[o]
            rows.append({
                "Team": t, "Opp": f"{side} {o}",
                "Off EPA/play": a.e_off_epa, "Opp def EPA allowed": d.e_def_epa,
                "Pass edge": a.e_off_pass_epa + d.e_def_pass_epa, "Run edge": a.e_off_rush_epa + d.e_def_rush_epa,
                "Success rate": a.e_off_sr, "Explosive %": a.e_off_explosive * 100,
                "Pass rate over exp. %": a.e_off_proe, "Red zone TD %": a.e_off_rz_rate * 100,
                "Sack rate %": a.e_off_sack_rate * 100, "Opp sack rate %": d.e_def_sack_rate * 100,
                "Plays/game": a.e_off_plays,
            })
    return pd.DataFrame(rows).round(3)


# ------------------------------------------------------------------ simplified props view
BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
              "fanatics": "Fanatics", "espnbet": "ESPN BET", "betrivers": "BetRivers", "ballybet": "Bally Bet",
              "hardrockbet": "Hard Rock", "betonlineag": "BetOnline", "bovada": "Bovada", "mybookieag": "MyBookie",
              "lowvig": "LowVig", "betus": "BetUS", "fliff": "Fliff", "pointsbetus": "PointsBet", "wynnbet": "WynnBET",
              "unibet_us": "Unibet", "superbook": "SuperBook", "betparx": "betPARX", "espnbet_us": "ESPN BET"}
STAT_NAMES = {"pass_yds": "Pass yds", "pass_att": "Pass att", "pass_cmp": "Completions", "pass_td": "Pass TDs",
              "rush_yds": "Rush yds", "rush_att": "Rush att", "rec": "Receptions", "rec_yds": "Rec yds", "any_td": "Anytime TD"}
CONSENSUS = "All books (consensus)"


def book_name(key: str) -> str:
    return BOOK_NAMES.get(key, str(key).replace("_", " ").title())


def _odds(v) -> str:
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return ""
    return f"+{v}" if v > 0 else str(v)


def props_view(b: pd.DataFrame, game: str | None = None, book: str = CONSENSUS, sort: str = "Biggest difference") -> pd.DataFrame:
    """One readable row per player/prop (consensus across books) or per book.

    Over/under props: book % is no-vig (over and under prices paired).
    Anytime TD: books quote only "Yes", so book % includes the book's margin — it runs high.
    A TD prop is flagged only when the model is above the book (a 'No' lean would mostly be the margin)."""
    if b is None or b.empty:
        return pd.DataFrame()
    d = b.copy()
    if game and game != "All games":
        d = d[d.game == game]
    d["Book"] = d.book.map(book_name)
    if book != CONSENSUS:
        d = d[d.Book == book]
    if d.empty:
        return pd.DataFrame()
    td = d.stat == "any_td"
    if book == CONSENSUS:
        g = d.groupby(["game", "player", "stat"], as_index=False).agg(
            line=("line", "median"), price=("price", "median"), model_pred=("model_pred", "first"),
            model_p_over=("model_p_over", "first"), book_p_over=("book_p_over", "median"), books=("book", "nunique"))
        g["Book"] = g.books.map(lambda n: f"{n} book" + ("s" if n != 1 else ""))
        d, td = g, g.stat == "any_td"
    diff = (d.model_p_over - d.book_p_over) * 100
    strong, lean = diff.abs() >= 8, diff.abs() >= 4

    def label(i):
        if pd.isna(diff[i]):
            return "no forecast"
        if td[i]:
            return ("Yes ★" if diff[i] >= 8 else "Yes") if diff[i] >= 4 else "—"
        if not lean[i]:
            return "—"
        side = "Over" if diff[i] > 0 else "Under"
        return side + (" ★" if strong[i] else "")
    out = pd.DataFrame({
        "Game": d.game.values,
        "Player": d.player.values,
        "Prop": [("Anytime TD" if s == "any_td" else f"{STAT_NAMES.get(s, s)} {l:g}") for s, l in zip(d.stat, d.line)],
        "Book": d.Book.values,
        "Odds": [_odds(p) for p in d.price],
        "Model": [("" if s == "any_td" or pd.isna(m) else f"{m:.1f}") for s, m in zip(d.stat, d.model_pred)],
        "Model %": [("" if pd.isna(p) else f"{p * 100:.0f}%") for p in d.model_p_over],
        "Book %": [("" if pd.isna(p) else f"{p * 100:.0f}%" + (" (incl. margin)" if s == "any_td" else ""))
                   for p, s in zip(d.book_p_over, d.stat)],
        "Lean": [label(i) for i in d.index],
        "_diff": diff.abs().values, "_book": d.Book.values,
    })
    if sort == "Sportsbook":
        out = out.sort_values(["_book", "Game", "Player"])
    elif sort == "Player":
        out = out.sort_values(["Player", "Prop", "_book"])
    else:
        out = out.sort_values("_diff", ascending=False, na_position="last")
    return out.drop(columns=["_diff", "_book"]).reset_index(drop=True)
