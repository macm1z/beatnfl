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


TOTAL_SD = 13.8   # spread of final game totals around the expected total (NFL, recent seasons)


def _phi(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def _odds_txt(v) -> str:
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return ""
    return f"+{v}" if v > 0 else str(v)


def _line_txt(team: str, handicap: float) -> str:
    if pd.isna(handicap):
        return ""
    return f"{team} PK" if abs(handicap) < 0.05 else f"{team} {handicap:+g}"


def games_data(season: int, week: int, raw_lines: pd.DataFrame | None = None, offline=False) -> pd.DataFrame:
    """One row per game: market lines/prices (Odds API median of books, else the nflverse schedule),
    the market's no-vig probabilities, and the model's numbers."""
    sched = nv.schedules(offline=offline)
    g = gm.run(sched)
    g = g[(g.season == season) & (g.week == week)].copy()
    sc = sched.set_index("game_id")
    for c in ("gameday", "gametime", "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds",
              "over_odds", "under_odds"):
        g[c] = g.game_id.map(sc[c]) if c in sc else np.nan
    g["src"] = "schedule"
    m = odds.market_summary(raw_lines) if raw_lines is not None else pd.DataFrame()
    if not m.empty:
        g = g.merge(m, on=["home", "away"], how="left")
        live = g.spread_nv.notna() | g.total.notna() | g.p_home_win.notna()
        g.loc[live, "src"] = "odds_api"
    for c in ("spread_nv", "home_spread_price", "away_spread_price", "p_home_cover", "ml_home", "ml_away", "p_home_win",
              "total", "over_price", "under_price", "p_over"):
        if c not in g:
            g[c] = np.nan
    # fall back to the schedule's (nflverse) lines and prices where the Odds API had nothing
    g["spread_nv"] = g.spread_nv.fillna(g.spread_line)
    g["total"] = g.total.fillna(g.total_line)
    g["home_spread_price"] = g.home_spread_price.fillna(g.home_spread_odds)
    g["away_spread_price"] = g.away_spread_price.fillna(g.away_spread_odds)
    g["ml_home"], g["ml_away"] = g.ml_home.fillna(g.home_moneyline), g.ml_away.fillna(g.away_moneyline)
    g["over_price"], g["under_price"] = g.over_price.fillna(g.over_odds), g.under_price.fillna(g.under_odds)

    def nv2(a, b):
        p = odds.no_vig(odds.implied_prob(a), odds.implied_prob(b))
        return np.nan if p is None else p
    g["p_home_cover"] = g.p_home_cover.fillna(pd.Series([nv2(a, b) for a, b in zip(g.home_spread_price, g.away_spread_price)], index=g.index))
    g["p_home_win"] = g.p_home_win.fillna(pd.Series([nv2(a, b) for a, b in zip(g.ml_home, g.ml_away)], index=g.index))
    g["p_over"] = g.p_over.fillna(pd.Series([nv2(a, b) for a, b in zip(g.over_price, g.under_price)], index=g.index))
    g["blend_spread"] = g.spread_nv + gm.BLEND_W * (g.model_spread - g.spread_nv)
    g["blend_total"] = g.total + gm.BLEND_W * (g.model_total - g.total)
    return g.reset_index(drop=True)


def games_table(season: int, week: int, raw_lines: pd.DataFrame | None = None, offline=False) -> pd.DataFrame:
    """Readable summary, one row per game, both teams' lines shown."""
    g = games_data(season, week, raw_lines, offline)

    def fav(home, away, m):
        if pd.isna(m):
            return ""
        return "Pick'em" if abs(m) < 0.05 else f"{home if m > 0 else away} -{abs(m):.1f}"
    kick = pd.to_datetime(g.gameday.astype(str) + " " + g.gametime.astype(str), errors="coerce")
    return pd.DataFrame({
        "Game": g.away + " @ " + g.home,
        "Kickoff (ET)": [k.strftime("%a %-I:%M %p") if pd.notna(k) else "" for k in kick],
        "Spread": [f"{_line_txt(a, s)} · {_line_txt(h, -s)}" if pd.notna(s) else "no line yet"
                   for a, h, s in zip(g.away, g.home, g.spread_nv)],
        "Moneyline": [f"{a} {_odds_txt(ma)} · {h} {_odds_txt(mh)}" if pd.notna(ma) and pd.notna(mh) else ""
                      for a, h, ma, mh in zip(g.away, g.home, g.ml_away, g.ml_home)],
        "Total": g.total,
        "Model spread": [fav(h, a, m) for h, a, m in zip(g.home, g.away, g.model_spread)],
        "Model total": g.model_total.round(1),
        "_src": g.src,
    })


def game_probs(season: int, week: int, raw_lines: pd.DataFrame | None = None, offline=False) -> pd.DataFrame:
    """One row per side of every spread, moneyline and total: the market's no-vig chance and the model's.

    The model's chance starts from the market's and moves by how far the (market-weighted) model
    number sits from the line, so a difference here is the model's disagreement only — not an
    artefact of a simple normal curve mis-pricing favourites."""
    g = games_data(season, week, raw_lines, offline)
    rows = []
    for r in g.itertuples():
        game = f"{r.away} @ {r.home}"
        if pd.notna(r.spread_nv) and pd.notna(r.p_home_cover):
            d = gm.cover_prob(r.blend_spread, r.spread_nv) - 0.5
            ph = float(np.clip(r.p_home_cover + d, 0.01, 0.99))
            rows += [(game, "Spread", _line_txt(r.home, -r.spread_nv), r.home_spread_price, r.p_home_cover, ph),
                     (game, "Spread", _line_txt(r.away, r.spread_nv), r.away_spread_price, 1 - r.p_home_cover, 1 - ph)]
        if pd.notna(r.p_home_win):
            sd = 13.4
            d = _phi(r.blend_spread / sd) - _phi(r.spread_nv / sd) if pd.notna(r.spread_nv) else 0.0
            ph = float(np.clip(r.p_home_win + d, 0.01, 0.99))
            rows += [(game, "Moneyline", f"{r.home} win", r.ml_home, r.p_home_win, ph),
                     (game, "Moneyline", f"{r.away} win", r.ml_away, 1 - r.p_home_win, 1 - ph)]
        if pd.notna(r.total) and pd.notna(r.p_over):
            d = _phi((r.blend_total - r.total) / TOTAL_SD) - 0.5
            po = float(np.clip(r.p_over + d, 0.01, 0.99))
            rows += [(game, "Total", f"Over {r.total:g}", r.over_price, r.p_over, po),
                     (game, "Total", f"Under {r.total:g}", r.under_price, 1 - r.p_over, 1 - po)]
    out = pd.DataFrame(rows, columns=["Game", "Bet type", "Side", "Odds", "Market %", "Model %"])
    out["Odds"] = pd.to_numeric(out.Odds, errors="coerce")
    out["Market %"] = (pd.to_numeric(out["Market %"]) * 100).round(1)
    out["Model %"] = (pd.to_numeric(out["Model %"]) * 100).round(1)
    out["Difference"] = (out["Model %"] - out["Market %"]).round(1)
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
    # the most likely outcome after correcting the forecast with its past errors — agrees with Model %
    b["model_median"] = [(pm.quantiles(s, pr, (0.5,))[0] if s != "any_td" and pd.notna(pr) else np.nan)
                         for s, pr in zip(b.stat, b.model_pred)]
    if not yes.empty and "player_anytime_td" in markets:
        y = yes.assign(book_p_over=yes.price.map(odds.implied_prob), model_p_over=yes.model_pred, line=0.5, under_price=np.nan)
        b = pd.concat([b[b.market != "player_anytime_td"], y], ignore_index=True)
    b["edge"] = (b.model_p_over - b.book_p_over).round(3)
    b["model_lean"] = np.where(b.edge > 0, "Over", np.where(b.edge < 0, "Under", ""))
    storage.save_quotes(b.rename(columns={}), "prop", season, week, model_version=getattr(pm, "version", ""))
    cols = ["game", "player", "stat", "book", "line", "price", "under_price", "model_pred", "model_median", "book_p_over",
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
    if "model_median" not in d:
        d["model_median"] = d.model_pred.where(d.stat != "any_td")
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
            model_median=("model_median", "first"),
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
        "Prop": [("Anytime TD*" if s == "any_td" else f"{STAT_NAMES.get(s, s)} {l:g}") for s, l in zip(d.stat, d.line)],
        "Book": d.Book.values,
        "Odds": pd.to_numeric(d.price, errors="coerce").round().values,
        "Model median": [(np.nan if s == "any_td" else m) for s, m in zip(d.stat, d.model_median)],
        "Model %": (d.model_p_over * 100).round(1).values,
        "Book %": (d.book_p_over * 100).round(1).values,
        "Difference": diff.round(1).values,
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
