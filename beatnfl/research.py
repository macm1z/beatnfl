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
    from .config import local_tz
    kick = pd.to_datetime(g.gameday.astype(str) + " " + g.gametime.astype(str), errors="coerce")
    kick = kick.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(local_tz())
    return pd.DataFrame({
        "Game": g.away + " @ " + g.home,
        "Kickoff": [k.strftime("%a %-I:%M %p") if pd.notna(k) else "" for k in kick],
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
        storage.save_prop_prices(p)  # both sides, every pull — closing prices are read from these
        p = match_props(p, fc, h, a)
        p["game"] = f"{a} @ {h}"
        p["commence"] = e.get("commence_time", "")
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
            "model_p_over", "edge", "model_lean", "source_ts", "event_id", "commence", "market"]
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


# ------------------------------------------------------------------ FantasyPros stat lines in forecasts
def blend_fp_stats(fc: pd.DataFrame, fp: pd.DataFrame, weights: dict) -> tuple[pd.DataFrame, int]:
    """Mix FantasyPros' projected stats into the model's stat forecasts, by the tested share per stat
    (Evidence → FantasyPros vs Sleeper vs model). Players FantasyPros doesn't list keep the model's number.
    Returns (forecasts, players blended)."""
    if fc is None or fc.empty or fp is None or fp.empty or not weights:
        return fc, 0
    from .names import match_key
    f = fc.copy()
    f["_key"] = [match_key(n, p, t) for n, p, t in zip(f.name, f.position, f.team)]
    src = fp.drop_duplicates("key").set_index("key")
    hit = f._key.isin(src.index)
    for stat, w in weights.items():
        col, fcol = f"p_{stat}", f"fp_{stat}"
        if not w or col not in f or fcol not in src:
            continue
        v = pd.to_numeric(f.loc[hit, "_key"].map(src[fcol]), errors="coerce")
        ok = v.notna() & f.loc[hit, col].notna()
        idx = v[ok].index
        f.loc[idx, col] = (1 - w) * f.loc[idx, col] + w * v[ok]
    f["fp_blended"] = hit
    return f.drop(columns="_key"), int(hit.sum())


# ------------------------------------------------------------------ midweek picks + closing-line grading
# The rule being tested forward. 2025 backtest on real lines (one prop per player-game, 2,472 props):
# no edge overall; the only hint was Unders where the model sits 10+ points below the market
# (53.8% won, ROI +2.1% ± 4.6 — not proven). Picks therefore log that rule so it builds evidence.
MIN_DIFF = 10.0
PICK_SIDES = ("Under",)


def _dec(american) -> float:
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def midweek_picks(board: pd.DataFrame, fp: pd.DataFrame | None = None, min_diff: float = MIN_DIFF,
                  need_fp: bool = True, sides=PICK_SIDES) -> pd.DataFrame:
    """Props where the model differs from the market by min_diff+ points and (when FantasyPros numbers
    are available) FantasyPros points the same way. One row per prop with the best price across books.
    Information only — a research list to track against the closing line, not betting advice."""
    if board is None or board.empty:
        return pd.DataFrame()
    b = board.copy()
    fpi = None
    if fp is not None and not fp.empty:
        f = fp[fp.pos != "DST"].assign(nn=lambda d: d.name.map(norm_name)).drop_duplicates("nn", keep=False)
        fpi = f.set_index("nn")
    rows = []
    for (game, player, stat, line), g in b.groupby(["game", "player", "stat", "line"], dropna=False):
        mp = g.model_p_over.iloc[0]
        bp = float(g.book_p_over.median())
        if pd.isna(mp) or pd.isna(bp):
            continue
        diff = (mp - bp) * 100
        td = stat == "any_td"
        if abs(diff) < min_diff or (td and diff < 0):
            continue                       # a TD "No" lean is mostly the book's one-sided margin
        side = ("Yes" if td else "Over") if diff > 0 else "Under"
        if sides and side not in sides:
            continue
        # FantasyPros direction
        fp_val, agrees = np.nan, None
        if fpi is not None:
            nn = norm_name(player)
            col = f"fp_{stat}"
            if nn in fpi.index and col in fpi.columns and pd.notna(fpi.at[nn, col]):
                fp_val = float(fpi.at[nn, col])
                ref = bp if td else line
                agrees = (fp_val > ref) if side in ("Over", "Yes") else (fp_val < ref)
        if need_fp and fpi is not None and not agrees:
            continue
        # best price for the side across books at this exact line
        prices = g.price if side in ("Over", "Yes") else g.under_price
        ok = prices.notna()
        if not ok.any():
            continue
        best_i = prices[ok].map(_dec).idxmax()
        price = float(prices[best_i])
        m_side = bp if side in ("Over", "Yes") else 1 - bp
        mo_side = mp if side in ("Over", "Yes") else 1 - mp
        dec = _dec(price)
        rows.append({"game": game, "event_id": g.event_id.iloc[0] if "event_id" in g else "",
                     "commence": g.commence.iloc[0] if "commence" in g else "", "player": player, "stat": stat,
                     "market": g.market.iloc[0] if "market" in g else "", "line": line, "side": side,
                     "book": g.book[best_i], "price": price, "market_p": round(m_side, 4), "model_p": round(mo_side, 4),
                     "fp_proj": fp_val, "model_median": g.model_median.iloc[0] if "model_median" in g else np.nan,
                     "fp_agrees": agrees, "edge_market": round((m_side * dec - 1) * 100, 1),
                     "edge_model": round((mo_side * dec - 1) * 100, 1), "diff": round(abs(diff), 1),
                     "source": "model+FantasyPros" if agrees else "model"})
    out = pd.DataFrame(rows)
    return out.sort_values("diff", ascending=False).reset_index(drop=True) if not out.empty else out


def picks_view(p: pd.DataFrame) -> pd.DataFrame:
    """Readable midweek list."""
    if p is None or p.empty:
        return pd.DataFrame()
    from .config import local_tz
    kick = pd.to_datetime(p.commence, utc=True, errors="coerce").dt.tz_convert(local_tz())
    return pd.DataFrame({
        "Game": p.game, "Kickoff": [k.strftime("%a %-I:%M %p") if pd.notna(k) else "" for k in kick],
        "Player": p.player,
        "Prop": [("Anytime TD" if s == "any_td" else f"{STAT_NAMES.get(s, s)} {l:g}") for s, l in zip(p.stat, p.line)],
        "Side": p.side, "Best odds": p.price.round(), "Book": p.book.map(book_name),
        "Market %": (p.market_p * 100).round(1), "Model %": (p.model_p * 100).round(1),
        "FantasyPros": p.fp_proj.round(2) if "fp_proj" in p else np.nan,
        "Model median": p.model_median.round(1) if "model_median" in p else np.nan,
        "Edge at best price %": p.edge_market,
    })


def capture_close(key: str, within_h: float = 4.0, offline=False, progress=lambda m: None) -> dict:
    """Pull current prices for logged picks whose games start within `within_h` hours (the closing prices).
    Costs one Odds API credit per market per game, only for games/markets with picks."""
    pk = storage.picks()
    if pk.empty:
        return {"games": 0, "credits": 0}
    now = pd.Timestamp.now(tz="UTC")
    k = pd.to_datetime(pk.commence, utc=True, errors="coerce")
    due = pk[(k > now) & (k <= now + pd.Timedelta(hours=within_h))]
    credits = 0
    for ev, g in due.groupby("event_id"):
        mk = sorted(set(g.market.dropna()))
        if not ev or not mk:
            continue
        progress(f"Closing prices: {g.game.iloc[0]}")
        p = odds.props(key, ev, mk, offline=offline)
        storage.save_prop_prices(p)
        credits += len(mk)
    return {"games": int(due.event_id.nunique()), "credits": credits}


def _close_prob(prices: pd.DataFrame, r) -> float:
    """Market's no-vig chance for the pick's side from the last prices seen before kickoff."""
    if prices.empty:
        return np.nan
    q = prices[(prices.event_id == r.event_id) & (prices.player == r.player) & (prices.market == r.market)]
    if r.stat != "any_td":
        q = q[np.isclose(q.line.astype(float), float(r.line))]
    q = q[pd.to_datetime(q.retrieved_utc, utc=True) < pd.to_datetime(r.commence, utc=True)]
    if q.empty:
        return np.nan
    last = q.sort_values("retrieved_utc").groupby(["book", "side"]).tail(1)
    if r.stat == "any_td":
        y = last[last.side == "Yes"].price.map(odds.implied_prob)
        return float(y.median()) if len(y) else np.nan
    pv = last.pivot_table(index="book", columns="side", values="price", aggfunc="last")
    if not {"Over", "Under"} <= set(pv.columns):
        return np.nan
    po = [odds.no_vig(odds.implied_prob(o), odds.implied_prob(u)) for o, u in zip(pv.Over, pv.Under) if pd.notna(o) and pd.notna(u)]
    if not po:
        return np.nan
    p_over = float(np.median(po))
    return p_over if r.side in ("Over", "Yes") else 1 - p_over


def grade_picks(offline=False) -> tuple[pd.DataFrame, dict]:
    """Every logged pick: result once the game is final, and closing-line value (CLV) — whether the
    logged price beat the market's last pre-kickoff price. CLV is the fast, honest test of an edge."""
    pk = storage.picks()
    if pk.empty:
        return pd.DataFrame(), {"n": 0}
    prices = storage.prop_prices(sorted(set(pk.event_id.dropna())))
    started = pd.to_datetime(pk.commence, utc=True, errors="coerce") <= pd.Timestamp.now(tz="UTC")
    # the close is only known once the game has started; before that the latest price is not a close
    pk["close_p"] = [_close_prob(prices, r) if go else np.nan for r, go in zip(pk.itertuples(), started)]
    pk["taken_p"] = pk.price.map(odds.implied_prob)
    pk["clv"] = ((pk.close_p - pk.taken_p) * 100).round(1)   # + = your price was better than the close
    # results
    seasons = sorted(set(pk.season.astype(int)))
    try:
        from .models.features import tidy_stats
        t = tidy_stats(nv.weekly_stats(seasons, offline=offline)).assign(nn=lambda d: d.name.map(norm_name))
    except StepError:
        t = pd.DataFrame()
    pk["nn"] = pk.player.map(norm_name)
    res = []
    for r in pk.itertuples():
        hit = t[(t.season == r.season) & (t.week == r.week) & (t.nn == r.nn)] if not t.empty else t
        if hit.empty or r.stat not in hit:
            res.append((np.nan, ""))
            continue
        a = float(hit[r.stat].iloc[0])
        if r.stat == "any_td":
            won = a > 0
            res.append((a, "won" if won else "lost"))
            continue
        if a == r.line:
            res.append((a, "push"))
        else:
            res.append((a, "won" if (a > r.line) == (r.side == "Over") else "lost"))
    pk["actual"], pk["result"] = [x[0] for x in res], [x[1] for x in res]
    pk["units"] = [(_dec(p) - 1) if rs == "won" else (-1.0 if rs == "lost" else 0.0) for p, rs in zip(pk.price, pk.result)]
    done = pk[pk.result != ""]
    cl = pk[pk.clv.notna()]
    summary = {"n": len(pk), "graded": len(done), "won": int((done.result == "won").sum()),
               "lost": int((done.result == "lost").sum()), "push": int((done.result == "push").sum()),
               "units": round(float(done.units.sum()), 2), "roi": round(float(done.units.mean() * 100), 1) if len(done) else None,
               "clv_n": len(cl), "clv_avg": round(float(cl.clv.mean()), 2) if len(cl) else None,
               "beat_close": round(float((cl.clv > 0).mean() * 100), 1) if len(cl) else None}
    return pk.drop(columns=["nn"]), summary


def build_midweek(settings, season: int | None = None, week: int | None = None, markets=None, offline=False,
                  progress=lambda m: None) -> tuple[pd.DataFrame, dict]:
    """The whole midweek routine: forecasts (+ FantasyPros stat mix if tested), props from every book,
    flagged picks, logged with their best price. Used by the Picks tab and the scheduled Mac check."""
    from .models import train
    from .sources import sleeper
    if not settings.odds_api_key:
        raise StepError("Midweek picks", "No Odds API key set.", "Props come from The Odds API.",
                        "Paste your Odds API key in Settings.")
    sched = nv.schedules(offline=offline)
    if season is None or week is None:
        season, week = nv.upcoming_week(sched)
    progress("Forecasting players…")
    pm, dm, meta = train.get_models(offline=offline)
    try:
        spp, _ = sleeper.players(offline=offline)
    except StepError:
        spp = None
    lines = None
    try:
        lines = odds.consensus(odds.game_lines(settings.odds_api_key, offline=offline))
    except StepError:
        pass
    fc, _ = train.forecast_week(season, week, spp, pm, lines, offline=offline)
    fp, fp_note = None, "no FantasyPros key — model only"
    if settings.fantasypros_api_key:
        try:
            from .sources import fantasypros
            fp, _ = fantasypros.projections(settings.fantasypros_api_key, season, week, offline=offline)
            fp_note = "FantasyPros must agree"
            if settings.stat_weights:
                fc, _ = blend_fp_stats(fc, fp, settings.stat_weights)
        except StepError as e:
            fp_note = "FantasyPros unavailable (" + e.what + ") — model only"
    progress("Pulling props from every book…")
    board, info = props_board(settings.odds_api_key, season, week, fc, pm, markets or CORE_PROP_MARKETS,
                              offline=offline, progress=progress)
    picks = midweek_picks(board, fp)
    if not picks.empty:
        picks = picks.assign(season=season, week=week)
        info["new"] = storage.save_picks(picks)
    else:
        info["new"] = 0
    info.update({"season": season, "week": week, "fp_note": fp_note, "n": len(picks)})
    return picks, info
