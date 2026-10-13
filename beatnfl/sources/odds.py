"""The Odds API (v4) — game lines and player props, read-only.

Free tier: 500 credits/month. Cost = markets x regions per call; props are fetched
per game. Every pull is stored in the local database (research_quotes) with the
book's own timestamp so prop accuracy can be graded later at no extra cost.
"""

from __future__ import annotations

import pandas as pd

from ..errors import StepError
from ..names import team as norm_team
from ..net import fetch_json

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
STEP = "Odds (The Odds API)"
GAME_MARKETS = "h2h,spreads,totals"
PROP_MARKETS = {
    "player_pass_yds": "pass_yds", "player_pass_attempts": "pass_att", "player_pass_completions": "pass_cmp",
    "player_pass_tds": "pass_td", "player_rush_yds": "rush_yds", "player_rush_attempts": "rush_att",
    "player_receptions": "rec", "player_reception_yds": "rec_yds", "player_anytime_td": "any_td",
}
FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _need_key(key: str):
    if not key:
        raise StepError(STEP, "No Odds API key set.", "Lines and props come from The Odds API, which needs a free key.",
                        "Get one at the-odds-api.com (free tier), then paste it in Settings. DFS lineups work without it.")


def game_lines(key: str, offline=False) -> pd.DataFrame:
    _need_key(key)
    data, meta = fetch_json(f"{BASE}/odds", params={"apiKey": key, "regions": "us", "markets": GAME_MARKETS,
                                                     "oddsFormat": "american"}, step=STEP, max_age_h=1, offline=offline)
    rows = []
    for ev in data:
        h, a = FULL_TO_ABBR.get(ev["home_team"], ev["home_team"]), FULL_TO_ABBR.get(ev["away_team"], ev["away_team"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    side = FULL_TO_ABBR.get(o["name"], o["name"])
                    rows.append({"event_id": ev["id"], "commence": ev["commence_time"], "home": h, "away": a,
                                 "book": bk["key"], "market": mk["key"], "side": side, "line": o.get("point"),
                                 "price": o.get("price"), "source_ts": mk.get("last_update") or bk.get("last_update"),
                                 "retrieved_stale": meta.stale})
    return pd.DataFrame(rows)


def consensus(lines: pd.DataFrame) -> pd.DataFrame:
    """Median home spread and total per game across books → implied team totals."""
    if lines.empty:
        return pd.DataFrame(columns=["home", "away", "spread_home", "total"])
    sp = lines[(lines.market == "spreads") & (lines.side == lines.home)].groupby(["home", "away"])["line"].median()
    to = lines[(lines.market == "totals") & (lines.side == "Over")].groupby(["home", "away"])["line"].median()
    c = pd.concat([sp.rename("spread_home"), to.rename("total")], axis=1).reset_index()
    # Odds API spread is the home team's handicap (negative = favourite) → nflverse convention is -handicap
    c["spread_nv"] = -c.spread_home
    return c


def market_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Per game: consensus spread/total, median prices, and the market's no-vig probability for
    home cover, home win (moneyline) and over — median across books, each book's margin removed."""
    cols = ["home", "away", "spread_nv", "home_spread_price", "away_spread_price", "p_home_cover", "ml_home", "ml_away",
            "p_home_win", "total", "over_price", "under_price", "p_over", "books"]
    if lines is None or lines.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (h, a), g in lines.groupby(["home", "away"]):
        r = {"home": h, "away": a, "books": g.book.nunique()}
        sp = g[g.market == "spreads"]
        sp = sp.drop_duplicates(["book", "side"])
        hs, as_ = sp[sp.side == h].set_index("book"), sp[sp.side == a].set_index("book")
        if len(hs):
            line = hs.line.median()
            r["spread_nv"] = -line
            at = hs[hs.line == line].index.intersection(as_.index)
            r["home_spread_price"] = hs.loc[at, "price"].median() if len(at) else hs.price.median()
            r["away_spread_price"] = as_.loc[at, "price"].median() if len(at) else as_.price.median()
            ps = [no_vig(implied_prob(hs.at[b, "price"]), implied_prob(as_.at[b, "price"])) for b in at]
            r["p_home_cover"] = float(pd.Series(ps, dtype=float).median()) if ps else None
        ml = g[g.market == "h2h"].drop_duplicates(["book", "side"])
        hm, am = ml[ml.side == h].set_index("book").price, ml[ml.side == a].set_index("book").price
        both = hm.index.intersection(am.index)
        if len(both):
            r["ml_home"], r["ml_away"] = hm[both].median(), am[both].median()
            r["p_home_win"] = float(pd.Series([no_vig(implied_prob(hm[b]), implied_prob(am[b])) for b in both]).median())
        to = g[g.market == "totals"].drop_duplicates(["book", "side"])
        ov, un = to[to.side == "Over"].set_index("book"), to[to.side == "Under"].set_index("book")
        if len(ov):
            line = ov.line.median()
            r["total"] = line
            at = ov[ov.line == line].index.intersection(un.index)
            r["over_price"] = ov.loc[at, "price"].median() if len(at) else ov.price.median()
            r["under_price"] = un.loc[at, "price"].median() if len(at) else None
            ps = [no_vig(implied_prob(ov.at[b, "price"]), implied_prob(un.at[b, "price"])) for b in at]
            r["p_over"] = float(pd.Series(ps, dtype=float).median()) if ps else None
        out.append(r)
    return pd.DataFrame(out).reindex(columns=cols)


def events(key: str, offline=False) -> list[dict]:
    _need_key(key)
    data, _ = fetch_json(f"{BASE}/events", params={"apiKey": key}, step=STEP, max_age_h=3, offline=offline)
    return data


def props(key: str, event_id: str, markets: list[str] | None = None, offline=False) -> pd.DataFrame:
    _need_key(key)
    mk = ",".join(markets or PROP_MARKETS)
    data, meta = fetch_json(f"{BASE}/events/{event_id}/odds",
                            params={"apiKey": key, "regions": "us", "markets": mk, "oddsFormat": "american"},
                            step=STEP, max_age_h=3, offline=offline)
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                rows.append({"event_id": event_id, "book": bk["key"], "market": m["key"],
                             "stat": PROP_MARKETS.get(m["key"], m["key"]), "player": o.get("description"),
                             "side": o.get("name"), "line": o.get("point"), "price": o.get("price"),
                             "source_ts": m.get("last_update"), "retrieved_stale": meta.stale})
    return pd.DataFrame(rows)


def implied_prob(american) -> float | None:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def no_vig(p_over: float | None, p_under: float | None) -> float | None:
    if p_over is None or p_under is None:
        return None
    return p_over / (p_over + p_under)
