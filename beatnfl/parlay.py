"""Parlay builder — combine legs from Props and Games, price the parlay at each of your sportsbooks.

A parlay pays the product of its legs' decimal odds, and only at one book, so every leg has to be
offered there at the same line. For each of your books this shows whether all legs are available,
the combined odds and payout, and the chances:
  · market chance = product of each leg's no-vig market chance (the fair price of the parlay)
  · model chance  = product of BeatNFL's chances for each leg
Both assume the legs are independent. Legs from the SAME game are related (a QB's yards and his
receiver's yards rise together), and books price same-game parlays with their own correlation model,
so the combined odds here are only a guide for those — the book's betslip shows the real price.
Every leg's betslip link is listed for the chosen book: open them one after another and they collect in
the book's betslip, where you pick "Parlay" and place it yourself. Information only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .research import STAT_NAMES, _dec, book_name
from .sources import odds


def american(dec: float) -> float:
    if not dec or np.isnan(dec) or dec <= 1:
        return np.nan
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def prop_leg(board: pd.DataFrame, key: str, side: str = "Over") -> dict | None:
    """A player-prop leg from the props board (key = 'game|player|stat'). Both sides' prices are kept so the
    side can be switched later."""
    game, player, stat = key.split("|", 2)
    g = board[(board.game == game) & (board.player == player) & (board.stat == stat)]
    if g.empty:
        return None
    ln = g.line.mode().iloc[0] if g.line.notna().any() else np.nan
    x = g[g.line == ln] if pd.notna(ln) else g
    quotes = [{"book": r.book, "line": r.line, "over": r.price, "under": getattr(r, "under_price", np.nan),
               "over_link": getattr(r, "link", None), "under_link": getattr(r, "under_link", None)} for r in x.itertuples()]
    td = stat == "any_td"
    return {"kind": "prop", "id": f"prop|{key}", "game": game, "player": player, "stat": stat, "line": ln,
            "side": "Yes" if td else (side if side in ("Over", "Under") else "Over"),
            "model_p_over": float(x.model_p_over.iloc[0]) if pd.notna(x.model_p_over.iloc[0]) else np.nan,
            "market_p_over": float(x.book_p_over.median()) if x.book_p_over.notna().any() else np.nan,
            "quotes": quotes}


def game_leg(row: dict, raw: pd.DataFrame | None) -> dict:
    """A spread / moneyline / total leg from one row of the Games chances table."""
    mk = {"Spread": "spreads", "Moneyline": "h2h", "Total": "totals"}[row["Bet type"]]
    parts = str(row["Side"]).split()
    quotes = []
    if raw is not None and not raw.empty:
        r = raw.assign(Game=raw.away + " @ " + raw.home)
        x = r[(r.Game == row["Game"]) & (r.market == mk) & (r.side.astype(str) == parts[0])]
        if mk != "h2h" and len(parts) > 1:
            want = 0.0 if parts[1] == "PK" else pd.to_numeric(parts[1], errors="coerce")
            exact = x[(x.line - want).abs() < 1e-9] if pd.notna(want) else x.iloc[0:0]
            if exact.empty and x.line.notna().any():
                exact = x[x.line == x.line.mode().iloc[0]]
            x = exact
        quotes = [{"book": q.book, "line": q.line, "over": q.price, "under": np.nan,
                   "over_link": getattr(q, "link", None), "under_link": None} for q in x.itertuples()]
    return {"kind": "game", "id": f"game|{row['Game']}|{row['Bet type']}|{row['Side']}", "game": row["Game"],
            "label": f"{row['Side']} ({row['Bet type'].lower()})", "side": "Bet",
            "model_p_over": row["Model %"] / 100 if pd.notna(row.get("Model %")) else np.nan,
            "market_p_over": row["Market %"] / 100 if pd.notna(row.get("Market %")) else np.nan, "quotes": quotes}


def _taking_over(leg: dict) -> bool:
    return leg["side"] in ("Over", "Yes", "Bet")


def leg_label(leg: dict) -> str:
    if leg["kind"] == "game":
        return leg["label"]
    if leg["stat"] == "any_td":
        return f"{leg['player']} anytime TD"
    return f"{leg['player']} {leg['side']} {leg['line']:g} {STAT_NAMES.get(leg['stat'], leg['stat']).lower()}"


def leg_probs(leg: dict) -> tuple[float, float]:
    """(model chance, market chance) of the leg's chosen side."""
    m, k = leg["model_p_over"], leg["market_p_over"]
    if _taking_over(leg):
        return m, k
    return (1 - m if pd.notna(m) else np.nan), (1 - k if pd.notna(k) else np.nan)


def legs_table(legs: list[dict]) -> pd.DataFrame:
    rows = []
    for lg in legs:
        m, k = leg_probs(lg)
        rows.append({"Leg": leg_label(lg), "Game": lg["game"], "Side": lg["side"], "Model %": round(100 * m, 1) if pd.notna(m) else np.nan,
                     "Market %": round(100 * k, 1) if pd.notna(k) else np.nan,
                     "Books": len({q["book"] for q in lg["quotes"] if pd.notna(q["over" if _taking_over(lg) else "under"])})})
    return pd.DataFrame(rows)


def price(legs: list[dict], books: list | None = None) -> pd.DataFrame:
    """One row per sportsbook: does it offer every leg, combined odds, payout on $10 and the chances."""
    if not legs:
        return pd.DataFrame()
    pm = np.prod([leg_probs(lg)[0] for lg in legs])
    pk = np.prod([leg_probs(lg)[1] for lg in legs])
    all_books = sorted({q["book"] for lg in legs for q in lg["quotes"]})
    if books:
        all_books = [b for b in all_books if b in books]
    rows = []
    rank = {b: i for i, b in enumerate(books or [])}
    for bk in all_books:
        decs, missing = [], []
        for lg in legs:
            q = next((q for q in lg["quotes"] if q["book"] == bk), None)
            p = q and q["over" if _taking_over(lg) else "under"]
            decs.append(_dec(p) if p is not None and pd.notna(p) else np.nan)
            if np.isnan(decs[-1]):
                missing.append(leg_label(lg))
        have = int(np.sum(~np.isnan(decs)))
        dec = float(np.prod(decs)) if have == len(legs) else np.nan
        rows.append({"book": bk, "Best": "", "Sportsbook": book_name(bk), "Legs offered": f"{have} of {len(legs)}",
                     "Not offered": "; ".join(missing),
                     "Parlay odds": american(dec), "Pays on $10": round(10 * dec, 2) if dec == dec else np.nan,
                     "Market chance %": round(100 * pk, 1) if pk == pk else np.nan,
                     "Model chance %": round(100 * pm, 1) if pm == pm else np.nan,
                     "Value vs market %": round(100 * (pk * dec - 1), 1) if dec == dec and pk == pk else np.nan,
                     "Value vs model %": round(100 * (pm * dec - 1), 1) if dec == dec and pm == pm else np.nan,
                     "_dec": round(dec, 6) if dec == dec else np.nan, "_o": rank.get(bk, len(rank))})
    out = pd.DataFrame(rows)
    out = out.sort_values(["_dec", "_o"], ascending=[False, True], na_position="last").reset_index(drop=True)
    if out._dec.notna().any():
        top = out._dec.max()
        n_top = int((out._dec == top).sum())
        out.loc[out._dec == top, "Best"] = "★ best" if n_top == 1 else "★ tied best"
    return out.drop(columns=["_dec", "_o"])


def slips(legs: list[dict], book: str) -> pd.DataFrame:
    """Each leg's betslip link at one book (None where the book gives no link or doesn't offer the leg)."""
    rows = []
    for lg in legs:
        q = next((q for q in lg["quotes"] if q["book"] == book), None)
        over = _taking_over(lg)
        p = q and q["over" if over else "under"]
        rows.append({"Leg": leg_label(lg), "Odds": p if p is not None and pd.notna(p) else np.nan,
                     "Slip": odds.bet_link(q and q["over_link" if over else "under_link"])})
    return pd.DataFrame(rows)


def same_game(legs: list[dict]) -> list[str]:
    """Games with more than one leg (their combined odds here ignore correlation)."""
    s = pd.Series([lg["game"] for lg in legs])
    return sorted(s[s.duplicated()].unique().tolist())
