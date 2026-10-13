"""Sportsbook prop lines as projection information (step 4d).

The books' player lines already price in injury news, role changes and sharp money. Tested on 2025
(model trained only on earlier seasons, real midweek lines, weeks 9–15 unseen while fitting):

  stat         average miss: model   line   (line weight that fit best)
  pass yds                   58.7    54.5    0.65
  rush yds                   20.4    19.5    1.00
  rec yds                    22.7    21.7    0.80
  DK points (players with a line):  6.03 → 5.95 with the yardage lines folded in; ranking correlation .405 → .416

So the model's DraftKings projection is moved toward the books' lines: for each stat with a line,
  adjusted = model + weight × points_per_unit × ((line + shift) − model's stat forecast)
`shift` turns the line (a median — half the time over, half under) into an average, which is what
DraftKings points need: actual minus line averaged +2.8 pass yds, +1.5 rush yds, +4.7 rec yds.

Receptions and anytime-TD lines weren't in the 2025 test (not downloaded), so they get half weight
until the live grading shows more. Information only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .names import norm_name
from .sources import odds

MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions", "player_anytime_td"]
PTS = {"pass_yds": 0.04, "rush_yds": 0.1, "rec_yds": 0.1, "rec": 1.0}
WEIGHT = {"pass_yds": 0.8, "rush_yds": 0.8, "rec_yds": 0.8, "rec": 0.5, "any_td": 0.5}
SHIFT = {"pass_yds": 2.8, "rush_yds": 1.5, "rec_yds": 4.7, "rec": 0.0}
TD_MARGIN = 0.92   # anytime-TD "Yes" prices carry the book's margin on one side only — shave it off


def slate_lines(key: str, pool: pd.DataFrame, offline=False, progress=lambda m: None) -> tuple[pd.DataFrame, dict]:
    """Consensus prop lines for slate players: one row per dk_id with line_<stat> columns and p_td.
    Costs 5 Odds API credits per game; responses are reused for 3 hours."""
    teams = set(pool.team.dropna())
    evs = [e for e in odds.events(key, offline=offline)
           if odds.FULL_TO_ABBR.get(e["home_team"]) in teams and odds.FULL_TO_ABBR.get(e["away_team"]) in teams]
    frames = []
    for e in evs:
        h, a = odds.FULL_TO_ABBR[e["home_team"]], odds.FULL_TO_ABBR[e["away_team"]]
        progress(f"Prop lines: {a} @ {h}")
        p = odds.props(key, e["id"], MARKETS, offline=offline)
        if not p.empty:
            frames.append(p.assign(home=h, away=a))
    info = {"games": len(evs), "games_with_props": len(frames), "credits_est": len(evs) * len(MARKETS)}
    if not frames:
        return pd.DataFrame(columns=["dk_id"]), info
    P = pd.concat(frames, ignore_index=True)
    P["nn"] = P.player.map(norm_name)
    rows = {}
    for (nn, home, away, stat), g in P.groupby(["nn", "home", "away", "stat"]):
        if stat == "any_td":
            y = g[g.side.isin(["Yes", "Over"])].price.map(odds.implied_prob).dropna()
            if len(y):
                rows.setdefault((nn, home, away), {})["p_td"] = float(y.median()) * TD_MARGIN
        elif stat in PTS:
            ln = g[g.side == "Over"].line.dropna()
            if len(ln):
                rows.setdefault((nn, home, away), {})[f"line_{stat}"] = float(ln.median())
                rows[(nn, home, away)]["books"] = max(rows[(nn, home, away)].get("books", 0), int(g.book.nunique()))
    L = pd.DataFrame([{"nn": k[0], "home": k[1], "away": k[2], **v} for k, v in rows.items()])
    if L.empty:
        return pd.DataFrame(columns=["dk_id"]), info
    sl = pool[["dk_id", "name", "team"]].assign(nn=lambda d: d.name.map(norm_name))
    m = []
    for r in L.itertuples(index=False):
        hit = sl[(sl.nn == r.nn) & sl.team.isin([r.home, r.away])]
        if len(hit) == 1:
            m.append({"dk_id": hit.dk_id.iloc[0], **{k: v for k, v in r._asdict().items() if k.startswith(("line_", "p_td"))}})
    out = pd.DataFrame(m)
    info["players"] = len(out)
    return out, info


def adjust(res: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Move the model's DK projection toward the books' lines. Adds proj_lines_adj (points moved)."""
    r = res.copy()
    r["proj_lines_adj"] = np.nan
    if lines is None or lines.empty or "proj_model" not in r:
        return r
    r = r.merge(lines, on="dk_id", how="left")
    adj = pd.Series(0.0, index=r.index)
    used = pd.Series(False, index=r.index)
    for stat, pts in PTS.items():
        lc, mc = f"line_{stat}", f"p_{stat}"
        if lc not in r or mc not in r:
            continue
        ok = r[lc].notna() & r[mc].notna() & r.proj_model.notna()
        adj[ok] += WEIGHT[stat] * pts * ((r.loc[ok, lc] + SHIFT[stat]) - r.loc[ok, mc])
        used |= ok
    if "p_td" in r and "p_any_td" in r:
        ok = r.p_td.notna() & r.p_any_td.notna() & r.proj_model.notna()
        e_mkt = -np.log1p(-r.loc[ok, "p_td"].clip(0, 0.95))       # expected TDs from P(at least one)
        e_mod = -np.log1p(-r.loc[ok, "p_any_td"].clip(0, 0.95))
        adj[ok] += WEIGHT["any_td"] * 6 * (e_mkt - e_mod)
        used |= ok
    r.loc[used, "proj_lines_adj"] = adj[used].round(2)
    r.loc[used, "proj_model"] = (r.loc[used, "proj_model"] + adj[used]).clip(lower=0).round(2)
    return r.drop(columns=[c for c in r.columns if c.startswith("line_") or c == "p_td"])

