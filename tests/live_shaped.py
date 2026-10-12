"""Realistic stand-ins for every live source (DraftKings, Sleeper, The Odds API, Yahoo), built from real
nflverse rosters and shaped like each provider's documented responses. Used by test_live_shaped.py to run
every workflow end to end with all sources "working". Nothing here is ever shown as live data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).parent / "fixtures"
SLEEPER_TEAM = {"LA": "LAR"}
ODDS_NAME = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
             "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CLE": "Cleveland Browns", "IND": "Indianapolis Colts",
             "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders",
             "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
             "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
             "SEA": "Seattle Seahawks", "DEN": "Denver Broncos", "HOU": "Houston Texans", "JAX": "Jacksonville Jaguars",
             "DAL": "Dallas Cowboys", "GB": "Green Bay Packers", "WAS": "Washington Commanders", "SF": "San Francisco 49ers",
             "MIN": "Minnesota Vikings", "DET": "Detroit Lions", "MIA": "Miami Dolphins", "CIN": "Cincinnati Bengals"}


def build(stats: pd.DataFrame, sched: pd.DataFrame, slate_csv: Path) -> dict:
    """Return {url_fragment: payload} for every endpoint the app calls."""
    rng = np.random.default_rng(6)
    st = stats[(stats.season == 2026) & stats.position.isin(["QB", "RB", "WR", "TE"])]
    last = st.sort_values("week").drop_duplicates("player_id", keep="last")
    avg = st.groupby("player_id")[["passing_yards", "passing_tds", "attempts", "completions", "rushing_yards",
                                   "carries", "rushing_tds", "receptions", "receiving_yards", "targets",
                                   "receiving_tds", "passing_interceptions"]].mean()
    players, projections = {}, []
    inj_cycle = ["Questionable", None, None, "Out", None, None, None, "Doubtful", None, "IR"] + [None] * 30
    for i, r in enumerate(last.itertuples()):
        sid = str(10000 + i)
        team = SLEEPER_TEAM.get(r.team, r.team)
        if i % 97 == 5:  # a few players whose Sleeper team lags behind a trade
            team = "NYJ" if team != "NYJ" else "MIA"
        first, _, lastn = r.player_display_name.partition(" ")
        inj = inj_cycle[i % len(inj_cycle)] if i % 3 == 0 else None
        players[sid] = {"player_id": sid, "full_name": r.player_display_name, "first_name": first, "last_name": lastn,
                        "position": r.position, "team": team, "gsis_id": r.player_id if i % 50 else None,
                        "injury_status": inj, "injury_body_part": "Ankle" if inj else None, "status": "Active",
                        "depth_chart_position": r.position, "depth_chart_order": 1 + i % 3, "active": True,
                        "news_updated": 1792000000000 + i}
        a = avg.loc[r.player_id]
        projections.append({"player_id": sid, "team": team, "opponent": None, "week": 6, "season": "2026",
                            "player": {"first_name": first, "last_name": lastn, "position": r.position, "team": team},
                            "stats": {"pass_yd": a.passing_yards, "pass_td": a.passing_tds, "pass_att": a.attempts,
                                      "pass_cmp": a.completions, "pass_int": a.passing_interceptions,
                                      "rush_yd": a.rushing_yards, "rush_att": a.carries, "rush_td": a.rushing_tds,
                                      "rec": a.receptions, "rec_yd": a.receiving_yards, "rec_tgt": a.targets,
                                      "rec_td": a.receiving_tds, "pts_ppr": 0}})
    for j in range(11000):  # real feed lists ~11k players incl. retired/practice squad — pad to that size
        players[f"x{j}"] = {"player_id": f"x{j}", "full_name": f"Inactive Player {j}", "position": ["OL", "LB", "CB", "K"][j % 4],
                            "team": None, "gsis_id": None, "injury_status": None, "status": "Inactive", "active": False}
    for t in sorted(set(sched[(sched.season == 2026)].home_team)):
        tt = SLEEPER_TEAM.get(t, t)
        players[tt] = {"player_id": tt, "full_name": None, "first_name": t, "last_name": "Defense", "position": "DEF",
                       "team": tt, "gsis_id": None, "injury_status": None, "status": "Active", "active": True}
        projections.append({"player_id": tt, "team": tt, "player": {"first_name": t, "last_name": "Defense",
                                                                     "position": "DEF", "team": tt},
                            "stats": {"pts_allow": 21.5, "sack": 2.4, "int": 0.8, "fum_rec": 0.6, "def_td": 0.15}})

    # DraftKings lobby + draftables from the fixture slate (same IDs as the official-CSV fixture)
    sal = pd.read_csv(slate_csv, encoding="utf-8-sig", dtype={"ID": str})
    comps, comp_id = {}, 700
    draftables = []
    for r in sal.itertuples():
        gi = r._7.split()
        game = gi[0]
        if game not in comps:
            a, h = game.split("@")
            hh, mm = gi[2][:5].split(":")
            hr = (int(hh) % 12) + (12 if gi[2].endswith("PM") else 0)
            comps[game] = {"competitionId": comp_id, "name": f"{a} @ {h}", "startTime": f"2026-10-18T{hr + 4:02d}:{mm}:00.0000000Z"}
            comp_id += 1
        status = ["None", "None", "None", "Q", "None", "O", "None", "None", "None", "None", "None", "D"][int(r.ID) % 12]
        d = {"draftableId": int(r.ID), "playerId": int(r.ID) + 9_000_000, "displayName": r.Name.strip(),
             "firstName": r.Name.split()[0], "lastName": r.Name.split()[-1], "position": r.Position,
             "rosterSlotId": 66, "salary": int(r.Salary), "status": status, "isDisabled": False,
             "teamAbbreviation": r.TeamAbbrev, "competition": comps[game]}
        draftables.append(d)
        if r.Position != "QB" and r.Position != "DST":
            draftables.append({**d, "rosterSlotId": 71})  # FLEX listing, same ID
    lobby = {"Contests": [], "DraftGroups": [
        {"DraftGroupId": 135000, "ContestTypeId": 21, "StartDate": "2026-10-18T17:00:00.0000000Z", "DraftGroupTag": "Featured",
         "GameType": "Classic", "GameCount": len(comps), "ContestStartTimeSuffix": None},
        {"DraftGroupId": 135001, "ContestTypeId": 21, "StartDate": "2026-10-18T20:05:00.0000000Z", "DraftGroupTag": "",
         "GameType": "Classic", "GameCount": 3, "ContestStartTimeSuffix": " (Afternoon Turbo)"}]}

    # The Odds API: week 6 lines + one game's props
    wk = sched[(sched.season == 2026) & (sched.week == 6)]
    events, odds = [], []
    for i, g in enumerate(wk.itertuples()):
        ev = {"id": f"ev{i}", "sport_key": "americanfootball_nfl", "commence_time": f"{g.gameday}T17:00:00Z",
              "home_team": ODDS_NAME[g.home_team], "away_team": ODDS_NAME[g.away_team]}
        events.append(ev)
        sp = -(g.spread_line or 0) + rng.choice([-0.5, 0, 0.5])
        books = []
        for bk in ("draftkings", "fanduel"):
            books.append({"key": bk, "title": bk, "last_update": "2026-10-12T21:00:00Z", "markets": [
                {"key": "spreads", "last_update": "2026-10-12T21:00:00Z", "outcomes": [
                    {"name": ODDS_NAME[g.home_team], "price": -110, "point": sp},
                    {"name": ODDS_NAME[g.away_team], "price": -110, "point": -sp}]},
                {"key": "totals", "last_update": "2026-10-12T21:00:00Z", "outcomes": [
                    {"name": "Over", "price": -110, "point": g.total_line}, {"name": "Under", "price": -110, "point": g.total_line}]},
                {"key": "h2h", "last_update": "2026-10-12T21:00:00Z", "outcomes": [
                    {"name": ODDS_NAME[g.home_team], "price": -150}, {"name": ODDS_NAME[g.away_team], "price": 130}]}]})
        odds.append({**ev, "bookmakers": books})
    # props for the CHI @ ATL game using real names
    g0 = next(e for e in events if e["home_team"] == "Atlanta Falcons")
    pr = []
    for name, mkt, line in (("Drake London", "player_reception_yds", 74.5), ("Drake London", "player_receptions", 5.5),
                            ("Bijan Robinson", "player_rush_yds", 81.5), ("Caleb Williams", "player_pass_yds", 238.5),
                            ("Michael Penix Jr.", "player_pass_yds", 224.5)):
        pr.append({"key": mkt, "last_update": "2026-10-12T21:00:00Z", "outcomes": [
            {"name": "Over", "description": name, "price": -115, "point": line},
            {"name": "Under", "description": name, "price": -105, "point": line}]})
    pr.append({"key": "player_anytime_td", "last_update": "2026-10-12T21:00:00Z",
               "outcomes": [{"name": "Yes", "description": "Bijan Robinson", "price": -150}]})
    props = {**g0, "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": pr}]}
    return {"players": players, "projections": projections, "lobby": lobby,
            "draftables": {"draftables": draftables, "competitions": list(comps.values())},
            "state": {"season": "2026", "week": 6, "season_type": "regular", "display_week": 6},
            "events": events, "odds": odds, "props": props, "props_event": g0["id"]}


def install(monkeypatch, payloads: dict):
    """Route every provider URL to its stand-in; nflverse (GitHub) passes through to the real cache."""
    from beatnfl import net
    real = net.fetch

    def fake(url, **kw):
        m = {"api.sleeper.app/v1/players/nfl": "players", "api.sleeper.app/projections/": "projections",
             "api.sleeper.app/v1/state/nfl": "state", "draftkings.com/lobby/getcontests": "lobby",
             "/draftables": "draftables", "/events/" + payloads["props_event"] + "/odds": "props",
             "americanfootball_nfl/events": "events", "americanfootball_nfl/odds": "odds"}
        if "the-odds-api.com" in url and "/events/" in url and not url.endswith("/events/" + payloads["props_event"] + "/odds"):
            return json.dumps({"bookmakers": []}).encode(), net.Meta(url, 1.8e9, False, False)
        for frag, key in m.items():
            if frag in url:
                return json.dumps(payloads[key]).encode(), net.Meta(url, 1.8e9, False, False)
        return real(url, **kw)
    monkeypatch.setattr(net, "fetch", fake)
    for mod in ("beatnfl.sources.dk", "beatnfl.sources.sleeper", "beatnfl.sources.odds", "beatnfl.sources.nflverse"):
        import importlib
        m = importlib.import_module(mod)
        if hasattr(m, "fetch"):
            monkeypatch.setattr(m, "fetch", fake)
        if hasattr(m, "fetch_json"):
            monkeypatch.setattr(m, "fetch_json", lambda url, **kw: (json.loads(fake(url, **kw)[0]), net.Meta(url, 1.8e9, False, False)))
