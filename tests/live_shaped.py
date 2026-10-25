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
FULL_TO_ABBR = {v: k for k, v in ODDS_NAME.items()}


def _utc(day: str, hhmm: str) -> pd.Timestamp:
    """'10/18/2026', '04:25PM' (Eastern) → UTC timestamp."""
    return pd.Timestamp(f"{day} {hhmm}").tz_localize("America/New_York").tz_convert("UTC")


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.0000000Z")


def build(stats: pd.DataFrame, sched: pd.DataFrame, slate_csv: Path, season: int = 2026, week: int = 6) -> dict:
    """Return {url_fragment: payload} for every endpoint the app calls, for the given (upcoming) week."""
    rng = np.random.default_rng(6)
    stats = stats[(stats.season < season) | (stats.week < week)]
    st = stats[(stats.season == stats.season.max()) & stats.position.isin(["QB", "RB", "WR", "TE"])]
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
        projections.append({"player_id": sid, "team": team, "opponent": None, "week": week, "season": str(season),
                            "player": {"first_name": first, "last_name": lastn, "position": r.position, "team": team},
                            "stats": {"pass_yd": a.passing_yards, "pass_td": a.passing_tds, "pass_att": a.attempts,
                                      "pass_cmp": a.completions, "pass_int": a.passing_interceptions,
                                      "rush_yd": a.rushing_yards, "rush_att": a.carries, "rush_td": a.rushing_tds,
                                      "rec": a.receptions, "rec_yd": a.receiving_yards, "rec_tgt": a.targets,
                                      "rec_td": a.receiving_tds, "pts_ppr": 0}})
    for j in range(11000):  # real feed lists ~11k players incl. retired/practice squad — pad to that size
        players[f"x{j}"] = {"player_id": f"x{j}", "full_name": f"Inactive Player {j}", "position": ["OL", "LB", "CB", "K"][j % 4],
                            "team": None, "gsis_id": None, "injury_status": None, "status": "Inactive", "active": False}
    for t in sorted(set(sched[(sched.season == season)].home_team)):
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
            comps[game] = {"competitionId": comp_id, "name": f"{a} @ {h}", "startTime": _iso(_utc(gi[1], gi[2]))}
            comp_id += 1
        status = ["None", "None", "None", "Q", "None", "O", "None", "None", "None", "None", "None", "D"][int(r.ID) % 12]
        d = {"draftableId": int(r.ID), "playerId": int(r.ID) + 9_000_000, "displayName": r.Name.strip(),
             "firstName": r.Name.split()[0], "lastName": r.Name.split()[-1], "position": r.Position,
             "rosterSlotId": 66, "salary": int(r.Salary), "status": status, "isDisabled": False,
             "teamAbbreviation": r.TeamAbbrev, "competition": comps[game]}
        draftables.append(d)
        if r.Position != "QB" and r.Position != "DST":
            draftables.append({**d, "rosterSlotId": 71})  # FLEX listing, same ID
    first = min(pd.Timestamp(c["startTime"]) for c in comps.values())
    lobby = {"Contests": [], "DraftGroups": [
        {"DraftGroupId": 135000, "ContestTypeId": 21, "StartDate": _iso(first), "DraftGroupTag": "Featured",
         "GameType": "Classic", "GameCount": len(comps), "ContestStartTimeSuffix": None},
        {"DraftGroupId": 135001, "ContestTypeId": 21, "StartDate": _iso(first + pd.Timedelta(hours=3, minutes=5)), "DraftGroupTag": "",
         "GameType": "Classic", "GameCount": 3, "ContestStartTimeSuffix": " (Afternoon Turbo)"}]}

    # The Odds API: this week's lines + one game's props
    wk = sched[(sched.season == season) & (sched.week == week)]
    events, odds = [], []
    for i, g in enumerate(wk.itertuples()):
        kick = pd.Timestamp(f"{g.gameday} {g.gametime}").tz_localize("America/New_York").tz_convert("UTC")
        ev = {"id": f"ev{i}", "sport_key": "americanfootball_nfl", "commence_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "home_team": ODDS_NAME[g.home_team], "away_team": ODDS_NAME[g.away_team]}
        events.append(ev)
        sp = -(g.spread_line if pd.notna(g.spread_line) else 0) + rng.choice([-0.5, 0, 0.5])
        tot = g.total_line if pd.notna(g.total_line) else 44.5
        books = []
        for bk in ("draftkings", "fanduel"):
            books.append({"key": bk, "title": bk, "last_update": "2026-10-12T21:00:00Z", "markets": [
                {"key": "spreads", "last_update": "2026-10-12T21:00:00Z", "outcomes": [
                    {"name": ODDS_NAME[g.home_team], "price": -110, "point": sp},
                    {"name": ODDS_NAME[g.away_team], "price": -110, "point": -sp}]},
                {"key": "totals", "last_update": "2026-10-12T21:00:00Z", "outcomes": [
                    {"name": "Over", "price": -110, "point": tot}, {"name": "Under", "price": -110, "point": tot}]},
                {"key": "h2h", "last_update": "2026-10-12T21:00:00Z", "outcomes": [
                    {"name": ODDS_NAME[g.home_team], "price": -150}, {"name": ODDS_NAME[g.away_team], "price": 130}]}]})
        # DraftKings gives deep links (bookmaker event page + per-outcome betslip), as includeLinks=true returns
        dk = books[0]
        dk["link"] = f"https://sportsbook.draftkings.com/event/{ev['id']}"
        for m in dk["markets"]:
            m["outcomes"] = [{**o, "link": f"https://sportsbook.draftkings.com/event/{ev['id']}?outcomes={m['key']}_{o['name'].replace(' ', '')}"}
                             for o in m["outcomes"]]
        odds.append({**ev, "bookmakers": books})
    # props for one slate game, using the real top players of its two teams
    slate_teams = set(sal.TeamAbbrev.replace({"LAR": "LA"}))
    g0 = next(e for e in events if FULL_TO_ABBR.get(e["home_team"]) in slate_teams)
    tms = {FULL_TO_ABBR[g0["home_team"]], FULL_TO_ABBR[g0["away_team"]]}
    recent = st[st.team.isin(tms)].groupby(["player_display_name", "position"]).agg(
        rec=("receiving_yards", "mean"), rush=("rushing_yards", "mean"), pas=("passing_yards", "mean"), n=("week", "size")).reset_index()
    recent = recent[recent.n >= 2]
    wr = recent.sort_values("rec", ascending=False).iloc[0].player_display_name
    rb = recent[recent.position == "RB"].sort_values("rush", ascending=False).iloc[0].player_display_name
    qbs = list(recent[recent.position == "QB"].sort_values("pas", ascending=False).player_display_name.head(2))
    pr = []
    for name, mkt, line in ([(wr, "player_reception_yds", 74.5), (wr, "player_receptions", 5.5), (rb, "player_rush_yds", 81.5)]
                            + [(q, "player_pass_yds", 230.5) for q in qbs]):
        pr.append({"key": mkt, "last_update": "2026-10-12T21:00:00Z", "outcomes": [
            {"name": "Over", "description": name, "price": -115, "point": line},
            {"name": "Under", "description": name, "price": -105, "point": line}]})
    pr.append({"key": "player_anytime_td", "last_update": "2026-10-12T21:00:00Z",
               "outcomes": [{"name": "Yes", "description": rb, "price": -150}]})
    pr_dk = [{**m, "outcomes": [{**o, "link": f"https://sportsbook.draftkings.com/event/{g0['id']}?outcomes="
                                             f"{m['key']}_{o['description'].replace(' ', '')}_{o['name']}"}
                                for o in m["outcomes"]]} for m in pr]
    props = {**g0, "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": pr_dk,
                                   "link": f"https://sportsbook.draftkings.com/event/{g0['id']}"},
                                  {"key": "williamhill_us", "title": "Caesars", "markets": pr},
                                  {"key": "fanatics", "title": "Fanatics", "markets": [m for m in pr if m["key"] == "player_anytime_td"]}]}
    # FantasyPros API: one payload per position, field names as recorded from the live API
    fp = {}
    for pos in ("QB", "RB", "WR", "TE", "DST"):
        rows = []
        for r in sal[sal["Position"] == pos].head(60).itertuples():
            a = float(getattr(r, "AvgPointsPerGame") or 0)
            team = getattr(r, "TeamAbbrev")
            if pos == "DST":
                st = {"points": a, "points_ppr": a, "def_sack": 2.5, "def_int": 0.8, "def_fr": 0.6, "def_td": 0.1,
                      "def_retd": 0.03, "def_safety": 0.04, "def_pa": 22.0}
                rows.append({"fpid": 9000, "name": f"{team} Defense", "position_id": "DST", "team_id": team, "stats": st})
                continue
            if pos == "QB":
                st = {"points": a, "points_ppr": a, "pass_yds": 12 * a, "pass_tds": a / 12, "pass_ints": 0.7,
                      "rush_yds": a / 2, "rush_tds": 0.1, "fumbles": 0.2}
            else:
                st = {"points": a * .8, "points_ppr": a, "rush_yds": a * (3 if pos == "RB" else .2), "rush_tds": .02 * a,
                      "rec_rec": a / 4, "rec_yds": a * 3.5, "rec_tds": .02 * a, "fumbles": 0.05}
            rows.append({"fpid": 1000, "name": getattr(r, "Name"), "position_id": pos, "team_id": team, "stats": st})
        fp[pos] = {"season": str(season), "week": str(week), "count": str(len(rows)), "positions": pos, "scoring": "STD",
                   "players": rows, "public_api_limited": True, "tier": "premium"}
    return {"fp": fp, "players": players, "projections": projections, "lobby": lobby,
            "draftables": {"draftables": draftables, "competitions": list(comps.values())},
            "state": {"season": str(season), "week": week, "season_type": "regular", "display_week": week},
            "_season": season, "_week": week, "_prop_players": [wr, rb] + qbs,
            "events": events, "odds": odds, "props": props, "props_event": g0["id"]}


def install(monkeypatch, payloads: dict):
    """Route every provider URL to its stand-in; nflverse (GitHub) passes through to the real cache."""
    from beatnfl import net
    real = net.fetch

    def fake(url, **kw):
        if "api.fantasypros.com" in url:
            return json.dumps(payloads["fp"][kw["params"]["position"]]).encode(), net.Meta(url, 1.8e9, False, False)
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
    for mod in ("beatnfl.sources.dk", "beatnfl.sources.sleeper", "beatnfl.sources.odds", "beatnfl.sources.nflverse",
                "beatnfl.sources.fantasypros"):
        import importlib
        m = importlib.import_module(mod)
        if hasattr(m, "fetch"):
            monkeypatch.setattr(m, "fetch", fake)
        if hasattr(m, "fetch_json"):
            monkeypatch.setattr(m, "fetch_json", lambda url, **kw: (json.loads(fake(url, **kw)[0]), net.Meta(url, 1.8e9, False, False)))
