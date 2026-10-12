"""Live check of every data source from *this* computer.

Each check records status code, response type, record count and elapsed time, so
"automatic import works" is a measured fact on your network, not an assumption.
"""

from __future__ import annotations

import time

from .config import Settings
from .errors import StepError


def _check(name, fn, out):
    t = time.time()
    try:
        msg = fn()
        out(f"✅ {name}: {msg} ({time.time() - t:.1f}s)")
        return True
    except StepError as e:
        out(f"❌ {name}: {e.what}\n     Why: {e.why}\n     Fix: {e.fix}" + (f"\n     Detail: {e.detail[:160]}" if e.detail else ""))
    except Exception as e:  # noqa: BLE001
        out(f"❌ {name}: {type(e).__name__}: {str(e)[:200]}")
    return False


def run(out=print) -> bool:
    from .sources import dk, nflverse as nv, odds, sleeper
    s = Settings.load()
    res = {}

    def lobby():
        sl = dk.discover()
        m = dk.pick_main_slate(sl)
        res["dg"] = m.draft_group_id
        return f"{len(sl)} Classic slates open; main = {m.label}"
    res["lobby"] = _check("DraftKings lobby (slate discovery)", lobby, out)

    def draftables():
        if "dg" not in res:
            raise StepError("draftables", "Skipped — lobby failed, no draft group to test.")
        ls = dk.load_auto(res["dg"])
        return f"{len(ls.players)} players via {ls.route}; IDs {'verified' if ls.ids_verified else 'not yet verified vs CSV'}"
    res["draftables"] = _check("DraftKings players & salaries", draftables, out)

    def sl_players():
        df, note = sleeper.players()
        return f"{len(df)} players, {int((df.injury != '').sum())} with an injury tag ({note})"
    res["sleeper"] = _check("Sleeper injuries & depth charts", sl_players, out)

    def sl_proj():
        season, week = nv.upcoming_week(nv.schedules())  # same week the weekly run uses
        df, note = sleeper.projections(season, week)
        return f"{len(df)} projections for {season} week {week}"
    res["sleeper_proj"] = _check("Sleeper projections", sl_proj, out)

    def nfl():
        st = nv.weekly_stats([nv.current_season()])
        sc = nv.schedules()
        return f"stats through week {int(st.week.max())}; schedule has {len(sc)} games"
    res["nflverse"] = _check("nflverse stats & schedule", nfl, out)

    if s.odds_api_key:
        res["odds"] = _check("The Odds API (lines)", lambda: f"{odds.game_lines(s.odds_api_key).event_id.nunique()} games with lines", out)
    else:
        out("⏭️  The Odds API: no key set (optional — needed for live lines and props).")

    if s.yahoo_client_id and s.yahoo_client_secret:
        from .sources import yahoo
        if yahoo.connected():
            res["yahoo"] = _check("Yahoo Fantasy (your leagues)", lambda: f"{len(yahoo.my_teams(s.yahoo_client_id, s.yahoo_client_secret))} NFL team(s) found", out)
        else:
            out("⏭️  Yahoo Fantasy: keys saved but not connected yet — use the My Yahoo team tab.")
    else:
        out("⏭️  Yahoo Fantasy: not set up (optional — My Yahoo team tab explains how).")

    core = res.get("nflverse") and (res.get("draftables") or True)
    out("")
    if not res.get("lobby") or not res.get("draftables"):
        out("DraftKings automatic import is NOT working from this network. The app will use DKSalaries.csv "
            "from ~/BeatNFL/inbox instead (DraftKings → any NFL Classic contest → Export to CSV).")
    return bool(core)
