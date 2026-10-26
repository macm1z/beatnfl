"""End-to-end run with every provider 'working' (realistic stand-ins, see live_shaped.py).

Slow (downloads nflverse history, trains the model), so it runs only with BEATNFL_FULL=1:
    BEATNFL_FULL=1 python -m pytest -q tests/test_live_shaped.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BEATNFL_FULL") != "1", reason="full end-to-end run (set BEATNFL_FULL=1)")
sys.path.insert(0, str(Path(__file__).parent))
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def payloads(tmp_path_factory):
    """Stand-ins for the current NFL week, so these tests keep working as the season moves on."""
    import live_shaped
    import make_fixture
    from beatnfl.sources import nflverse as nv
    import pandas as pd
    sched = nv.schedules()
    reg = sched[sched.game_type == "REG"].copy()
    reg["kick"] = pd.to_datetime(reg.gameday.astype(str) + " " + reg.gametime.astype(str)).dt.tz_localize("America/New_York")
    sun = reg[pd.to_datetime(reg.gameday).dt.weekday == 6]
    # the next week whose Sunday games are all still in the future (a slate that hasn't started)
    fut = sun.groupby(["season", "week"]).kick.min()
    fut = fut[fut > pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(hours=1)]
    if fut.empty:
        pytest.skip("no upcoming regular-season week to test against")
    season, week = (int(x) for x in fut.index[0])
    csv = tmp_path_factory.mktemp("slate") / f"DKSalaries_FIXTURE_{season}wk{week}.csv"
    make_fixture.main(csv, season, week)
    p = live_shaped.build(nv.weekly_stats(range(season - 1, season + 1)), sched, csv, season, week)
    p["_n_games"] = int(len(sched[(sched.season == season) & (sched.week == week) & (sched.game_type == "REG")]))
    sal = pd.read_csv(csv, encoding="utf-8-sig")
    two = pd.concat([sal[sal.Position == "QB"].head(1), sal[sal.Position == "WR"].head(1)])
    p["_paste"] = "Player\tTeam\tPos\tProj\n" + "".join(f"{r.Name.strip()}\t{r.TeamAbbrev}\t{r.Position}\t{20 + i}\n"
                                                         for i, r in enumerate(two.itertuples()))
    return p


@pytest.fixture
def live(monkeypatch, payloads, tmp_path):
    import live_shaped
    live_shaped.install(monkeypatch, payloads)
    return payloads


def _no_bugs(run):
    bad = [s.as_text() for s in run.steps if "Unexpected error" in s.message or s.status == "fail"]
    assert not bad, "\n".join(bad)


def _yahoo_mock(monkeypatch):
    import test_core as T
    from beatnfl.sources import yahoo
    players = [T._yp("461.p.01", "Josh Allen", "Buf", "QB", "QB", pct="98"), T._yp("461.p.02", "Bijan Robinson", "Atl", "RB", "RB", pct="99"),
               T._yp("461.p.03", "James Cook", "Buf", "RB", "BN", status="Q", pct="95"),
               T._yp("461.p.05", "Drake London", "Atl", "WR", "WR", pct="97"), T._yp("461.p.06", "Kyle Pitts", "Atl", "TE", "TE", pct="80"),
               T._yp("461.p.07", "Tony Pollard", "Ten", "RB", "W/R/T", status="O", pct="60"), T._yp("461.p.09", "Kansas City", "KC", "DEF", "DEF")]

    def get(path, cid, sec):
        if path.startswith("users;"):
            return {"x": [{"team": [[{"team_key": "461.l.777.t.3"}, {"name": "Harry's Team"}]]}]}
        if "/settings" in path:
            return {"x": [{"roster_position": {"position": p, "count": n}} for p, n in
                          (("QB", 1), ("WR", 2), ("RB", 2), ("TE", 1), ("W/R/T", 1), ("K", 1), ("DEF", 1))]}
        if "/players;" in path:
            return {"x": players} if "start=0" in path else {"x": []}
        return {"x": players}
    monkeypatch.setattr(yahoo, "get", get)
    monkeypatch.setattr(yahoo, "connected", lambda: True)


def test_weekly_run_all_sources(live, monkeypatch):
    from beatnfl import pipeline
    from beatnfl.config import Settings
    from beatnfl.optimizer import OptSettings, validate
    _yahoo_mock(monkeypatch)
    s = Settings(odds_api_key="test-key", yahoo_client_id="cid", yahoo_client_secret="sec", fantasypros_api_key="fp-key")
    run = pipeline.run_week(s)
    fp = [x for x in run.steps if x.step == "4b. FantasyPros projections"]
    assert fp and fp[0].status == "ok", [x.as_text() for x in run.steps if x.step.startswith("4b")]
    assert run.pool.proj_paid.notna().sum() > 150
    _no_bugs(run)
    steps = {x.step: x for x in run.steps}
    assert run.slate.route == "json" and steps["1. Slate"].status in ("ok", "warn")
    assert steps["4d. Sportsbook lines"].status == "ok" and "moved toward" in steps["4d. Sportsbook lines"].message, steps["4d. Sportsbook lines"].as_text()
    assert run.pool.proj_lines_adj.notna().sum() >= 1
    rng_ = run.pool.dropna(subset=["p_dk_floor", "p_dk_ceiling", "proj_model"])
    assert len(rng_) > 200 and (rng_.p_dk_floor <= rng_.proj_model + 1e-6).all() and (rng_.p_dk_ceiling >= rng_.proj_model - 1e-6).all()
    assert (rng_.p_dk_ceiling - rng_.p_dk_floor).median() > 5
    for k in ("2. Injuries & roles", "2b. Injuries (Yahoo)", "3. Schedule", "3b. Betting lines", "4a. Sleeper projections",
              "4c. In-house model"):
        assert steps[k].status in ("ok", "warn"), steps[k].as_text()
    assert run.pool.out.sum() > 0 and (run.pool.status == "Q").sum() > 0
    assert run.pool.loc[~run.pool.out, "proj"].notna().mean() > 0.95
    assert run.pool.proj_sleeper.notna().sum() > 200 and run.pool.proj_model.notna().sum() > 200
    assert run.pool.loc[run.pool.out, "proj"].isna().all()
    out = pipeline.build_lineups(run, OptSettings(n_lineups=20), name="live-shaped")
    assert out["result"].status == "complete" and len(out["result"].lineups) == 20
    assert all(not validate(lu, run.pool) for lu in out["result"].lineups)
    assert all(not run.pool.set_index("dk_id").loc[lu].out.any() for lu in out["result"].lineups)


def test_research_all_sources(live):
    from beatnfl import research
    from beatnfl.models import train
    from beatnfl.sources import odds, sleeper
    pm, dm, meta = train.get_models()
    spp, _ = sleeper.players()
    lines = odds.consensus(odds.game_lines("k"))
    fc, warns = train.forecast_week(live['_season'], live['_week'], spp, pm, lines)
    assert len(fc) > 250 and fc.p_dk_pts.notna().all()
    raw = odds.game_lines("k")
    S, W = live["_season"], live["_week"]
    g = research.games_table(S, W, raw)
    assert len(g) == live["_n_games"] and (g._src == "odds_api").sum() >= len(g) - 1
    gp = research.game_probs(S, W, raw)
    assert set(gp["Bet type"]) == {"Spread", "Moneyline", "Total"} and gp["Market %"].between(1, 99).all()
    board, info = research.props_board("k", S, W, fc, pm, research.CORE_PROP_MARKETS)
    assert len(board) >= 5 and board.model_p_over.notna().sum() >= 4
    # deep links: game sides from DraftKings, and props (Over/Under slips; consensus view uses DraftKings)
    assert raw[raw.book == "draftkings"].link.str.contains("outcomes=").all() and raw[raw.book == "fanduel"].link.isna().all()
    slips = research.game_slips(gp, raw[raw.book == "draftkings"])
    assert all(isinstance(x, str) and "draftkings" in x for x in slips)
    assert research.game_slips(gp, raw[raw.book == "fanduel"]) == [None] * len(gp)
    dkb = board[board.book == "draftkings"]
    assert dkb.link.notna().all() and dkb[dkb.stat != "any_td"].under_link.str.endswith("_Under").all()
    pv = research.props_view(board)
    lean = pv.Bet != ""
    assert (pv[~lean]["Bet slip"].isna()).all()                          # a slip only for the recommended side
    assert pv[lean]["Bet slip"].dropna().str.contains("draftkings").all()
    pv_c = research.props_view(board, book="Caesars")
    assert pv_c["Bet slip"].isna().all()
    m = research.matchups(S, W)
    assert len(m) == 2 * live["_n_games"]
    # FantasyPros stat lines mix into props forecasts
    from beatnfl.sources import fantasypros
    fpdf, _ = fantasypros.projections("fp-key", S, W)
    fc2, n = research.blend_fp_stats(fc, fpdf, {"rec_yds": 0.4, "rush_yds": 0.3})
    assert n > 50 and (fc2.p_rec_yds != fc.p_rec_yds).sum() > 20


def test_midweek_picks_end_to_end(live):
    from beatnfl import research, storage
    from beatnfl.config import Settings
    s = Settings(odds_api_key="k", fantasypros_api_key="fp-key")
    S, W = live["_season"], live["_week"]
    picks, info = research.build_midweek(s, S, W)
    assert info["season"] == S and info["week"] == W and "FantasyPros" in info["fp_note"]
    loose, _ = research.build_midweek(Settings(odds_api_key="k"), S, W)          # model-only list is a superset
    assert len(loose) >= len(picks) and (loose.empty or (loose.side == "Under").all())
    assert len(storage.picks()) >= len(loose)
    tbl, sm = research.grade_picks()
    assert sm["n"] == len(storage.picks())


def test_source_compare_end_to_end(live):
    """FantasyPros vs Sleeper vs model on completed 2026 weeks (stand-in FantasyPros/Sleeper numbers)."""
    from beatnfl import backtest
    out = backtest.source_compare("fp-key", seasons=(live["_season"],), current_weights={"paid": .5, "sleeper": .25, "model": .25},
                                  progress=lambda m: None)
    assert out["n"] > 500 and out["weeks_fantasypros"] >= 3
    assert {r["pos"] for r in out["by_pos"]} >= {"QB", "RB", "WR", "TE"}
    for pos, w in out["weights"].items():
        assert abs(sum(w.values()) - 1) < 1e-6 and set(w) == {"paid", "sleeper", "model"}
    assert all(r["mae_best_blend"] <= min(r["mae_paid"], r["mae_sleeper"], r["mae_model"]) + 1e-6 for r in out["by_pos"])
    assert out["stats"] and set(out["stat_weights"]) <= set(backtest.STAT_BLEND)


def test_app_every_button(live, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from beatnfl import pipeline, update
    from beatnfl.config import Settings
    _yahoo_mock(monkeypatch)
    s = Settings.load()
    s.odds_api_key, s.yahoo_client_id, s.yahoo_client_secret, s.fantasypros_api_key = "k", "cid", "sec", "fp-key"
    s.anthropic_api_key = "ak"
    s.bettingpros_api_key = "good"
    s.save()
    # BettingPros: FIXTURE server for its URLs only; everything else goes through untouched
    from test_bettingpros import FakeBP
    from beatnfl import net as _net
    from beatnfl.sources import bettingpros as _bps
    _fbp, _real_get = FakeBP(), _net.requests.get

    def _get(url, *a, **k):
        return _fbp.get(url, k.get("params"), k.get("timeout"), k.get("headers")) if url.startswith(_bps.BASE) else _real_get(url, *a, **k)
    monkeypatch.setattr(_net.requests, "get", _get)
    from beatnfl import ask as _ask
    _replies = []

    def _fake_post(key, body, session=None):
        if not _replies:
            _replies.extend([
                {"stop_reason": "tool_use", "usage": {"input_tokens": 500, "output_tokens": 50}, "content": [
                    {"type": "tool_use", "id": "a", "name": "pool_players", "input": {"position": "RB", "max_salary": 6000}},
                    {"type": "tool_use", "id": "b", "name": "build_lineups", "input": {"contest": "cash"}}]},
                {"stop_reason": "end_turn", "usage": {"input_tokens": 900, "output_tokens": 80},
                 "content": [{"type": "text", "text": "Here is your cash lineup."}]}])
        assert key == "ak" and body["tools"]
        return _replies.pop(0)
    monkeypatch.setattr(_ask, "_post", _fake_post)
    monkeypatch.setattr(update, "check", lambda r, t="": {"sha": "x", "date": "2026-10-12", "message": "test", "update_available": False})
    # a finished props backtest on disk, so the Evidence section renders its tables
    from beatnfl import props_backtest as pb
    import numpy as np
    rng = np.random.default_rng(3)
    D = pd.DataFrame({"line": 60.5, "actual": rng.normal(62, 25, 400), "model_p_over": rng.uniform(.3, .7, 400),
                      "p_over": rng.uniform(.45, .55, 400), "close_p_over": rng.uniform(.45, .55, 400),
                      "best_over": -105.0, "best_under": -108.0, "stat": "rec_yds",
                      "event_id": [f"e{i}" for i in range(400)], "player": "P", "books": 2})
    res = pb.summarize(D) | {"season": 2025, "status": "paused", "weeks": [1, 2, 3], "props_matched": 400}
    pb.EVID.mkdir(parents=True, exist_ok=True)
    (pb.EVID / "props_backtest_2025.json").write_text(json.dumps(res, default=str))
    app = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = ["⟳ Refresh", "⚙ Build lineups", "💾 Save", "Load roster & projections", "Load matchups",
              "Forecast every player this week", "Load props", "Grade frozen forecasts", "Grade live weights", "Copy diagnostics", "Check for updates",
              "🔎 Find this week's picks", "Grade picks", "Record closing prices now", "Use these weights", "📊 Grade against actual results", "⟳ Refresh"]
    at.text_area(key="pasted_proj").input(live["_paste"])
    for lab in labels:
        btn = [b for b in at.button if b.label == lab]
        assert btn, f"button missing: {lab}"
        btn[0].click().run()
        assert not at.exception, f"{lab}: {[e.value for e in at.exception]}"
        errs = [e.value for e in at.error]
        assert not any("Unexpected" in e or "Traceback" in e for e in errs), f"{lab}: {errs}"

    # props controls: pick a game, a sportsbook, each sort order, and the disagreement filter
    for key, pick in (("props_game", 1), ("props_book", 1), ("props_sort", 1), ("props_sort", 2)):
        sb = at.selectbox(key=key)
        sb.set_value(sb.options[min(pick, len(sb.options) - 1)]).run()
        assert not at.exception, f"{key}: {[e.value for e in at.exception]}"
    # the fixture week may already be under way (tests run on game days too) — show started games for these checks
    at.toggle(key="games_started").set_value(True).run()
    at.toggle(key="props_started").set_value(True).run()
    for key in ("props_show", "games_show", "games_type", "games_game", "games_book", "games_book_game"):
        sb = at.selectbox(key=key)
        sb.set_value(sb.options[1]).run()
        assert not at.exception, f"{key}: {[e.value for e in at.exception]}"
    # BettingPros: load this week's board (saved as a snapshot) and build the record
    [b for b in at.button if b.label == "Load BettingPros props"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["bp"]) == 60
    at.selectbox(key="bp_stars").set_value("4+").run()
    assert not at.exception, [e.value for e in at.exception]
    [b for b in at.button if b.label.startswith("Collect finished weeks now")][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not [e.value for e in at.error if "BettingPros" in e.value], [e.value for e in at.error]
    from beatnfl import bp_record as _bpr
    assert _bpr.load_result("archive") and _bpr.load_result("snapshots")
    assert any("BettingPros' archive" in m.value for m in at.markdown)
    # Parlay tab: legs from the props board and a game side, priced per book, with slips; then cleared
    from beatnfl import parlay as _pl
    bd = at.session_state["board"]
    keys = (bd.game + "|" + bd.player + "|" + bd.stat).drop_duplicates().head(2).tolist()
    legs = [_pl.prop_leg(bd, k, "Under") for k in keys]
    at.session_state["parlay"] = legs
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("leg parlay" in m.value for m in at.markdown)
    assert any(m.label.endswith("parlay odds") and m.value.startswith(("+", "-")) for m in at.metric)   # the final odds
    assert any(w for w in at.warning if "Same-game" in w.value)          # both legs are from the one props game
    [b for b in at.button if b.label == "Clear parlay"][0].click().run()
    assert not at.exception and at.session_state["parlay"] == []
    # weekly review renders (whatever has been graded for that week)
    [b for b in at.button if b.label == "📋 Review this week"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["review"]["week"] >= 1
    # contest types: each builds; single entry gives best + 2 backups, multi-entry the full set
    for ct, n in (("cash", 3), ("multi", None), ("se", 3)):
        at.radio(key="contest").set_value(ct).run()
        [b for b in at.button if b.label == "⚙ Build lineups"][0].click().run()
        assert not at.exception, f"contest {ct}: {[e.value for e in at.exception]}"
        got = len(at.session_state["lineups"]["result"].lineups)
        assert got == n if n else got > 3, (ct, got)
    assert Settings.load().contest == "se"
    # Ask tab: a question runs the tools on this week's pool and offers the built lineups to This week
    at.chat_input(key="ask_q").set_value("Build my cash lineup").run()
    assert not at.exception, [e.value for e in at.exception]
    h = at.session_state["ask_hist"][-1]
    assert h["a"] == "Here is your cash lineup." and h["tools"] == ["pool_players", "build_lineups"] and h["cost"] > 0
    assert len(h["built"]["result"].lineups) == 3
    use = [b for b in at.button if b.label == "Use on This week"]
    assert use
    use[0].click().run()
    assert not at.exception and at.session_state["lineups"]["result"].lineups == h["built"]["result"].lineups
    [b for b in at.button if b.label == "Clear conversation"][0].click().run()
    assert at.session_state["ask_hist"] == []
    # saved: one set saved, building+saving the same lineups again must not add a duplicate
    from beatnfl import storage
    assert len(storage.list_lineup_sets()) >= 1
    run = at.session_state["run"]
    assert any(x.step == "4b. Your projections" and x.status in ("ok", "warn") and "2 of" in x.message for x in run.steps)
    assert any(b.label == "Run the source test (≈10–15 min)" for b in at.button)
    assert any(b.label == "Run the props backtest" for b in at.button)
    assert any("paused by the credit limit" in m.value for m in at.markdown)
    s2 = Settings.load()
    assert s2.pos_weights, "Use these weights did not save per-position weights"
    s2.odds_api_key = s2.yahoo_client_id = s2.yahoo_client_secret = s2.fantasypros_api_key = s2.anthropic_api_key = s2.bettingpros_api_key = ""
    s2.pos_weights, s2.stat_weights = {}, {}
    s2.save()


def test_yahoo_sign_in_and_refresh(monkeypatch, tmp_path):
    import time
    from beatnfl.sources import yahoo
    monkeypatch.setattr(yahoo, "TOKEN_PATH", tmp_path / "tok.json")
    calls = []

    class R:
        status_code = 200
        def __init__(self, js): self._js = js
        def json(self): return self._js
    def post(url, data=None, auth=None, timeout=None):
        calls.append(data["grant_type"])
        return R({"access_token": f"A{len(calls)}", "refresh_token": "R1", "expires_in": 3600})
    monkeypatch.setattr(yahoo.requests, "post", post)
    yahoo.connect("cid", "sec", " CODE ")
    assert yahoo.connected() and calls == ["authorization_code"]
    assert yahoo._access("cid", "sec") == "A1"            # fresh token reused
    tok = json.loads(yahoo.TOKEN_PATH.read_text()); tok["obtained_at"] = time.time() - 4000
    yahoo.TOKEN_PATH.write_text(json.dumps(tok))
    assert yahoo._access("cid", "sec") == "A2" and calls[-1] == "refresh_token"   # expired → refreshed
    assert oct(yahoo.TOKEN_PATH.stat().st_mode)[-3:] == "600"
