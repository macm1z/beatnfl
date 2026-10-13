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
def payloads():
    import live_shaped
    from beatnfl.sources import nflverse as nv
    return live_shaped.build(nv.weekly_stats([2026]), nv.schedules(), FIX / "DKSalaries_FIXTURE_2026wk6.csv")


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
    fc, warns = train.forecast_week(2026, 6, spp, pm, lines)
    assert len(fc) > 250 and fc.p_dk_pts.notna().all()
    raw = odds.game_lines("k")
    g = research.games_table(2026, 6, raw)
    assert len(g) >= 13 and (g._src == "odds_api").sum() >= 13
    gp = research.game_probs(2026, 6, raw)
    assert set(gp["Bet type"]) == {"Spread", "Moneyline", "Total"} and gp["Market %"].between(1, 99).all()
    board, info = research.props_board("k", 2026, 6, fc, pm, research.CORE_PROP_MARKETS)
    assert len(board) >= 5 and board.model_p_over.notna().sum() >= 4
    m = research.matchups(2026, 6)
    assert len(m) == 28
    # FantasyPros stat lines mix into props forecasts
    from beatnfl.sources import fantasypros
    fpdf, _ = fantasypros.projections("fp-key", 2026, 6)
    fc2, n = research.blend_fp_stats(fc, fpdf, {"rec_yds": 0.4, "rush_yds": 0.3})
    assert n > 50 and (fc2.p_rec_yds != fc.p_rec_yds).sum() > 20


def test_midweek_picks_end_to_end(live):
    from beatnfl import research, storage
    from beatnfl.config import Settings
    s = Settings(odds_api_key="k", fantasypros_api_key="fp-key")
    picks, info = research.build_midweek(s, 2026, 6)
    assert info["season"] == 2026 and info["week"] == 6 and "FantasyPros" in info["fp_note"]
    loose, _ = research.build_midweek(Settings(odds_api_key="k"), 2026, 6)          # model-only list is a superset
    assert len(loose) >= len(picks) and len(loose) > 0
    assert len(storage.picks()) >= len(loose)
    tbl, sm = research.grade_picks()
    assert sm["n"] == len(storage.picks())


def test_source_compare_end_to_end(live):
    """FantasyPros vs Sleeper vs model on completed 2026 weeks (stand-in FantasyPros/Sleeper numbers)."""
    from beatnfl import backtest
    out = backtest.source_compare("fp-key", seasons=(2026,), current_weights={"paid": .5, "sleeper": .25, "model": .25},
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
    s.save()
    monkeypatch.setattr(update, "check", lambda r, t="": {"sha": "x", "date": "2026-10-12", "message": "test", "update_available": False})
    # a finished props backtest on disk, so the Evidence section renders its tables
    from beatnfl import props_backtest as pb
    import numpy as np
    rng = np.random.default_rng(3)
    D = pd.DataFrame({"line": 60.5, "actual": rng.normal(62, 25, 400), "model_p_over": rng.uniform(.3, .7, 400),
                      "p_over": rng.uniform(.45, .55, 400), "close_p_over": rng.uniform(.45, .55, 400),
                      "best_over": -105.0, "best_under": -108.0, "stat": "rec_yds"})
    res = pb.summarize(D) | {"season": 2025, "status": "paused", "weeks": [1, 2, 3], "props_matched": 400}
    pb.EVID.mkdir(parents=True, exist_ok=True)
    (pb.EVID / "props_backtest_2025.json").write_text(json.dumps(res, default=str))
    app = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = ["⟳ Refresh", "⚙ Build lineups", "💾 Save", "Load roster & projections", "Load matchups",
              "Forecast every player this week", "Load props", "Grade frozen forecasts", "Grade live weights", "Copy diagnostics", "Check for updates",
              "🔎 Find this week's picks", "Grade picks", "Record closing prices now", "Use these weights", "⟳ Refresh"]
    at.text_area(key="pasted_proj").input("Player\tTeam\tPos\tProj\nJosh Allen\tBUF\tQB\t24.9\nPuka Nacua\tLAR\tWR\t21.0\n")
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
    for key in ("props_show", "games_show", "games_type", "games_game"):
        sb = at.selectbox(key=key)
        sb.set_value(sb.options[1]).run()
        assert not at.exception, f"{key}: {[e.value for e in at.exception]}"
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
    s2.odds_api_key = s2.yahoo_client_id = s2.yahoo_client_secret = s2.fantasypros_api_key = ""
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
