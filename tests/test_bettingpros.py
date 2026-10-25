"""BettingPros Consumer API client and track record — against a FIXTURE server shaped like the
published OpenAPI spec (consumer.yaml): x-api-key header, _pagination, consumer_api tier, 403/429 errors."""

import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("BEATNFL_HOME", tempfile.mkdtemp(prefix="beatnfl_test_"))

from beatnfl import bp_record, net  # noqa: E402
from beatnfl.config import CACHE, HOME  # noqa: E402
from beatnfl.errors import StepError  # noqa: E402
from beatnfl.sources import bettingpros as bp  # noqa: E402

FIRST = ["Aaron", "Blake", "Caleb", "Derek", "Evan", "Felix", "Grant", "Henry", "Isaac", "Jared"]
LAST = ["Adams", "Brooks", "Carter", "Dawson", "Ellis", "Foster", "Garner"]


def pname(i):
    return f"{FIRST[i % 10]} {LAST[i // 10]}"


MARKETS = [(103, "passing-yards", "Pass Yds"), (106, "rushing-yards", "Rush Yds"), (105, "receiving-yards", "Rec Yds"),
           (104, "receptions", "Receptions"), (71, "anytime-touchdown-scorer", "Anytime TD"),
           (333, "rushing-receiving-yards", "Rush+Rec Yds")]


class R:
    def __init__(self, code, js):
        self.status_code, self._js = code, js
        self.content = json.dumps(js).encode()
        self.text = self.content.decode()
        self.headers = {}


class FakeBP:
    """Serves 60 props a week; completed weeks are graded. Counts real requests; can run out for the day."""
    def __init__(self, n_props=60, daily=1000, tier="plus", stars=True):
        self.calls, self.n, self.daily, self.tier, self.stars = 0, n_props, daily, tier, stars

    def prop(self, i, week, completed):
        mid, slug, _ = MARKETS[i % len(MARKETS)]
        line = 50.5 + i
        side = "over" if i % 3 else "under"
        actual = line + (8 if i % 2 else -8) if completed else None
        return {"sport": "NFL", "market_id": mid, "event_id": 1000 + week * 20 + i % 8,
                "participant": {"id": str(5000 + i), "name": pname(i),
                                "player": {"position": "WR", "team": "DET", "slug": f"player-{i}"}},
                "over": {"line": line, "odds": -110, "book": 12, "consensus_line": line, "consensus_odds": -115},
                "under": {"line": line, "odds": 100, "book": 10, "consensus_line": line, "consensus_odds": -105},
                "projection": {"recommended_side": side, "value": line + (5 if side == "over" else -5),
                               "probability": 0.58 if self.stars else None, "expected_value": 0.05 if self.stars else None,
                               "bet_rating": (1 + i % 5) if self.stars else None, "diff": 5, "signal_source": "projection",
                               "consensus_line": line},
                "scoring": {"is_scored": completed, "is_push": False, "push_reason": None, "actual": actual}}

    def get(self, url, params=None, timeout=None, headers=None):
        assert url.startswith(bp.BASE)
        if headers.get("x-api-key") != "good":
            return R(403, {"message": "Forbidden"})
        if self.calls >= self.daily:
            return R(429, {"message": "Limit Exceeded"})
        self.calls += 1
        ca = {"tier": self.tier, "page_limit": 20}
        path = url[len(bp.BASE):]
        if path == "/markets":
            return R(200, {"_parameters": {}, "consumer_api": ca, "markets": [
                {"id": m, "slug": s, "meta": {"short_label": l}} for m, s, l in MARKETS]})
        if path == "/props":
            lim = min(int(params.get("limit", 20)), 20)
            page = int(params.get("page", 1))
            completed = params.get("event_status") == "completed"
            allp = [self.prop(i, int(params["week"]), completed) for i in range(self.n)]
            pages = max(1, math.ceil(len(allp) / lim))
            return R(200, {"_parameters": dict(params), "consumer_api": ca,
                           "_pagination": {"page": page, "limit": lim, "total_pages": pages, "total_items": len(allp)},
                           "props": allp[(page - 1) * lim: page * lim]})
        return R(400, {"message": "bad path"})


@pytest.fixture()
def fake(monkeypatch):
    shutil.rmtree(CACHE, ignore_errors=True)
    shutil.rmtree(bp_record.DIR, ignore_errors=True)
    bp.USAGE.unlink(missing_ok=True)
    f = FakeBP()
    monkeypatch.setattr(net.requests, "get", f.get)
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    return f


def test_props_paginate_parse_and_cache(fake):
    df, info = bp.props("good", 2026, 6, completed=True)
    assert len(df) == 60 and info["pages"] == 3 and fake.calls == 4          # markets + 3 pages
    assert bp.usage()["used"] == 4 and bp.usage()["tier"] == "plus" and bp.remaining() == 96
    r = df.set_index("player").loc[pname(5)]                              # market 333: rush+rec, not rec yds
    assert r.stat == "rush_rec_yds" and r.bp_side == "over" and r.rec_odds == -110 and r.bp_stars == 1 + 5 % 5
    assert df.set_index("player").loc[pname(0)].stat == "pass_yds"
    assert df.scored.all() and df.actual.notna().all()
    df2, _ = bp.props("good", 2026, 6, completed=True)                        # completed weeks: cached for good
    assert fake.calls == 4 and len(df2) == 60


def test_bad_key_and_daily_limit_are_explained(fake):
    with pytest.raises(StepError) as e:
        bp.props("bad", 2026, 6)
    assert "refused" in e.value.what and "Settings" in e.value.fix
    fake.daily = 2                                   # markets + one page, then the API says the day is used up
    df, info = bp.props("good", 2026, 7, completed=True)
    assert info["partial"] and len(df) == 20 and bp.remaining() == 0
    with pytest.raises(StepError) as e:                                     # no more calls once out
        bp.get("good", "/props", {"sport": "NFL", "week": 9})
    assert "allowance" in e.value.what and fake.calls == 2


def test_collect_resumes_across_days(fake, monkeypatch):
    fake.daily = 9                                                           # markets + 2 weeks (3 pages each) + 2 spare
    bp_record.RESERVE = 0
    r = bp_record.collect("good", [2025], weeks=range(1, 6))
    assert r["stopped"] == "allowance" and len(r["fetched_now"]) == 2
    assert not bp_record.week_file(2025, 3).exists()                        # partial week not saved
    monkeypatch.setattr(bp, "_today", lambda: "2099-01-01")                 # next day
    fake.calls, fake.daily = 0, 1000
    r = bp_record.collect("good", [2025], weeks=range(1, 6))
    assert r["weeks_done"] == 5 and r["stopped"] is None
    g = bp_record.grade(bp_record.load_archive())
    assert len(g) == 300 and set(g.won) == {0, 1}
    s = bp_record.summarize(g)
    assert set(s["by_stars"].stars) == {1, 2, 3, 4, 5} and "too few" in s["verdict"]
    bp_record.RESERVE = 5


def test_grading_math():
    rows = pd.DataFrame([
        # over 50.5, actual 60 → win at -110 → +0.909
        dict(scored=True, push=False, bp_side="over", actual=60, rec_line=50.5, rec_odds=-110, rec_cons_odds=-120,
             bp_stars=5, bp_prob=0.6, market="Rec Yds", bp_signal="projection"),
        # under 50.5, actual 60 → loss
        dict(scored=True, push=False, bp_side="under", actual=60, rec_line=50.5, rec_odds=+120, rec_cons_odds=+100,
             bp_stars=5, bp_prob=0.6, market="Rec Yds", bp_signal="projection"),
        # lands on the line → push, dropped; and a void (player out) → dropped
        dict(scored=True, push=False, bp_side="over", actual=50, rec_line=50, rec_odds=-110, rec_cons_odds=-110,
             bp_stars=4, bp_prob=0.6, market="Rec Yds", bp_signal="projection"),
        dict(scored=True, push=True, bp_side="over", actual=None, rec_line=50.5, rec_odds=-110, rec_cons_odds=-110,
             bp_stars=4, bp_prob=0.6, market="Rec Yds", bp_signal="projection")])
    g = bp_record.grade(rows)
    assert list(g.won) == [1, 0]
    assert g.profit_best.round(3).tolist() == [round(100 / 110, 3), -1.0]
    assert g.profit_cons.round(3).tolist() == [round(100 / 120, 3), -1.0]
    s = bp_record.summarize(g)["by_stars"].iloc[0]
    assert s.picks == 2 and s.won_pct == 50.0


def test_no_ratings_in_archive_is_reported(fake):
    fake.stars = False
    df, _ = bp.props("good", 2026, 2, completed=True)
    s = bp_record.summarize(bp_record.grade(df))
    assert "no star ratings" in s["verdict"]


def test_snapshots_graded_against_results(fake):
    up, _ = bp.props("good", 2026, 4, completed=False)
    assert not up.scored.any()
    assert bp_record.save_snapshot(up) == 60
    assert bp_record.save_snapshot(up) == 60                                 # saved twice → last one kept
    g = bp_record.snapshot_record("good")
    assert len(g) == 60 and g.won.isin([0, 1]).all()
    bp_record.save_result("snapshots", bp_record.summarize(g))
    back = bp_record.load_result("snapshots")
    assert not back["by_stars"].empty and back["verdict"]


def test_ask_tool_reads_bettingpros_board(fake):
    from beatnfl import ask
    df, _ = bp.props("good", 2026, 8, completed=False)
    ctx = ask.Context(bettingpros=lambda: df)
    out = json.loads(ask.run_tool(ctx, "bettingpros", {"min_stars": 5}))
    assert out["count"] == 12 and all(p["bp_stars"] == 5 for p in out["props"])
    one = json.loads(ask.run_tool(ctx, "bettingpros", {"player": pname(7)}))
    assert one["count"] == 1 and one["props"][0]["player"] == pname(7)
    none = json.loads(ask.run_tool(ask.Context(), "bettingpros", {}))
    assert "Load BettingPros props" in none["board"]


def test_information_test_beyond_the_line():
    rng = np.random.default_rng(7)
    n = 1500
    p_true = rng.uniform(0.3, 0.7, n)
    over = rng.random(n) < p_true
    line = 50.5
    actual = np.where(over, line + 5, line - 5)
    def frame(bp_p_over):
        side = np.where(bp_p_over >= 0.5, "over", "under")
        return pd.DataFrame({"scored": True, "push": False, "actual": actual, "line": line, "rec_line": line,
                             "bp_side": side, "bp_prob": np.where(side == "over", bp_p_over, 1 - bp_p_over),
                             "over_cons_odds": over_odds, "under_cons_odds": under_odds, "market": "Rec Yds"})
    over_odds = rng.choice([-125, -115, -110, -105, 100, 105], n)          # market prices vary, but know nothing
    under_odds = np.where(over_odds < 0, -110 + (over_odds + 110) * -1, -120)
    good = bp_record.info_test(frame(p_true))                       # knows the true chance; market says 50/50
    assert good["info"]["z"] >= 2 and good["brier_bp"] < good["brier_market"] and "adds real information" in good["verdict"]
    noise = bp_record.info_test(frame(rng.uniform(0.3, 0.7, n)))    # random opinions
    assert "no proven information" in noise["verdict"]
    assert not good["by_stat"].empty
    small = bp_record.info_test(frame(p_true).head(100))
    assert "need at least 200" in small["verdict"]
    bp_record.save_result("t", {"verdict": "x", "by_stars": pd.DataFrame(), "info": good})
    assert not bp_record.load_result("t")["info"]["by_stat"].empty


def test_information_test_small_groups_do_not_crash():
    d = pd.DataFrame({"scored": True, "push": False, "actual": [60, 40] * 30, "line": 50.5, "rec_line": 50.5,
                      "bp_side": "over", "bp_prob": 0.6, "over_cons_odds": -110, "under_cons_odds": -110,
                      "market": [f"m{i}" for i in range(60)]})
    r = bp_record.info_test(d)
    assert r["n"] == 60 and r["by_stat"].empty and "need at least 200" in r["verdict"]
