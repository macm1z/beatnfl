"""Offline tests. Fixtures are labeled FIXTURE and never treated as live data.

Mapped to the acceptance list in the reference spec where relevant (IMPORT-01, OPT-01 …).
Run:  python -m pytest -q tests
"""

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIX = Path(__file__).parent / "fixtures"
os.environ.setdefault("BEATNFL_HOME", tempfile.mkdtemp(prefix="beatnfl_test_"))

from beatnfl import export, optimizer, projections, storage  # noqa: E402
from beatnfl.config import Settings  # noqa: E402
from beatnfl.errors import StepError  # noqa: E402
from beatnfl.sources import dk  # noqa: E402

SAL = FIX / "DKSalaries_FIXTURE_2026wk6.csv"


def pool_from_fixture():
    ls = dk.load_from_file(SAL)
    p = projections.attach_status(ls.players, None)
    raw = pd.read_csv(SAL, encoding="utf-8-sig")
    proj = raw[["ID", "AvgPointsPerGame"]].rename(columns={"ID": "dk_id", "AvgPointsPerGame": "proj_paid"}).astype({"dk_id": str})
    return ls, projections.blend(p, Settings(), {"paid": proj})


# ---------------------------------------------------------------- IMPORT
def test_import_classic_csv_preserves_ids_salaries():  # IMPORT-01
    ls = dk.load_from_file(SAL)
    raw = pd.read_csv(SAL, encoding="utf-8-sig", dtype={"ID": str})
    m = ls.players.merge(raw, left_on="dk_id", right_on="ID")
    assert len(m) == len(raw) == len(ls.players)
    assert (m.salary == m.Salary).all()
    assert ls.players.dk_id.map(type).eq(str).all()
    assert ls.players.game.nunique() == 10
    assert set(ls.players.team) >= {"LA", "KC"}  # LAR normalised to LA
    assert ls.players.start_utc.str.endswith("Z").all()


def test_showdown_csv_rejected_with_explanation():
    txt = "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n" \
          "QB,A (1),A,1,CPT,12000,DAL@GB 10/18/2026 08:20PM ET,DAL,20\n"
    with pytest.raises(StepError) as e:
        dk.parse_salary_csv(txt)
    assert "Showdown" in e.value.what


def test_malformed_and_conflicting_rows():  # IMPORT-03
    with pytest.raises(StepError) as e:
        dk.parse_salary_csv("Player,Team,Proj\nX,DAL,10\n")
    assert "does not look like" in e.value.what
    hdr = "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
    row = 'WR,"Smith, Jr. (5)","Smith, Jr.",5,WR/FLEX,{s},DAL@GB 10/18/2026 01:00PM ET,DAL,10\n'
    with pytest.raises(StepError):
        dk.parse_salary_csv(hdr + row.format(s=5000) + row.format(s=6000))
    df, warns = dk.parse_salary_csv(hdr + row.format(s=5000) + row.format(s=-5).replace(",5,", ",6,"))
    assert len(df) == 1 and df.name.iloc[0] == "Smith, Jr." and warns  # quoted comma kept, bad salary rejected


def test_draftables_json_parse_and_duplicate_slots():
    data = json.loads((FIX / "dk_draftables.json").read_text())
    df, warns = dk.parse_draftables(data)
    assert df.dk_id.is_unique and "Disabled Guy" not in set(df.name)
    assert not warns  # same ID listed for WR and FLEX slots → one row, no warning
    first = dict(data["draftables"][1])
    data["draftables"].append({**first, "draftableId": 99999999, "rosterSlotId": 71})  # slot-specific ID
    df2, warns2 = dk.parse_draftables(data)
    assert len(df2) == len(df) and any("more than one draftable ID" in w for w in warns2)
    assert (df[df.team == "LA"].opp == "ARI").all()
    assert dk.validate_slate(df) == []


def test_lobby_main_slate_choice():
    sl = dk.parse_lobby(json.loads((FIX / "dk_lobby.json").read_text()))
    assert {s.draft_group_id for s in sl} == {"9001", "9002", "9004"}  # showdown excluded
    assert dk.pick_main_slate(sl).draft_group_id == "9001"  # featured, no suffix


def test_verify_auto_against_csv():
    ls = dk.load_from_file(SAL)
    a = ls.players.copy()
    ok, _ = dk.verify_against_csv(a, ls.players)
    assert ok
    a.loc[0, "salary"] += 100
    ok, issues = dk.verify_against_csv(a, ls.players)
    assert not ok and "salary" in issues[0]


# ---------------------------------------------------------------- NETWORK FAILURES (AUTO-02, mocked)
def test_network_failures_are_explained(monkeypatch):
    from beatnfl import net
    import requests

    class R:
        def __init__(self, code, text="x"):
            self.status_code, self.text, self.content, self.headers = code, text, text.encode(), {}

    monkeypatch.setattr(net.time, "sleep", lambda *_: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(403))
    with pytest.raises(StepError) as e:
        net.fetch("https://example.com/a", step="t", max_age_h=0)
    assert "HTTP 403" in e.value.why

    def boom(*a, **k):
        raise requests.Timeout("timed out")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(StepError) as e:
        net.fetch("https://example.com/b", step="t", max_age_h=0)
    assert "connection" in e.value.why.lower() or "down" in e.value.why.lower()

    monkeypatch.setattr(requests, "get", lambda *a, **k: R(200, "<html>login</html>"))
    with pytest.raises(StepError) as e:
        net.fetch_json("https://example.com/c", step="t", max_age_h=0)
    assert "not data" in e.value.what

    # cached copy is served but labeled stale, never fresh
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(200, '{"ok": 1}'))
    net.fetch("https://example.com/d", step="t", max_age_h=0)
    monkeypatch.setattr(requests, "get", boom)
    body, meta = net.fetch("https://example.com/d", step="t", max_age_h=0)
    assert meta.stale and meta.from_cache and "failed" in meta.note


# ---------------------------------------------------------------- PROJECTIONS
def test_missing_projection_is_nan_not_zero_and_out_players_excluded():
    ls, pool = pool_from_fixture()
    p = projections.attach_status(ls.players.assign(dk_status=["O"] + ["None"] * (len(ls.players) - 1)), None)
    b = projections.blend(p, Settings(), {"paid": pd.DataFrame({"dk_id": [p.dk_id.iloc[1]], "proj_paid": [10.0]})})
    assert pd.isna(b.proj.iloc[0]) and b.out.iloc[0]
    assert b.proj.iloc[1] == 10.0
    assert b.proj.iloc[2:].isna().all()


def test_blend_renormalises_and_questionable_haircut():
    ls = dk.load_from_file(SAL)
    p = projections.attach_status(ls.players.assign(dk_status=["Q"] + ["None"] * (len(ls.players) - 1)), None)
    ids = p.dk_id.iloc[:2]
    b = projections.blend(p, Settings(w_paid=0.5, w_sleeper=0.25, w_model=0.25), {
        "paid": pd.DataFrame({"dk_id": ids, "proj_paid": [20.0, 20.0]}),
        "model": pd.DataFrame({"dk_id": ids, "proj_model": [10.0, 10.0]})})
    assert abs(b.proj_raw.iloc[1] - (0.5 * 20 + 0.25 * 10) / 0.75) < 0.01
    assert abs(b.proj.iloc[0] - b.proj_raw.iloc[0] * 0.9) < 0.01


def test_paid_csv_column_detection():
    from beatnfl.sources import paidcsv
    txt = "Player,Pos,Team,Salary,DK Proj,DK Ownership\nJosh Allen,QB,BUF,8200,24.1,12%\nPuka Nacua,WR,LAR,8100,20.5,18%\n"
    df, mp = paidcsv.parse(txt)
    assert mp["projection"] == "DK Proj" and len(df) == 2
    assert df.key.iloc[1] == "WR|LA|puka nacua"


# ---------------------------------------------------------------- OPTIMIZER
def test_twenty_valid_distinct_lineups():  # OPT-01
    _, pool = pool_from_fixture()
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=20, min_unique=2, max_exposure=0.6))
    assert r.status == "complete" and len(r.lineups) == 20
    assert len({frozenset(lu) for lu in r.lineups}) == 20
    for lu in r.lineups:
        assert optimizer.validate(lu, pool) == []
    for a in range(20):
        for b in range(a):
            assert len(set(r.lineups[a]) & set(r.lineups[b])) <= 7
    counts = pd.Series([d for lu in r.lineups for d in lu]).value_counts()
    assert counts.max() <= 12  # floor(0.6*20)


def test_stack_rule_and_no_offense_vs_dst():
    _, pool = pool_from_fixture()
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=5, stack_min=2, bring_back=1))
    idx = pool.set_index("dk_id")
    for lu in r.lineups:
        rows = idx.loc[lu]
        qb = rows[rows.pos == "QB"].iloc[0]
        assert ((rows.team == qb.team) & rows.pos.isin(["WR", "TE"])).sum() >= 2
        assert ((rows.team == qb.opp) & rows.pos.isin(["RB", "WR", "TE"])).sum() >= 1
        dst = rows[rows.pos == "DST"].iloc[0]
        assert not ((rows.team == dst.opp) & (rows.pos != "DST")).any()


def test_impossible_rules_explained_not_padded():  # OPT-02
    _, pool = pool_from_fixture()
    qbs = pool[pool.pos == "QB"].dk_id.tolist()[:2]
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=5, locks=qbs))
    assert r.lineups == [] and r.status == "infeasible" and "No valid lineup" in r.diagnostics[0]
    small = pd.concat([pool[pool.pos.isin(["RB", "WR", "TE", "DST"])], pool[pool.pos == "QB"].head(2)])
    r = optimizer.build(small, optimizer.OptSettings(n_lineups=50, max_exposure=0.1, randomness=0))
    assert len(r.lineups) == 10 and r.status == "partial"  # 2 QBs × 5 uses each
    assert "Only 10 distinct lineups" in r.diagnostics[-1]
    assert len({frozenset(l) for l in r.lineups}) == len(r.lineups)


def test_validator_catches_bad_lineups():
    _, pool = pool_from_fixture()
    lu = optimizer.build(pool, optimizer.OptSettings(n_lineups=1)).lineups[0]
    assert optimizer.validate(lu[:8], pool)
    swapped = lu.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert any("can't fill" in e for e in optimizer.validate(swapped, pool))
    rich = pool[pool.pos != "DST"].nlargest(8, "salary").dk_id.tolist()


def test_single_entry_presets_give_best_lineup_first():
    _, pool = pool_from_fixture()
    idx = pool.set_index("dk_id")
    best = optimizer.build(pool, optimizer.OptSettings(n_lineups=1, randomness=0, stack_min=0)).lineups[0]
    for ct in ("cash", "se"):
        r = optimizer.build(pool, optimizer.OptSettings(**optimizer.preset(ct)))
        assert len(r.lineups) == 3 and r.status == "complete"
        pts = [idx.loc[lu].proj.sum() for lu in r.lineups]
        assert pts == sorted(pts, reverse=True)
        assert all(len(set(a) ^ set(b)) >= 4 for i, a in enumerate(r.lineups) for b in r.lineups[i + 1:])
        if ct == "cash":     # no stack rule → the single highest-projected lineup on the slate
            assert abs(pts[0] - idx.loc[best].proj.sum()) < 1e-6
    assert optimizer.preset("unknown") == optimizer.PRESETS["se"]
    s = Settings()
    assert s.contest == "se"


def test_flex_is_latest_kickoff():
    _, pool = pool_from_fixture()
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=10))
    idx = pool.set_index("dk_id")
    for lu in r.lineups:
        flex_t = idx.at[lu[7], "start_utc"]
        others = [idx.at[d, "start_utc"] for i, d in enumerate(lu) if i in (1, 2, 3, 4, 5, 6)]
        swappable = [t for i, t in zip((1, 2, 3, 4, 5, 6), others) if idx.at[lu[i], "pos"] == idx.at[lu[7], "pos"]]
        assert all(flex_t >= t for t in swappable)


# ---------------------------------------------------------------- SAVE / EXPORT
def test_save_reload_and_export_roundtrip():  # SAVE-01, EXPORT-01
    ls, pool = pool_from_fixture()
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=5))
    sid = storage.save_slate(ls, 2026, 6)
    assert storage.save_slate(ls, 2026, 6) == sid  # no duplicate snapshot
    run = storage.save_projection_run(sid, pool, "test", {"paid": "fixture"})
    set_id = storage.save_lineup_set("t", sid, run, {}, r.lineups, pool, r.status, r.diagnostics)
    d = storage.load_lineup_set(set_id)
    assert d["lineups"] == r.lineups
    _, pool2 = storage.get_projection_run(d["projection_run_id"])
    assert pool2.dk_id.tolist() == pool.dk_id.tolist()
    storage.rename_lineup_set(set_id, "renamed", d["revision"])
    with pytest.raises(RuntimeError):
        storage.rename_lineup_set(set_id, "stale", d["revision"])  # revision check
    path = export.dk_upload(r.lineups, pool, "test slate")
    rows = list(csv.reader(open(path, newline="")))
    assert rows[0] == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    assert rows[1:] == r.lineups
    bad = [r.lineups[0][:8] + [r.lineups[0][0]]]
    with pytest.raises(ValueError):
        export.dk_upload(bad, pool, "bad")


# ---------------------------------------------------------------- MODEL: no look-ahead (MODEL-01)
def _toy_stats():
    rng = np.random.default_rng(0)
    rows = []
    for wk in range(1, 9):
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            for i, pos in enumerate(["QB", "RB", "WR", "WR", "TE"]):
                rows.append({"player_id": f"{team}{i}", "player_display_name": f"{team} P{i}", "position": pos,
                             "season": 2025, "week": wk, "season_type": "REG", "game_id": f"g{wk}", "team": team,
                             "opponent_team": opp, "attempts": 30 if pos == "QB" else 0,
                             "passing_yards": 250 if pos == "QB" else 0, "carries": rng.integers(0, 15),
                             "rushing_yards": rng.integers(0, 80), "targets": rng.integers(0, 10),
                             "receptions": rng.integers(0, 7), "receiving_yards": rng.integers(0, 90),
                             "dk_pts": float(rng.integers(0, 30))})
    st = pd.DataFrame(rows)
    tg = pd.concat([pd.DataFrame({"team": t, "opp": o, "season": 2025, "week": range(1, 9), "pf": 21, "pa": 20,
                                  "spread": 0.0, "total": 44.0, "implied": 22.0, "home": h, "game_id": [f"g{w}" for w in range(1, 9)]})
                    for t, o, h in (("AAA", "BBB", 1), ("BBB", "AAA", 0))])
    return st, tg


def test_features_use_only_prior_games():
    from beatnfl.models import features as F
    st, tg = _toy_stats()
    f1 = F.build(F.tidy_stats(st), tg)
    st2 = st.copy()
    st2.loc[st2.week >= 6, ["dk_pts", "receiving_yards", "targets", "carries"]] = 999  # poison the future
    f2 = F.build(F.tidy_stats(st2), tg)
    k = ["player_id", "season", "week"]
    a = f1[f1.week <= 6].sort_values(k).reset_index(drop=True)
    b = f2[f2.week <= 6].sort_values(k).reset_index(drop=True)
    for c in F.FEATURES:
        pd.testing.assert_series_equal(a[c], b[c], check_names=False)


def test_vacated_share_when_teammate_misses():
    from beatnfl.models import features as F
    st, tg = _toy_stats()
    st = st[~((st.player_id == "AAA2") & (st.week == 5))]  # WR misses week 5
    f = F.build(F.tidy_stats(st), tg)
    v = f[(f.team == "AAA") & (f.week == 5)].vac_tgt.iloc[0]
    assert v > 0
    assert f[(f.team == "AAA") & (f.week == 4)].vac_tgt.iloc[0] == 0


# ---------------------------------------------------------------- PROPS
def test_props_board_offline(monkeypatch):
    from beatnfl import research
    from beatnfl.models.players import PlayerModel
    from beatnfl.sources import odds
    monkeypatch.setattr(odds, "events", lambda *a, **k: json.loads((FIX / "odds_events.json").read_text()))
    raw = json.loads((FIX / "odds_props_evt1.json").read_text())

    def fake_fetch(url, **k):
        return raw, type("M", (), {"stale": False})()
    monkeypatch.setattr(odds, "fetch_json", fake_fetch)
    pm = PlayerModel()
    rng = np.random.default_rng(1)
    preds = pd.DataFrame({"p_rec_yds": rng.uniform(20, 100, 2000)})
    act = pd.DataFrame({"rec_yds": preds.p_rec_yds + rng.normal(0, 25, 2000)})
    pm.set_residuals(preds.assign(**{f"p_{t}": np.nan for t in ["dk_pts"]}), act.assign(dk_pts=np.nan))
    fc = pd.DataFrame({"name": ["Drake London", "Bijan Robinson"], "team": ["ATL", "ATL"], "p_rec_yds": [85.0, 20.0],
                       "p_any_td": [0.3, 0.62], "game_id": ["g", "g"]})
    board, info = research.props_board("k", 2026, 6, fc, pm, ["player_reception_yds", "player_anytime_td"])
    dl = board[board.player == "Drake London"].iloc[0]
    assert dl.model_p_over > 0.6 and 0.45 < dl.book_p_over < 0.56 and dl.model_lean == "Over"
    bj = board[board.player == "Bijan Robinson"].iloc[0]
    assert abs(bj.model_p_over - 0.62) < 1e-9
    assert len(storage.quotes("prop")) >= 2


def test_old_salary_file_is_refused_not_substituted(tmp_path):
    from beatnfl import pipeline
    old = SAL.read_text(encoding="utf-8-sig").replace("10/18/2026", "10/04/2020")
    r = pipeline.run_week(Settings(), salary_file=old.encode(), use_model=False)
    assert r.pool is None and r.steps[-1].status == "fail" and "already been played" in r.steps[-1].message


def test_pasted_table_from_browser():
    from beatnfl.sources import paidcsv
    pasted = "Player\tTeam\tPos\tSalary\tProj\nJosh Allen\tBUF\tQB\t$8,200\t24.1\nPuka Nacua\tLAR\tWR\t$8,100\t20.5\n"
    df, mp = paidcsv.parse(pasted)
    assert mp["projection"] == "Proj" and df.proj_paid.tolist() == [24.1, 20.5]


# ---------------------------------------------------------------- YAHOO (fixture shapes from Yahoo's documented JSON)
def _yp(key, name, team, pos, slot, status=None, elig=None, pct=None):
    info = [{"player_key": key}, {"player_id": key[-2:]}, {"name": {"full": name}}, {"editorial_team_abbr": team},
            {"display_position": pos}, {"eligible_positions": [{"position": e} for e in (elig or [pos])]}]
    if status:
        info.append({"status": status})
    p = [info, {"selected_position": [{"coverage_type": "week"}, {"position": slot}]}]
    if pct is not None:
        p.append({"percent_owned": [{"coverage_type": "week"}, {"value": pct}, {"delta": "1.5"}]})
    return {"player": p}


def test_yahoo_parse_and_start_sit():
    from beatnfl.sources import yahoo
    players = [_yp("461.p.01", "Josh Allen", "Buf", "QB", "QB"), _yp("461.p.02", "Bijan Robinson", "Atl", "RB", "RB"),
               _yp("461.p.03", "James Cook", "Buf", "RB", "BN"), _yp("461.p.04", "Puka Nacua", "LAR", "WR", "WR", status="Q"),
               _yp("461.p.05", "Drake London", "Atl", "WR", "WR"), _yp("461.p.06", "Kyle Pitts", "Atl", "TE", "TE"),
               _yp("461.p.07", "Tony Pollard", "Ten", "RB", "W/R/T"), _yp("461.p.08", "Injured Guy", "NYJ", "WR", "BN", status="O"),
               _yp("461.p.09", "Chiefs", "KC", "DEF", "DEF")]
    doc = {"fantasy_content": {"team": [[{"team_key": "461.l.777.t.3"}, {"name": "Harry's Team"}],
                                        {"roster": {"0": {"players": {str(i): p for i, p in enumerate(players)} | {"count": len(players)}}}}]}}
    rows = yahoo.parse_roster(doc)
    assert len(rows) == 9 and rows[3]["team"] == "LA" and rows[3]["status"] == "Q" and rows[2]["slot"] == "BN"
    proj = {"Josh Allen": 25, "Bijan Robinson": 20, "James Cook": 19, "Puka Nacua": 18, "Drake London": 16,
            "Kyle Pitts": 9, "Tony Pollard": 12, "Injured Guy": 30}
    for r in rows:
        r["proj"] = proj.get(r["name"])
    sug = yahoo.suggest(rows, [("QB", 1), ("WR", 2), ("RB", 2), ("TE", 1), ("W/R/T", 1)])
    names = [s["player"] for s in sug]
    assert "Injured Guy" not in names           # ruled out never suggested
    assert sug[-1]["player"] == "Tony Pollard"  # flex filled last with best remaining
    assert {"James Cook", "Bijan Robinson"} <= set(names)
    teams = yahoo.parse_teams({"fantasy_content": {"users": {"0": {"user": [{"guid": "x"}, {"games": {"0": {"game": [
        {"game_key": "461"}, {"teams": {"0": {"team": [[{"team_key": "461.l.777.t.3"}, {"name": "Harry's Team"}]]}, "count": 1}}]}}}]}}}})
    assert teams == [{"team_key": "461.l.777.t.3", "name": "Harry's Team", "league_key": "461.l.777"}]
    slots = yahoo.parse_roster_slots({"fantasy_content": {"league": [{}, {"settings": [{"roster_positions": [
        {"roster_position": {"position": "QB", "count": 1}}, {"roster_position": {"position": "W/R/T", "count": 1}},
        {"roster_position": {"position": "BN", "count": 6, "is_starting_position": 0}}]}]}]}})
    assert slots == [("QB", 1), ("W/R/T", 1)]


def test_yahoo_injuries_merge_worst_status_wins():
    from beatnfl.sources import yahoo
    ls = dk.load_from_file(SAL)
    pool = projections.attach_status(ls.players, None)
    first = pool.iloc[0]
    doc = {"x": [_yp("461.p.1", first["name"], "LAR" if first.team == "LA" else first.team, first.pos, "BN", status="O", pct="64")]}
    yp = pd.DataFrame(yahoo.parse_players(doc))
    assert yp.pct_rostered.iloc[0] == 64.0
    out, n = projections.attach_external_status(pool, yp, "yahoo")
    assert n == 1 and out.out.iloc[0] and out.status.iloc[0] == "O" and out.pct_rostered_yahoo.iloc[0] == 64.0
    assert not out.out.iloc[1:].any()


def test_self_update_installs_and_rolls_back(tmp_path, monkeypatch):
    import shutil, subprocess as sp, zipfile
    from beatnfl import update
    app = tmp_path / "app"
    app.mkdir()
    real = Path(update.__file__).resolve().parent.parent
    for n in ("beatnfl", "app.py", "requirements.txt"):
        (shutil.copytree if (real / n).is_dir() else shutil.copy2)(real / n, app / n)
    # the "GitHub" version: same code plus a marker file in the package
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in app.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, "owner-beatnfl-abc123/" + str(f.relative_to(app)))
        z.writestr("owner-beatnfl-abc123/beatnfl/NEW_MARKER.txt", "v2")

    class R:
        def __init__(self, body=None, js=None):
            self.status_code, self.content, self._js = 200, body, js
        def json(self):
            return self._js
        def raise_for_status(self):
            pass

    def fake_get(url, **k):
        if url.endswith("/commits/main"):
            return R(js={"sha": "abc123", "commit": {"committer": {"date": "2026-10-12T00:00:00Z"}, "message": "Add thing"}})
        return R(body=buf.getvalue())
    monkeypatch.setattr(update, "APP_DIR", app)
    monkeypatch.setattr(update, "STAMP", app / ".installed_commit")
    monkeypatch.setattr(update.requests, "get", fake_get)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, "ok", ""))
    assert update.check("owner/beatnfl")["update_available"]
    res = update.apply("owner/beatnfl")
    assert res["changed"] and (app / "beatnfl" / "NEW_MARKER.txt").exists() and update.installed() == "abc123"
    assert not update.check("owner/beatnfl")["update_available"]
    # a failing self-test restores the previous code
    (app / ".installed_commit").write_text("old")
    (app / "beatnfl" / "NEW_MARKER.txt").unlink()
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 1, "1 failed", ""))
    with pytest.raises(StepError):
        update.apply("owner/beatnfl")
    assert not (app / "beatnfl" / "NEW_MARKER.txt").exists() and update.installed() == "old"


def test_team_matchup_priors_use_only_earlier_games():
    from beatnfl.models.features import team_metric_priors, OFF_M, DEF_M
    rows = []
    for w in range(1, 7):
        r = {"team": "AAA", "season": 2026, "week": w, "off_rz_tds": 1, "off_rz_trips": 2, "def_rz_tds": 1, "def_rz_trips": 2}
        r.update({c: float(w) for c in OFF_M + DEF_M})
        rows.append(r)
    tm = pd.DataFrame(rows)
    a = team_metric_priors(tm)
    tm2 = tm.copy()
    tm2.loc[tm2.week == 6, OFF_M] = 999.0
    b = team_metric_priors(tm2)
    assert a.loc[a.week == 1, "e_off_epa"].isna().all()           # nothing known before week 1
    assert a.loc[a.week == 6, "e_off_epa"].iloc[0] < 6            # week 6 prior excludes week 6
    pd.testing.assert_frame_equal(a, b)                            # changing week 6 results changes no prior


def test_identity_with_sleeper_including_team_changes():
    from beatnfl.models.train import identify
    from beatnfl.names import loose_key, match_key
    hist = pd.DataFrame({"player_id": ["00-1", "00-2", "00-3", "00-4"], "name": ["Josh Allen", "Davante Adams", "Rookie Guy", "Same Name"],
                         "position": ["QB", "WR", "RB", "WR"], "team": ["BUF", "NYJ", "KC", "DAL"], "season": 2026, "week": 5})
    sl = pd.DataFrame({"name": ["Josh Allen", "Davante Adams", "Nobody Here", "Same Name"], "pos": ["QB", "WR", "RB", "WR"],
                       "team": ["BUF", "LA", "NE", "DAL"]})
    sl["key"] = [match_key(n, p, t) for n, p, t in zip(sl.name, sl.pos, sl.team)]
    sp = pd.DataFrame({"name": ["Josh Allen", "Davante Adams", "Other"], "pos": ["QB", "WR", "TE"], "team": ["BUF", "LV", "SF"],
                       "gsis_id": ["00-1", "00-2", None]})  # Sleeper still lists Adams on his old team
    sp["key"] = [match_key(n, p, t) for n, p, t in zip(sp.name, sp.pos, sp.team)]
    sp["lkey"] = [loose_key(n, p) for n, p in zip(sp.name, sp.pos)]
    assert identify(sl, sp, hist) == ["00-1", "00-2", None, "00-4"]
    assert identify(sl, None, hist) == ["00-1", "00-2", None, "00-4"]   # works without Sleeper too
    assert identify(sl, sp.drop(columns="gsis_id"), hist) == ["00-1", "00-2", None, "00-4"]


def test_status_merge_with_unordered_index():
    """JSON-route slates arrive with a non-contiguous index; injury merges must not misalign."""
    ls = dk.load_from_file(SAL)
    shuffled = ls.players.sample(frac=1, random_state=3)  # index out of order, like a sorted/deduped feed
    sp = shuffled.head(5).assign(injury="O", injury_raw="Out", injury_body="", depth_order=1, depth_pos="", news_utc=None,
                                 team=lambda d: d.team.where(d.pos == "DST", "NYJ"))  # force the name-only path
    sp["lkey"] = [__import__("beatnfl.names", fromlist=["x"]).loose_key(n, p) for n, p in zip(sp.name, sp.pos)]
    sp["key"] = "nomatch|" + sp.dk_id
    out = projections.attach_status(shuffled, sp)
    hit = out[out.dk_id.isin(sp.dk_id)]
    assert hit.out.all() and out.out.sum() == 5


def test_props_view_readable_consensus_and_filters():
    from beatnfl import research
    rows = []
    for bk, ov, un in (("draftkings", -115, -105), ("fanduel", -110, -110), ("williamhill_us", -120, 100)):
        rows.append({"game": "CHI @ ATL", "player": "Drake London", "stat": "rec_yds", "book": bk, "line": 74.5, "price": ov,
                     "under_price": un, "model_pred": 85.0, "book_p_over": 0.52, "model_p_over": 0.63, "source_ts": ""})
    for bk, pr, bp in (("draftkings", 190, 0.345), ("fanatics", 175, 0.364)):
        rows.append({"game": "BAL @ CLE", "player": "Denzel Boston", "stat": "any_td", "book": bk, "line": 0.5, "price": pr,
                     "under_price": None, "model_pred": 0.276, "book_p_over": bp, "model_p_over": 0.276, "source_ts": ""})
    rows.append({"game": "BAL @ CLE", "player": "Chris Moore", "stat": "any_td", "book": "betonlineag", "line": 0.5, "price": 1400,
                 "under_price": None, "model_pred": 0.163, "book_p_over": 0.067, "model_p_over": 0.163, "source_ts": ""})
    b = pd.DataFrame(rows)
    v = research.props_view(b)
    assert len(v) == 3                                         # one consensus row per player/prop
    dl = v[v.Player == "Drake London"].iloc[0]
    assert dl.Prop == "Rec yds 74.5" and dl.Book == "3 books" and dl["Model %"] == 63 and dl.Lean == "Over ★"
    db = v[v.Player == "Denzel Boston"].iloc[0]
    assert db.Lean == "—" and db.Prop.endswith("*")   # TD 'No' leans are not shown (mostly book margin)
    assert v[v.Player == "Chris Moore"].iloc[0].Lean == "Yes ★"
    g = research.props_view(b, game="BAL @ CLE", book="Fanatics")
    assert list(g.Player) == ["Denzel Boston"] and g.iloc[0].Odds == 175
    s = research.props_view(b, book="Caesars")
    assert list(s.Book) == ["Caesars"] and s.iloc[0].Odds == -120
    assert list(research.props_view(b, sort="Sportsbook").Book)[0] == "1 book"


# ---------------------------------------------------------------- review-board fixes (v1.5)
def test_build_does_not_save_and_save_skips_duplicates():
    from beatnfl import pipeline
    ls, pool = pool_from_fixture()
    sid = storage.save_slate(ls, 2026, 6)
    run_id = storage.save_projection_run(sid, pool, "test", {})
    run = pipeline.WeekRun.__new__(pipeline.WeekRun)
    run.pool, run.slate_id, run.run_id, run.season, run.week = pool, sid, run_id, 2026, 6
    before = len(storage.list_lineup_sets())
    out = pipeline.build_lineups(run, optimizer.OptSettings(n_lineups=4, randomness=0))
    assert out["set_id"] is None and len(storage.list_lineup_sets()) == before      # building never saves
    proj = pool.set_index("dk_id").proj
    tot = [proj.loc[lu].sum() for lu in out["result"].lineups]
    assert tot == sorted(tot, reverse=True)                                          # rank 1 = best projection
    a = pipeline.save_lineups(run, out, "x", export_files=False)
    b = pipeline.save_lineups(run, out, "x again", export_files=False)
    assert a["set_id"] == b["set_id"] and b["duplicate"] and len(storage.list_lineup_sets()) == before + 1
    assert storage.duplicate_sets() == {} or a["set_id"] not in storage.duplicate_sets()


def test_game_probabilities_market_and_model():
    from beatnfl.sources import odds
    rows = []
    for bk, hs, as_, mh, ma, ov, un in (("dk", -110, -110, -150, 130, -105, -115), ("fd", -108, -112, -155, 135, -110, -110)):
        for mk, side, line, price in (("spreads", "KC", -3.0, hs), ("spreads", "LAC", 3.0, as_), ("h2h", "KC", None, mh),
                                      ("h2h", "LAC", None, ma), ("totals", "Over", 44.5, ov), ("totals", "Under", 44.5, un)):
            rows.append({"home": "KC", "away": "LAC", "book": bk, "market": mk, "side": side, "line": line, "price": price})
    m = odds.market_summary(pd.DataFrame(rows)).iloc[0]
    assert m.spread_nv == 3.0 and m.total == 44.5 and m.books == 2
    assert 0.49 < m.p_home_cover < 0.51 and 0.58 < m.p_home_win < 0.61 and 0.48 < m.p_over < 0.51


def test_app_recovers_from_stale_modules_after_update(tmp_path, monkeypatch):
    """Server still holding the previous version's modules (the v1.5.0 'no attribute duplicate_sets' crash)."""
    from streamlit.testing.v1 import AppTest
    import sys
    import beatnfl
    from beatnfl import storage as old_storage
    saved = {k: v for k, v in sys.modules.items() if k == "beatnfl" or k.startswith("beatnfl.")}
    for k in saved:   # put the original modules back afterwards so later tests share one copy
        monkeypatch.setitem(sys.modules, k, v := saved[k])
    monkeypatch.setattr(beatnfl, "__version__", "0.0-old")
    monkeypatch.delattr(old_storage, "duplicate_sets")
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert hasattr(sys.modules["beatnfl.storage"], "duplicate_sets")
    finally:   # forget modules first imported during the reload, so later tests all share the original copies
        for k in [m for m in sys.modules if (m == "beatnfl" or m.startswith("beatnfl.")) and m not in saved]:
            del sys.modules[k]



def test_restart_pattern_cannot_match_itself():
    import re
    import inspect
    from beatnfl import update
    src = inspect.getsource(update.restart)
    pat = re.search(r"pkill -f '([^']+)'", src).group(1)
    assert pat.startswith("[s]treamlit")                       # regex trick: never matches its own command line
    assert not re.search(pat.replace("{APP_DIR}", "/a/BeatNFL"), "sleep 1; pkill -f '" + pat + "'")


# ---------------------------------------------------------------- FantasyPros API (v1.6)
def _fp_payload(pos, players, tier="premium", count=None):
    return {"season": "2026", "week": "6", "count": str(count if count is not None else len(players)), "positions": pos,
            "scoring": "STD", "players": players, "public_api_limited": True, "tier": tier}


def _fp_fake(ls, tier="premium", short=False):
    """FantasyPros-shaped responses (field names as recorded from the live API) for slate players."""
    pl = ls.players
    by = {}
    for pos in ("QB", "RB", "WR", "TE", "DST"):
        rows = []
        for r in pl[pl.pos == pos].head(12).itertuples():
            if pos == "DST":
                st = {"points": 8.0, "points_ppr": 8.0, "points_half": 8.0, "def_sack": 2.8, "def_int": 0.9, "def_td": 0.15,
                      "def_pa": 21.5, "def_tyda": 320, "def_safety": 0.05, "def_ff": 0.8, "def_fr": 0.6, "def_retd": 0.05}
                rows.append({"fpid": 1, "mflid": 1, "name": f"{r.team} Defense", "position_id": "DST", "team_id": r.team,
                             "filename": "x.php", "stats": st})
            elif pos == "QB":
                st = {"points": 18.0, "points_ppr": 18.0, "points_half": 18.0, "pass_att": 34, "pass_cmp": 22.5,
                      "pass_yds": 245.0, "pass_tds": 1.7, "pass_ints": 0.7, "rush_att": 4, "rush_yds": 22.0, "rush_tds": 0.2,
                      "fumbles": 0.2, "2pt_tds": 0}
                rows.append({"fpid": 2, "name": r.name, "position_id": "QB", "team_id": "JAC" if r.team == "JAX" else r.team, "stats": st})
            else:
                st = {"points": 9.0, "points_ppr": 13.5, "points_half": 11.2, "rush_att": 3, "rush_yds": 12.0, "rush_tds": 0.05,
                      "rec_rec": 4.5, "rec_yds": 55.0, "rec_tds": 0.35, "fumbles": 0.05, "ret_tds": 0, "2pt_tds": 0}
                rows.append({"fpid": 3, "name": r.name, "position_id": pos, "team_id": r.team, "stats": st})
        by[pos] = _fp_payload(pos, rows, tier, count=len(rows) + (5 if short else 0))
    return by


def test_fantasypros_api_dk_points_and_matching(monkeypatch):
    from beatnfl.sources import fantasypros
    from beatnfl.net import Meta
    ls, pool = pool_from_fixture()
    fake = _fp_fake(ls)
    calls = []

    def fetch_json(url, params=None, headers=None, **kw):
        calls.append((url, params, headers))
        return fake[params["position"]], Meta(url, 0, False, False)
    monkeypatch.setattr(fantasypros, "fetch_json", fetch_json)
    df, info = fantasypros.projections("KEY", 2026, 6)
    assert all(h == {"x-api-key": "KEY"} for _, _, h in calls) and calls[0][0].endswith("/nfl/2026/projections")
    assert {p["position"] for _, p, _ in calls} == {"QB", "RB", "WR", "TE", "DST"} and all(p["week"] == 6 for _, p, _ in calls)
    qb = df[df.pos == "QB"].iloc[0].proj_paid
    # 245*.04 + 1.7*4 - .7 + 22*.1 + .2*6 - .2 + bonus(P(300+)≈.2 → ~0.6)
    assert 18.5 < qb < 20.5
    wr = df[df.pos == "WR"].iloc[0].proj_paid
    assert 13.4 < wr < 14.5                                   # PPR-like, plus a small 100-yd bonus chance
    assert df[df.pos == "DST"].proj_paid.between(4, 9).all()
    m, n = projections.match_external(pool, df, "proj_paid")
    assert n == len(df)                                       # every row matched, DSTs by team, JAC→JAX
    assert info["tier"] == "premium" and not info["limited"] and not info["short"]


def test_fantasypros_sample_key_and_bad_key_are_explained(monkeypatch):
    from beatnfl.sources import fantasypros
    from beatnfl.net import Meta
    ls, _ = pool_from_fixture()
    fake = _fp_fake(ls, tier="free", short=True)
    monkeypatch.setattr(fantasypros, "fetch_json", lambda url, params=None, **kw: (fake[params["position"]], Meta(url, 0, False, False)))
    _, info = fantasypros.projections("K", 2026, 6)
    assert info["limited"] and info["short"]

    def refuse(url, **kw):
        raise StepError("x", "Could not download", detail="HTTP 403: Forbidden")
    monkeypatch.setattr(fantasypros, "fetch_json", refuse)
    with pytest.raises(StepError) as e:
        fantasypros.projections("BAD", 2026, 6)
    assert "refused the API key" in e.value.what


def test_best_weights_finds_the_better_source():
    from beatnfl.backtest import best_weights
    rng = np.random.default_rng(1)
    y = rng.normal(15, 6, 3000)
    d = pd.DataFrame({"dk_pts": y, "good": y + rng.normal(0, 2, 3000), "bad": y + rng.normal(0, 8, 3000)})
    w, mae = best_weights(d, ["good", "bad"])
    assert w["good"] >= 0.85 and abs(sum(w.values()) - 1) < 1e-6
    assert mae <= float((d.good - d.dk_pts).abs().mean()) + 1e-9


def test_blend_uses_position_weights_and_never_drops_a_covered_player():
    ls, pool = pool_from_fixture()
    p = pool.drop(columns=[c for c in pool if c.startswith("proj")] + ["value", "sources", "n_sources"], errors="ignore")
    ids = p.dk_id.tolist()
    paid = pd.DataFrame({"dk_id": ids, "proj_paid": 10.0})
    model = pd.DataFrame({"dk_id": ids[:-5], "proj_model": 20.0})       # last 5 covered only by FantasyPros
    s = Settings()
    s.pos_weights = {"QB": {"paid": 0.0, "sleeper": 0.0, "model": 1.0}, "WR": {"paid": 0.75, "sleeper": 0.0, "model": 0.25}}
    b = projections.blend(p, s, {"paid": paid, "model": model}).set_index("dk_id")
    act = b[~b.out & (b.status != "Q")]
    qb = act[(act.pos == "QB") & act.proj_model.notna()]
    wr = act[(act.pos == "WR") & act.proj_model.notna()]
    rb = act[(act.pos == "RB") & act.proj_model.notna()]
    assert (qb.proj == 20.0).all() and (wr.proj == 12.5).all()
    assert np.allclose(rb.proj, (0.5 * 10 + 0.25 * 20) / 0.75, atol=0.01)                # no position weights → globals
    tail = act.loc[act.index.isin(ids[-5:])]
    assert tail.proj.notna().all()                                         # only-FantasyPros players keep a number


def test_fantasypros_stats_mix_into_forecasts():
    from beatnfl import research
    from beatnfl.names import match_key
    fc = pd.DataFrame({"name": ["Drake London", "Bijan Robinson"], "position": ["WR", "RB"], "team": ["ATL", "ATL"],
                       "p_rec_yds": [70.0, 30.0], "p_rush_yds": [np.nan, 80.0]})
    fp = pd.DataFrame({"name": ["Drake London"], "pos": ["WR"], "team": ["ATL"], "fp_rec_yds": [90.0], "fp_rush_yds": [None]})
    fp["key"] = [match_key(n, p, t) for n, p, t in zip(fp.name, fp.pos, fp.team)]
    out, n = research.blend_fp_stats(fc, fp, {"rec_yds": 0.5, "rush_yds": 0.4})
    assert n == 1 and out.p_rec_yds.tolist() == [80.0, 30.0] and out.p_rush_yds.iloc[1] == 80.0


# ---------------------------------------------------------------- midweek picks + CLV (v1.8)
def _board_rows():
    rows = []
    # Drake London rec yds 74.5: model 63% vs market ~52% → Over; best price is FanDuel +105
    for bk, ov, un in (("draftkings", -115, -105), ("fanduel", 105, -125), ("betmgm", -110, -110)):
        rows.append({"game": "CHI @ ATL", "player": "Drake London", "stat": "rec_yds", "market": "player_reception_yds",
                     "book": bk, "line": 74.5, "price": ov, "under_price": un, "model_pred": 85.0, "model_median": 82.0,
                     "book_p_over": odds_nv(ov, un), "model_p_over": 0.63, "event_id": "e1", "commence": "2099-10-18T17:00:00Z"})
    # Bijan rush yds 80.5: model says Under (38% over vs 50%)
    for bk, ov, un in (("draftkings", -110, -110), ("fanduel", -112, -108)):
        rows.append({"game": "CHI @ ATL", "player": "Bijan Robinson", "stat": "rush_yds", "market": "player_rush_yds",
                     "book": bk, "line": 80.5, "price": ov, "under_price": un, "model_pred": 70.0, "model_median": 69.0,
                     "book_p_over": odds_nv(ov, un), "model_p_over": 0.38, "event_id": "e1", "commence": "2099-10-18T17:00:00Z"})
    # Kyle Pitts rec 4.5: model only 2 pts off → not flagged
    rows.append({"game": "CHI @ ATL", "player": "Kyle Pitts", "stat": "rec", "market": "player_receptions", "book": "draftkings",
                 "line": 4.5, "price": -110, "under_price": -110, "model_pred": 4.6, "model_median": 4.5,
                 "book_p_over": 0.5, "model_p_over": 0.52, "event_id": "e1", "commence": "2099-10-18T17:00:00Z"})
    return pd.DataFrame(rows)


def odds_nv(o, u):
    from beatnfl.sources.odds import implied_prob, no_vig
    return no_vig(implied_prob(o), implied_prob(u))


def test_midweek_picks_need_fantasypros_agreement_and_take_best_price():
    from beatnfl import research
    b = _board_rows()
    fp_yes = pd.DataFrame({"name": ["Drake London", "Bijan Robinson"], "pos": ["WR", "RB"], "fp_rec_yds": [80.0, None],
                           "fp_rush_yds": [None, 72.0]})
    p = research.midweek_picks(b, fp_yes, min_diff=4, sides=None)
    assert set(p.player) == {"Drake London", "Bijan Robinson"}             # Pitts below the 4-point bar
    dl = p[p.player == "Drake London"].iloc[0]
    assert dl.side == "Over" and dl.book == "fanduel" and dl.price == 105    # best price across books
    bj = p[p.player == "Bijan Robinson"].iloc[0]
    assert bj.side == "Under" and bj.book == "fanduel" and bj.price == -108
    fp_no = fp_yes.assign(fp_rec_yds=[70.0, None])                           # FantasyPros disagrees on London
    assert "Drake London" not in set(research.midweek_picks(b, fp_no, min_diff=4, sides=None).player)
    assert len(research.midweek_picks(b, None, min_diff=4, sides=None)) == 2                         # no FantasyPros key → model only
    default = research.midweek_picks(b, fp_yes)                            # the rule under test: Unders, 10+ pts
    assert list(default.player) == ["Bijan Robinson"] and (default.side == "Under").all()
    v = research.picks_view(p)
    assert {"Best odds", "Book", "Market %", "Model %", "Edge at best price %"} <= set(v.columns)


def test_picks_logged_once_and_graded_with_clv(monkeypatch):
    from beatnfl import research
    from beatnfl.models import features
    p = research.midweek_picks(_board_rows(), None, min_diff=4, sides=None).assign(season=2026, week=6, commence="2026-10-11T17:00:00Z")
    assert storage.save_picks(p) == 2 and storage.save_picks(p) == 0        # first price kept, no duplicates
    # closing prices: London moved to -130 over everywhere (we took +105 → beat the close)
    close = pd.DataFrame([{"event_id": "e1", "book": bk, "market": "player_reception_yds", "player": "Drake London",
                           "side": sd, "line": 74.5, "price": pr, "source_ts": f"t{bk}{sd}"}
                          for bk in ("draftkings", "fanduel") for sd, pr in (("Over", -130), ("Under", 110))])
    storage.save_prop_prices(close)
    with storage.db() as c:   # mark them as seen before kickoff
        c.execute("UPDATE prop_prices SET retrieved_utc='2026-10-11T15:00:00+00:00'")
    stats = pd.DataFrame({"name": ["Drake London", "Bijan Robinson"], "season": [2026, 2026], "week": [6, 6],
                          "rec_yds": [91.0, 10.0], "rush_yds": [0.0, 95.0], "any_td": [0, 1]})
    monkeypatch.setattr(research.nv, "weekly_stats", lambda *a, **k: stats)
    monkeypatch.setattr(features, "tidy_stats", lambda d: d)
    tbl, sm = research.grade_picks()
    dl = tbl[tbl.player == "Drake London"].iloc[0]
    assert dl.result == "won" and abs(dl.units - 1.05) < 1e-9 and dl.clv > 5   # ~55% close vs 48.8% taken
    bj = tbl[tbl.player == "Bijan Robinson"].iloc[0]
    assert bj.result == "lost" and bj.units == -1 and pd.isna(bj.clv)          # no closing price recorded
    assert sm["won"] == 1 and sm["lost"] == 1 and sm["clv_n"] == 1 and sm["beat_close"] == 100.0


def test_mac_schedule_plists():
    from beatnfl import schedule
    p = schedule.plist("com.beatnfl.midweek", tz="America/New_York")
    assert p["ProgramArguments"][-3:] == ["beatnfl", "midweek", "--notify"]
    assert p["StartCalendarInterval"] == [{"Weekday": 3, "Hour": 19, "Minute": 0}]
    den = schedule.plist("com.beatnfl.gamecheck", tz="America/Denver")["StartCalendarInterval"]
    assert {"Weekday": 0, "Hour": 9, "Minute": 50} in den       # 11:50 AM Eastern = 9:50 AM in Denver
    c = schedule.plist("com.beatnfl.close", tz="America/New_York")
    assert {t["Weekday"] for t in c["StartCalendarInterval"]} == {0, 1, 4}
    import plistlib
    assert plistlib.loads(plistlib.dumps(p))["Label"] == "com.beatnfl.midweek"


def test_no_clv_before_kickoff():
    from beatnfl import research
    p = research.midweek_picks(_board_rows(), None, min_diff=4, sides=None).assign(season=2099, week=1, commence="2099-09-10T17:00:00Z",
                                                            player=lambda d: d.player + " Future")
    storage.save_picks(p)
    tbl, _ = research.grade_picks()
    assert tbl[tbl.season == 2099].clv.isna().all()


# ---------------------------------------------------------------- props backtest on historical lines (v1.9)
def test_props_backtest_flow_budget_and_grading(monkeypatch):
    from beatnfl import props_backtest as pb
    from beatnfl import net
    from beatnfl.sources import nflverse as nv, odds
    sched = pd.DataFrame({"season": [2025] * 2, "game_type": ["REG"] * 2, "week": [1, 2], "home_team": ["ATL", "KC"],
                          "away_team": ["CHI", "LAC"], "gameday": ["2025-09-07", "2025-09-14"], "gametime": ["13:00", "16:25"]})
    monkeypatch.setattr(nv, "schedules", lambda **k: sched)

    class PM:
        def prob_over(self, stat, pred, line):
            return 0.70 if pred > line else 0.30
    te = pd.DataFrame({"name": ["Drake London", "Rashee Rice"], "team": ["ATL", "KC"], "week": [1, 2],
                       "p_rec_yds": [90.0, 40.0], "rec_yds": [101.0, 70.0]})
    monkeypatch.setattr(pb, "season_model", lambda season, **k: (PM(), te))
    ev = {1: [{"id": "e1", "home_team": "Atlanta Falcons", "away_team": "Chicago Bears", "commence_time": "2025-09-07T17:00:00Z"}],
          2: [{"id": "e2", "home_team": "Kansas City Chiefs", "away_team": "Los Angeles Chargers", "commence_time": "2025-09-14T20:25:00Z"}]}
    monkeypatch.setattr(odds, "hist_events", lambda key, at, **k: ev[1] if at.startswith("2025-09-03") else ev[2])

    def props(key, eid, at, markets, **k):
        player, line = ("Drake London", 74.5) if eid == "e1" else ("Rashee Rice", 60.5)
        close = not at.endswith("T23:00:00Z")
        o, u = ((-130, 110) if close else (-110, -110)) if eid == "e1" else ((-110, -110))
        rows = [{"event_id": eid, "book": b, "market": "player_reception_yds", "stat": "rec_yds", "player": player,
                 "side": s, "line": line, "price": p} for b in ("draftkings", "fanduel")
                for s, p in (("Over", o + (10 if b == "fanduel" and not close else 0)), ("Under", u))]
        return pd.DataFrame(rows), at
    monkeypatch.setattr(odds, "hist_props", props)
    monkeypatch.setitem(net.QUOTA, "remaining", 20000)
    res = pb.run("k", 2025, ["player_reception_yds"], reserve=4000)
    assert res["status"] == "complete" and res["props_matched"] == 2
    p4 = res["picks"][0]
    assert p4["n"] == 2 and p4["won"] == 1 and p4["lost"] == 1      # London over won (+100 at FanDuel), Rice under lost
    assert abs(p4["units"] - 0.0) < 0.01
    assert p4["clv_n"] == 2 and p4["clv_avg"] is not None
    # the budget guard pauses before spending past the reserve
    monkeypatch.setitem(net.QUOTA, "remaining", 4010)
    assert pb.run("k", 2025, ["player_reception_yds"], reserve=4000)["status"] == "paused"


def test_backtest_summary_counts_one_prop_per_player_game():
    from beatnfl import props_backtest as pb
    rows = []
    for ln, books in ((74.5, 3), (72.5, 1), (76.5, 1)):          # main line + two alternate lines
        rows.append({"event_id": "e", "player": "Drake London", "stat": "rec_yds", "line": ln, "books": books,
                     "p_over": .5, "close_p_over": .52, "model_p_over": .65, "actual": 90.0,
                     "best_over": -110.0, "best_under": -110.0})
    s = pb.summarize(pd.DataFrame(rows * 1))
    assert s["n"] == 1 and s["picks"][0]["n"] == 1 and s["picks"][0]["sides"]["over"]["n"] == 1
    assert s["picks"][0]["moved_toward"] == 100.0


# ---------------------------------------------------------------- sportsbook lines in the model (v1.11)
def test_book_lines_move_model_projection(monkeypatch):
    from beatnfl import market
    from beatnfl.sources import odds
    pool = pd.DataFrame({"dk_id": ["1", "2", "3"], "name": ["Drake London", "Bijan Robinson", "Nobody Lines"],
                         "team": ["ATL", "ATL", "CHI"]})
    monkeypatch.setattr(odds, "events", lambda *a, **k: [{"id": "e1", "home_team": "Atlanta Falcons", "away_team": "Chicago Bears"}])
    rows = [{"event_id": "e1", "book": b, "market": m, "stat": st, "player": pl, "side": sd, "line": ln, "price": pr}
            for b in ("draftkings", "fanduel")
            for (m, st, pl, sd, ln, pr) in (("player_reception_yds", "rec_yds", "Drake London", "Over", 80.5, -110),
                                            ("player_reception_yds", "rec_yds", "Drake London", "Under", 80.5, -110),
                                            ("player_receptions", "rec", "Drake London", "Over", 6.5, -120),
                                            ("player_anytime_td", "any_td", "Drake London", "Yes", None, 150),
                                            ("player_rush_yds", "rush_yds", "Bijan Robinson", "Over", 60.5, -110))]
    monkeypatch.setattr(odds, "props", lambda *a, **k: pd.DataFrame(rows))
    lines, info = market.slate_lines("k", pool)
    assert set(lines.dk_id) == {"1", "2"} and lines.set_index("dk_id").at["1", "line_rec_yds"] == 80.5
    res = pd.DataFrame({"dk_id": ["1", "2", "3"], "proj_model": [15.0, 16.0, 8.0], "p_rec_yds": [70.0, 20.0, 30.0],
                        "p_rec": [5.5, 3.0, 2.0], "p_rush_yds": [np.nan, 75.0, np.nan], "p_any_td": [0.35, 0.5, 0.2]})
    out = market.adjust(res, lines).set_index("dk_id")
    # London: +0.8*0.1*(80.5+4.7-70) + 0.5*1*(6.5-5.5) + TD term (book 40%*.92 vs model 35%)
    exp_td = 0.5 * 6 * (-np.log(1 - 0.4 * 0.92) + np.log(1 - 0.35))
    assert abs(out.at["1", "proj_lines_adj"] - round(0.8 * 0.1 * 15.2 + 0.5 + exp_td, 2)) < 0.02
    assert out.at["2", "proj_model"] < 16.0                                   # books lower on Bijan's rushing
    assert pd.isna(out.at["3", "proj_lines_adj"]) and out.at["3", "proj_model"] == 8.0


def test_lineup_alerts_flag_players_ruled_out_after_saving():
    from beatnfl import pipeline
    ls, pool = pool_from_fixture()
    sid = storage.save_slate(ls, 2026, 6)
    run_id = storage.save_projection_run(sid, pool, "test", {})
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=3, randomness=0))
    storage.save_lineup_set("alerts", sid, run_id, {}, r.lineups, pool, r.status, r.diagnostics)
    star = r.lineups[0][0]                                        # the QB in lineup 1
    later = pool.copy()
    later.loc[later.dk_id == star, "status"] = "O"                # inactive list posts
    later["dk_id"] = "x" + later.dk_id                            # and a different slate (new IDs) is loaded
    run = pipeline.WeekRun.__new__(pipeline.WeekRun)
    run.pool, run.season, run.week = later, 2026, 6
    al = pipeline.lineup_alerts(run)
    hit = al[(al.status == "O") & (al.set_name == "alerts")]
    assert len(hit) == 1 and hit.iloc[0].dk_id == star and hit.iloc[0].lineups >= 1


def test_current_week_follows_kickoffs_not_game_counts():
    from beatnfl.sources import nflverse as nv
    rows = [(6, "2026-10-15", "20:15")] + [(6, "2026-10-18", "13:00")] * 12 + [(6, "2026-10-19", "20:15")] \
        + [(7, "2026-10-22", "20:15")] + [(7, "2026-10-25", "13:00")] * 13
    s = pd.DataFrame({"season": 2026, "game_type": "REG", "week": [w for w, _, _ in rows],
                      "gameday": [d for _, d, _ in rows], "gametime": [t for _, _, t in rows]})
    at = lambda t: nv.upcoming_week(s, pd.Timestamp(t, tz="America/New_York"))
    assert at("2026-10-13 09:00") == (2026, 6)
    assert at("2026-10-17 11:00") == (2026, 6)     # after Thursday night: still this week (the old rule jumped to 7)
    assert at("2026-10-18 15:00") == (2026, 6)     # Sunday afternoon
    assert at("2026-10-19 23:00") == (2026, 6)     # Monday night game in progress
    assert at("2026-10-20 09:00") == (2026, 7)     # Tuesday morning → next week


def test_expected_starting_qb_skips_ruled_out_and_backups():
    from beatnfl.models import train
    tg = pd.DataFrame({"team": ["CIN", "CIN", "CIN", "KC"], "season": 2026, "week": [4, 5, 6, 6],
                       "qb_id": ["burrow", "burrow", None, None]})
    sp = pd.DataFrame({"gsis_id": ["burrow", "browning", "mahomes"], "pos": "QB", "team": ["CIN", "CIN", "KC"],
                       "injury": ["O", "", ""], "depth_order": [1, 2, 1]})
    out = train.expected_starters(tg, 2026, 6, sp, out_ids={"burrow"}).set_index("team")
    wk6 = out[out.week == 6].qb_id
    assert wk6["CIN"] == "browning" and wk6["KC"] == "mahomes"
    # no Sleeper data: last listed starter, unless he's ruled out
    out2 = train.expected_starters(tg, 2026, 6, None).set_index("team")
    assert out2[out2.week == 6].qb_id["CIN"] == "burrow"
    out3 = train.expected_starters(tg, 2026, 6, None, out_ids={"burrow"}).set_index("team")
    assert pd.isna(out3[out3.week == 6].qb_id["CIN"])


def test_grade_saved_lineups_against_results(monkeypatch):
    from beatnfl import grading
    from beatnfl.sources import nflverse as nv
    ls, pool = pool_from_fixture()
    sid = storage.save_slate(ls, 2026, 6)
    run_id = storage.save_projection_run(sid, pool, "t", {})
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=3, randomness=0))
    set_id = storage.save_lineup_set("grade-me", sid, run_id, {}, r.lineups, pool, r.status, r.diagnostics)
    idx = pool.set_index("dk_id")
    used = sorted({x for lu in r.lineups for x in lu})
    off = [x for x in used if idx.at[x, "pos"] != "DST"]
    scored = off[:-1]                                     # the last one has no stat line → didn't play → 0
    stats = pd.DataFrame({"player_display_name": [idx.at[x, "name"] for x in scored], "position": [idx.at[x, "pos"] for x in scored],
                          "team": [idx.at[x, "team"] for x in scored], "week": 6, "season": 2026, "dk_pts": 10.0})
    teams = sorted(set(pool.team))
    sched = pd.DataFrame({"season": 2026, "week": 6, "game_type": "REG", "home_team": teams[::2], "away_team": teams[1::2],
                          "home_score": 20.0, "away_score": 17.0})
    monkeypatch.setattr(nv, "weekly_stats", lambda *a, **k: stats)
    monkeypatch.setattr(nv, "schedules", lambda *a, **k: sched)
    monkeypatch.setattr(nv, "team_games", lambda s: pd.DataFrame())
    monkeypatch.setattr(nv, "dst_weekly", lambda st, tg: pd.DataFrame({"season": 2026, "week": 6, "team": teams, "dk_pts": 5.0}))
    g = grading.grade_set(set_id)
    lu1 = r.lineups[0]
    exp = sum(10.0 if x in scored else (5.0 if idx.at[x, "pos"] == "DST" else 0.0) for x in lu1)
    assert g["lineups"].iloc[0].actual == exp and g["summary"]["graded"] == 3
    assert g["players"].set_index("player").at[idx.at[off[-1], "name"], "actual"] == 0.0
    assert g["hindsight"] and g["summary"]["hindsight_best"] >= g["summary"]["best"]
    ht = g["hindsight"]["table"]
    assert len(ht) == 9 and {"projected", "pos_rank", "value_rank", "in_lineups"} <= set(ht.columns)


def test_compare_sources_rebuilds_from_each_source(monkeypatch):
    from beatnfl import grading
    ls, pool = pool_from_fixture()
    pool = pool.assign(proj_sleeper=pool.proj * 0.9, proj_model=pool.proj * 1.1)
    sid = storage.save_slate(ls, 2026, 6)
    run_id = storage.save_projection_run(sid, pool, "t", {})
    r = optimizer.build(pool, optimizer.OptSettings(n_lineups=3, randomness=0))
    old = storage.find_same_set(sid, r.lineups)          # identical set saved by an earlier test → start fresh
    if old:
        storage.delete_lineup_set(old)
    set_id = storage.save_lineup_set("cmp", sid, run_id, {"n_lineups": 3, "randomness": 0.0}, r.lineups, pool, r.status, r.diagnostics)
    monkeypatch.setattr(grading, "actual_points", lambda pool, s, w, offline=False: pd.Series(10.0, index=pool.dk_id.values))
    t = grading.compare_sources(set_id)
    assert {"Combined (what you used)", "Sleeper only", "BeatNFL model only"} <= set(t.source), t.to_string()
    assert (t.lineups == 3).all() and (t.avg_actual == 90.0).all()


# ---------------------------------------------------------------- ASK (Claude over the app's data; API faked)
class _FakeResp:
    def __init__(self, code, js):
        self.status_code, self._js, self.text = code, js, json.dumps(js)

    def json(self):
        return self._js


class _FakeSession:
    """Plays back API responses in order and records each request body."""
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def post(self, url, headers=None, json=None, timeout=None):
        import copy
        assert headers["x-api-key"] == "k" and headers["anthropic-version"]
        self.bodies.append(copy.deepcopy(json))
        return self.replies.pop(0)


def _ask_ctx():
    from beatnfl import ask, pipeline
    _, pool = pool_from_fixture()
    run = pipeline.WeekRun(pool=pool, season=2026, week=7)
    return ask, ask.Context(run=run, season=2026, week=7, forecasts=lambda: None, props=lambda: None,
                            games=lambda: (pd.DataFrame([{"Game": "CHI @ ATL", "Spread": "ATL -3"}]), pd.DataFrame()))


def test_ask_runs_tools_and_returns_answer_with_cost():
    ask, ctx = _ask_ctx()
    qb = ctx.run.pool[ctx.run.pool.pos == "QB"].sort_values("proj").name.iloc[-1]
    think = {"type": "thinking", "thinking": "checking the pool", "signature": "sig"}
    s = _FakeSession([
        _FakeResp(200, {"stop_reason": "tool_use", "usage": {"input_tokens": 1000, "output_tokens": 100}, "content": [
            think,
            {"type": "tool_use", "id": "t1", "name": "pool_players", "input": {"position": "QB", "limit": 3}},
            {"type": "tool_use", "id": "t2", "name": "build_lineups", "input": {"contest": "cash", "lock": [qb]}}]}),
        _FakeResp(200, {"stop_reason": "end_turn", "usage": {"input_tokens": 3000, "output_tokens": 200},
                        "content": [{"type": "text", "text": "Top QB is " + qb}]})])
    res = ask.ask("k", "best QB?", ctx, history=[("hi", "hello")], session=s)
    assert res["answer"] == "Top QB is " + qb
    assert res["tools_used"] == ["pool_players", "build_lineups"]
    assert res["cost"] == round(4000 / 1e6 * 2 + 300 / 1e6 * 10, 4)
    b = res["built"]["result"].lineups
    assert len(b) == 3 and all(ctx.run.pool.set_index("dk_id").loc[lu].name.eq(qb).any() for lu in b)
    first, second = s.bodies
    assert first["messages"][:2] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert first["tool_choice"] == {"type": "auto"} and "temperature" not in first
    assert second["messages"][-2]["content"][0] == think            # thinking block passed back unchanged
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    top = json.loads(results[0]["content"])
    assert top["players"][0]["name"] == qb and "combined" in top["players"][0]


def test_ask_errors_are_explained_and_thinking_fallback():
    ask, ctx = _ask_ctx()
    with pytest.raises(StepError) as e:
        ask.ask("", "q", ctx)
    assert "Settings" in e.value.fix
    s = _FakeSession([_FakeResp(401, {"error": {"message": "invalid x-api-key"}})])
    with pytest.raises(StepError) as e:
        ask.ask("k", "q", ctx, session=s)
    assert "refused the API key" in e.value.what
    s = _FakeSession([_FakeResp(400, {"error": {"message": "anthropic-workspace-id header is required for this key"}})])
    with pytest.raises(StepError) as e:
        ask.ask("k", "q", ctx, session=s)
    assert "Default" in e.value.fix
    s = _FakeSession([_FakeResp(400, {"error": {"message": "thinking type between_tools not supported"}}),
                      _FakeResp(200, {"stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ok"}]})])
    assert ask.ask("k", "q", ctx, session=s)["answer"] == "ok"
    assert "thinking" in s.bodies[0] and "thinking" not in s.bodies[1]
    # tools never crash the loop; unknown players are reported
    assert "Couldn't find" in ask.run_tool(ctx, "build_lineups", {"lock": ["Nobody Atall"]})
    assert "error" in json.loads(ask.run_tool(ctx, "player", {"name": "zzzz qqqq"}))
    assert json.loads(ask.run_tool(ctx, "games", {}))["lines"][0]["Spread"] == "ATL -3"
    assert "available" in json.loads(ask.run_tool(ctx, "evidence", {}))
    assert "unknown tool" in ask.run_tool(ctx, "nope", {})
    spec = {t["name"] for t in ask.tool_specs()}
    assert {"pool_players", "player", "build_lineups", "saved_set", "evidence"} <= spec



def test_bet_links_fallback_and_state():
    from beatnfl.sources import odds
    o, m, b = {"link": "https://x/addToBetslip?id=1"}, {"link": "https://x/market"}, {"link": "https://x/event"}
    assert odds._link(o, m, b).endswith("id=1") and odds._link({}, m, b).endswith("market") and odds._link({}, {}, b).endswith("event")
    assert odds._link({}, {"link": None}, {}) is None
    assert odds.bet_link("https://sports.{state}.betmgm.com/en/sports?options=1") == "https://sports.nj.betmgm.com/en/sports?options=1"
    assert odds.bet_link(None) is None and odds.bet_link(float("nan")) is None and odds.bet_link("javascript:alert(1)") is None


def test_picks_carry_the_best_books_slip():
    from beatnfl import research
    b = _board_rows()
    b["link"] = [f"https://{bk}/over" for bk in b.book]
    b["under_link"] = [f"https://{bk}/under" for bk in b.book]
    p = research.midweek_picks(b, None, min_diff=4, sides=None)
    v = research.picks_view(p).set_index("Player")
    assert v.at["Drake London", "Bet slip"] == "https://fanduel/over"          # best Over price was FanDuel
    assert v.at["Bijan Robinson", "Bet slip"] == "https://fanduel/under"
    assert research.picks_view(research.midweek_picks(_board_rows(), None, min_diff=4, sides=None))["Bet slip"].isna().all()



def test_best_odds_among_my_books():
    from beatnfl import research
    b = _board_rows()
    b["link"] = [f"https://{bk}/over" for bk in b.book]
    b["under_link"] = [f"https://{bk}/under" for bk in b.book]
    b = pd.concat([b, b[b.book == "fanduel"].assign(book="bovada", price=150, under_price=150,
                                                    link="https://bovada/over", under_link="https://bovada/under")])
    mine = research.DEFAULT_BOOKS
    v = research.props_view(b, books=mine).set_index("Player")
    dl, bj, kp = v.loc["Drake London"], v.loc["Bijan Robinson"], v.loc["Kyle Pitts"]
    assert (dl.Bet, dl["Bet odds"], dl["Best book"], dl["Bet slip"]) == ("Over 74.5", 105, "FanDuel", "https://fanduel/over")
    assert (bj.Bet, bj["Bet odds"], bj["Best book"], bj["Bet slip"]) == ("Under 80.5", -108, "FanDuel", "https://fanduel/under")
    assert kp.Bet == "" and pd.isna(kp["Bet slip"]) and pd.isna(kp["Bet odds"])          # no lean → no slip
    assert research.props_view(b).set_index("Player").at["Drake London", "Best book"] == "Bovada"   # no filter → offshore wins
    one = research.props_view(b, book="DraftKings").set_index("Player")                  # one book: its own price and slip
    assert one.at["Bijan Robinson", "Bet odds"] == -110 and one.at["Bijan Robinson", "Bet slip"] == "https://draftkings/under"
    p = research.midweek_picks(b, None, min_diff=4, sides=None, books=mine)
    assert set(p.book) == {"fanduel"}
    gp = pd.DataFrame({"Game": ["CHI @ ATL"] * 4, "Bet type": ["Spread", "Spread", "Total", "Moneyline"],
                       "Side": ["ATL -3", "CHI PK", "Over 47.25", "ATL win"]})
    raw = pd.DataFrame([
        dict(home="ATL", away="CHI", book="draftkings", market="spreads", side="ATL", line=-3.0, price=-110, link="https://dk/s"),
        dict(home="ATL", away="CHI", book="fanduel", market="spreads", side="ATL", line=-3.0, price=-105, link="https://fd/s"),
        dict(home="ATL", away="CHI", book="bovada", market="spreads", side="ATL", line=-3.0, price=110, link="https://bov/s"),
        dict(home="ATL", away="CHI", book="betmgm", market="spreads", side="ATL", line=-2.5, price=+120, link="https://mgm/s"),
        dict(home="ATL", away="CHI", book="draftkings", market="spreads", side="CHI", line=0.0, price=-102, link=None),
        dict(home="ATL", away="CHI", book="draftkings", market="totals", side="Over", line=47.5, price=-108, link="https://dk/o"),
        dict(home="ATL", away="CHI", book="fanduel", market="totals", side="Over", line=47.5, price=-112, link="https://fd/o"),
        dict(home="ATL", away="CHI", book="fanduel", market="totals", side="Over", line=47.0, price=-120, link="https://fd/o2"),
        dict(home="ATL", away="CHI", book="fanduel", market="h2h", side="ATL", line=None, price=-150, link="https://fd/ml"),
        dict(home="ATL", away="CHI", book="betrivers", market="h2h", side="ATL", line=None, price=-140, link="https://br/ml")])
    best = research.game_best(gp, raw, mine)
    assert best["Best odds"].tolist() == [-105, -102, -108, -140]
    assert best["Best book"].tolist() == ["FanDuel", "DraftKings", "DraftKings (at 47.5)", "BetRivers"]
    assert best["Best slip"].tolist() == ["https://fd/s", None, "https://dk/o", "https://br/ml"]



def test_best_price_tie_prefers_a_book_with_a_slip():
    from beatnfl import research
    r = pd.DataFrame({"book": ["williamhill_us", "draftkings"], "price": [-110, -110], "link": [None, "https://dk/x"]})
    assert research._best(r, "price", "link") == (-110.0, "draftkings = williamhill_us", "https://dk/x")



def test_parlay_pricing_and_slips():
    from beatnfl import parlay, research
    b = _board_rows()
    b["link"] = [f"https://{bk}/over/{p}" for bk, p in zip(b.book, b.player)]
    b["under_link"] = [f"https://{bk}/under/{p}" for bk, p in zip(b.book, b.player)]
    dl = parlay.prop_leg(b, "CHI @ ATL|Drake London|rec_yds", "Over")
    bj = parlay.prop_leg(b, "CHI @ ATL|Bijan Robinson|rush_yds", "Under")
    gl = parlay.game_leg({"Game": "KC @ DEN", "Bet type": "Spread", "Side": "DEN +3.5", "Model %": 55.0, "Market %": 50.0},
                         pd.DataFrame([dict(home="DEN", away="KC", book="draftkings", market="spreads", side="DEN", line=3.5,
                                            price=-110, link="https://draftkings/den"),
                                       dict(home="DEN", away="KC", book="fanduel", market="spreads", side="DEN", line=3.5,
                                            price=-105, link="https://fanduel/den")]))
    legs = [dl, bj, gl]
    t = parlay.legs_table(legs)
    assert list(t.Side) == ["Over", "Under", "Bet"] and t["Model %"].tolist() == [63.0, 62.0, 55.0]
    pr = parlay.price(legs, research.DEFAULT_BOOKS).set_index("book")
    # FanDuel: +105 (2.05) × -108 (1.9259) × -105 (1.9524) = 7.708
    dec = 2.05 * (1 + 100 / 108) * (1 + 100 / 105)
    assert pr.at["fanduel", "Pays on $10"] == round(10 * dec, 2) and pr.index[0] == "fanduel"
    assert pr.at["betmgm", "Legs offered"] == "1 of 3" and pd.isna(pr.at["betmgm", "Parlay odds"])
    assert abs(pr.at["fanduel", "Model chance %"] - round(100 * .63 * .62 * .55, 1)) < 1e-9
    sl = parlay.slips(legs, "fanduel")
    assert sl.Slip.tolist()[:3] == ["https://fanduel/over/Drake London", "https://fanduel/under/Bijan Robinson", "https://fanduel/den"]
    assert sl.Leg.iloc[3].startswith("PARLAY — all 3 legs") and sl.Odds.iloc[3] == parlay.american(dec)    # the final odds
    top = parlay.best(pr.reset_index())
    assert top["book"] == "fanduel" and top["pays10"] == round(10 * dec, 2)
    bj["side"] = "Over"                                                          # switching side uses the other prices
    assert parlay.slips([bj], "draftkings").Odds.tolist()[0] == -110 and parlay.leg_probs(bj)[0] == 0.38
    assert parlay.same_game(legs) == ["CHI @ ATL"]
    assert parlay.american(2.5) == 150 and parlay.american(1.5) == -200



def test_ties_listed_and_in_your_book_order():
    from beatnfl import parlay, research
    r = pd.DataFrame({"book": ["betmgm", "fanduel", "draftkings"], "price": [-110, -110, -115],
                      "link": ["https://mgm", "https://fd", "https://dk"]})
    p, bk, lk = research._best(r, "price", "link", ["fanduel", "betmgm", "draftkings"])
    assert (p, bk, lk) == (-110.0, "fanduel = betmgm", "https://fd") and research.book_name(bk) == "FanDuel = BetMGM"
    r2 = r.assign(link=[None, None, "https://dk"]).iloc[:2]
    assert research._best(r2, "price", "link", ["fanduel", "betmgm"])[1] == "fanduel = betmgm"   # no slip anywhere → your order
    q = lambda bk, pr: {"book": bk, "line": 1.5, "over": pr, "under": np.nan, "over_link": f"https://{bk}", "under_link": None}
    legs = [{"kind": "game", "id": "a", "game": "A @ B", "label": "x", "side": "Bet", "model_p_over": .5, "market_p_over": .5,
             "quotes": [q("draftkings", -110), q("fanduel", -110), q("betmgm", -120)]},
            {"kind": "game", "id": "b", "game": "C @ D", "label": "y", "side": "Bet", "model_p_over": .5, "market_p_over": .5,
             "quotes": [q("draftkings", 100), q("fanduel", 100)]}]
    pr = parlay.price(legs, ["fanduel", "draftkings", "betmgm"])
    assert pr.book.tolist() == ["fanduel", "draftkings", "betmgm"]
    assert pr.Best.tolist() == ["★ tied best", "★ tied best", ""] and pr["Not offered"].iloc[2] == "y"


def test_draftkings_block_keeps_saved_slate_and_refreshes_the_rest(monkeypatch):
    from beatnfl import pipeline
    from beatnfl.sources import dk as _dk
    future = SAL.read_text(encoding="utf-8-sig").replace("10/18/2026", "12/27/2099")
    ls = _dk.load_from_file(future.encode())
    ls.content_hash = "FIXTURE-future-week"          # same players/salaries as the week-6 fixture, different week
    storage.save_slate(ls, 2099, 17)
    assert storage.recent_open_slate() is not None
    monkeypatch.setattr(pipeline, "newest_salary_file", lambda: None)

    def blocked(*a, **k):
        raise StepError("Load DraftKings slate", "DraftKings is refusing automatic downloads from this connection right now.",
                        "bot protection", "Wait 15–30 minutes and click ⟳ Refresh.", detail="HTTP 403: Access Denied")
    monkeypatch.setattr(_dk, "load_auto", blocked)
    r = pipeline.run_week(Settings(), use_model=False, offline=True)
    s1 = r.steps[0]
    assert s1.step == "1. Slate (automatic)" and s1.status == "warn" and "Kept the slate" in s1.message, (s1.step, s1.message)
    assert "Wait 15" in s1.fix
    assert r.pool is not None and len(r.pool) == len(ls.players)                       # the rest of the refresh ran
    assert any(x.step.startswith("2.") for x in r.steps) and any(x.step.startswith("5.") for x in r.steps)


def test_csv_endpoint_block_page_is_recognised(monkeypatch):
    from beatnfl.sources import dk as _dk
    monkeypatch.setattr(_dk, "discover", lambda offline=False: [_dk.SlateInfo("153728", "Sun 1:00 PM · 12 games · DG 153728", "", 12, 21)])

    def fj(url, **k):
        raise StepError("x", "Could not download", detail="HTTP 403: <HTML><HEAD><TITLE>Access Denied</TITLE>")
    monkeypatch.setattr(_dk, "fetch_json", fj)
    monkeypatch.setattr(_dk, "fetch", lambda url, **k: (b"<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD>", None))
    with pytest.raises(StepError) as e:
        _dk.load_auto()
    assert "refusing automatic downloads" in e.value.what and "DG 153728" in e.value.what
    assert "Export to CSV" in e.value.fix


def test_started_games_are_hidden():
    from beatnfl import parlay, research
    b = _board_rows()
    b.loc[b.player == "Bijan Robinson", "commence"] = "2020-01-01T18:00:00Z"            # this game already kicked off
    b.loc[b.player == "Bijan Robinson", "game"] = "KC @ DEN"
    v = research.props_view(b)
    assert "Bijan Robinson" not in set(v.Player) and "Drake London" in set(v.Player)
    assert "Bijan Robinson" in set(research.props_view(b, include_started=True).Player)
    legs = [parlay.prop_leg(b, "KC @ DEN|Bijan Robinson|rush_yds", "Under"), parlay.prop_leg(b, "CHI @ ATL|Drake London|rec_yds")]
    assert parlay.started_legs(legs) == ["Bijan Robinson Under 80.5 rush yds"]
    assert research.started(["2020-01-01T00:00:00Z", "2099-01-01T00:00:00Z", None]).tolist() == [True, False, False]


def test_rows_ticked_on_props_become_parlay_legs():
    # the exact path the "Add to parlay" button takes: selected view rows → their hidden key/side → legs
    from beatnfl import parlay, research
    b = _board_rows()
    view = research.props_view(b)
    legs = [parlay.prop_leg(b, r["_key"], r["_side"] or "Over") for r in view.to_dict("records")]
    assert all(legs) and {lg["player"] for lg in legs} == {"Drake London", "Bijan Robinson", "Kyle Pitts"}
    assert {lg["player"]: lg["side"] for lg in legs}["Bijan Robinson"] == "Under"
    import ast
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    assert "r_._key" not in src and "r_._side" not in src                   # itertuples would rename these columns



def test_parlay_uses_each_books_nearest_line_and_says_so():
    from beatnfl import parlay
    b = _board_rows()
    b = pd.concat([b, b[(b.player == "Drake London") & (b.book == "draftkings")].assign(book="betrivers", line=75.5, price=-105)])
    lg = parlay.prop_leg(b, "CHI @ ATL|Drake London|rec_yds", "Over")
    pr = parlay.price([lg], ["draftkings", "betrivers"]).set_index("book")
    assert pr.at["betrivers", "Different line"] == "Drake London at 75.5" and pr.at["draftkings", "Different line"] == ""
    assert "this book: 75.5" in parlay.slips([lg], "betrivers").Leg.iloc[0]
    assert parlay.best(parlay.price([lg], ["betmgm_only_not_listed"])) is None


def test_yahoo_redirect_uri_and_pasted_address(monkeypatch, tmp_path):
    from urllib.parse import parse_qs, urlparse
    from beatnfl.sources import yahoo
    u = yahoo.auth_url(" cid ", "https://localhost:8000")
    q = parse_qs(urlparse(u).query)
    assert q["redirect_uri"] == ["https://localhost:8000"] and q["client_id"] == ["cid"] and "scope" not in q
    assert parse_qs(urlparse(yahoo.auth_url("cid", "oob")).query)["redirect_uri"] == ["oob"]
    assert yahoo.extract_code("https://localhost:8000/?code=abc123&state=x") == "abc123"
    assert yahoo.extract_code("  abc123 ") == "abc123"
    sent = []

    class R:
        status_code = 200
        def json(self): return {"access_token": "A", "refresh_token": "R", "expires_in": 3600}
    monkeypatch.setattr(yahoo, "TOKEN_PATH", tmp_path / "t.json")
    monkeypatch.setattr(yahoo.requests, "post", lambda url, data=None, auth=None, timeout=None: sent.append(data) or R())
    yahoo.connect("cid", "sec", "https://localhost:8000/?code=xyz", "https://localhost:8000")
    assert sent[-1]["code"] == "xyz" and sent[-1]["redirect_uri"] == "https://localhost:8000"
    assert json.loads((tmp_path / "t.json").read_text())["redirect_uri"] == "https://localhost:8000"
    with pytest.raises(StepError) as e:
        yahoo.connect("cid", "sec", "https://localhost:8000/?error=access_denied", "https://localhost:8000")
    assert "No sign-in code" in e.value.what
    with pytest.raises(StepError):
        yahoo.connect("cid", "sec", "https://localhost:8000/?code=%E2%80%A6", "https://localhost:8000")   # the "…" placeholder


def test_yahoo_signin_catcher_finishes_by_itself(monkeypatch, tmp_path):
    import socket, time as _t, requests as _rq, urllib3
    from beatnfl.sources import yahoo
    urllib3.disable_warnings()
    s_ = socket.socket(); s_.bind(("127.0.0.1", 0)); port = s_.getsockname()[1]; s_.close()
    got = []
    monkeypatch.setattr(yahoo, "connect", lambda cid, sec, code, ru: got.append((code, ru)))
    ru = f"https://localhost:{port}"
    assert yahoo.start_catcher("cid", "sec", ru) == "waiting"
    r = _rq.get(f"https://127.0.0.1:{port}/?code=abc123", verify=False, timeout=10)
    assert r.status_code == 200 and "connected" in r.text
    assert got == [("abc123", ru)] and yahoo.CATCH["status"] == "connected"
    yahoo.stop_catcher()
    assert yahoo.start_catcher("cid", "sec", "oob") == ""                       # nothing to catch for oob
    assert yahoo.start_catcher("cid", "sec", ru) == "waiting"
    r = _rq.get(f"https://127.0.0.1:{port}/", verify=False, timeout=10)
    assert "without a sign-in code" in r.text and yahoo.CATCH["status"].startswith("error")
    yahoo.stop_catcher()
