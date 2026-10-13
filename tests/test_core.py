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
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    import sys
    assert hasattr(sys.modules["beatnfl.storage"], "duplicate_sets")



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
