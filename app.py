"""BeatNFL — Streamlit app. Start with:  python -m beatnfl app   (or the Start BeatNFL shortcut)."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beatnfl import __version__, export, optimizer, pipeline, storage  # noqa: E402
from beatnfl.config import EXPORTS, HOME, INBOX, LOG_PATH, Settings, ensure_dirs  # noqa: E402
from beatnfl.errors import StepError  # noqa: E402

st.set_page_config(page_title="BeatNFL", page_icon="🏈", layout="wide")
ensure_dirs()
S = Settings.load()
ss = st.session_state


def show_steps(steps):
    for s in steps:
        box = {"ok": st.success, "warn": st.warning, "fail": st.error, "skip": st.info}[s.status]
        box(f"**{s.step}** — {s.message}" + (f"\n\n➜ {s.fix}" if s.fix else ""))


def slate_banner():
    r = ss.get("run")
    if r and r.slate:
        sl = r.slate
        verified = "IDs verified" if sl.ids_verified else "IDs not yet verified vs official CSV"
        st.caption(f"**Slate:** {sl.slate.label if sl.slate else ''} · {r.season or '?'} week {r.week or '?'} · "
                   f"source: {sl.route} ({sl.fetched_note}) · {verified}")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏈 BeatNFL")
    st.caption(f"v{__version__} · research & predictions only — no bets, no contest entry")
    with st.expander("Settings"):
        key = st.text_input("The Odds API key (optional)", S.odds_api_key, type="password",
                            help="Free key at the-odds-api.com — powers live lines and props.")
        yid = st.text_input("Yahoo Client ID (optional)", S.yahoo_client_id, type="password")
        ysec = st.text_input("Yahoo Client Secret (optional)", S.yahoo_client_secret, type="password")
        c1, c2, c3 = st.columns(3)
        wp = c1.number_input("Paid wt", 0.0, 1.0, S.w_paid, 0.05)
        ws = c2.number_input("Sleeper wt", 0.0, 1.0, S.w_sleeper, 0.05)
        wm = c3.number_input("Model wt", 0.0, 1.0, S.w_model, 0.05)
        qh = st.slider("Questionable haircut", 0.5, 1.0, S.questionable_haircut, 0.05)
        if st.button("Save settings"):
            S.odds_api_key, S.w_paid, S.w_sleeper, S.w_model, S.questionable_haircut = key, wp, ws, wm, qh
            S.yahoo_client_id, S.yahoo_client_secret = yid.strip(), ysec.strip()
            S.save()
            st.success("Saved.")
    with st.expander("App updates"):
        from beatnfl import update as upd
        repo = st.text_input("GitHub repository (owner/name)", S.update_repo, key="upd_repo")
        tok = st.text_input("Read-only token (private repo only)", S.github_token, type="password", key="upd_tok")
        if (repo, tok) != (S.update_repo, S.github_token) and st.button("Save update source"):
            S.update_repo, S.github_token = repo.strip(), tok.strip()
            S.save()
            st.success("Saved.")
        if S.update_repo:
            if st.button("Check for updates"):
                try:
                    ss["upd"] = upd.check(S.update_repo, S.github_token)
                except StepError as e:
                    st.error(e.as_text())
                except Exception as e:  # noqa: BLE001
                    st.error(f"Couldn't reach GitHub: {e}")
            u = ss.get("upd")
            if u and u.get("update_available"):
                st.info(f"New version: {u['message']} ({u['date'][:10]})")
                if st.button("Install update", type="primary"):
                    with st.spinner("Updating…"):
                        try:
                            res = upd.apply(S.update_repo, S.github_token)
                            st.success(res["note"])
                            ss.pop("upd", None)
                            ss["needs_restart"] = True
                        except StepError as e:
                            st.error(e.as_text())
            elif u:
                st.success("You're on the latest version.")
        if ss.get("needs_restart") and st.button("Restart BeatNFL"):
            if upd.restart():
                st.info("Restarting… the page reopens in a few seconds.")
            else:
                st.info("Close the BeatNFL window and start it again to load the update.")
    offline = st.toggle("Offline mode (use cached data only)", False)
    st.caption(f"Data folder: `{HOME}`\n\nInbox: `{INBOX}`")

tab_week, tab_saved, tab_yahoo, tab_research, tab_evidence, tab_status = st.tabs(
    ["This week", "Saved lineups", "My Yahoo team", "Research: games & props", "Evidence", "Data status"])

# ------------------------------------------------------------------ THIS WEEK
with tab_week:
    slate_banner()
    st.subheader("1 · Load the week")
    mode = st.radio("Slate source", ["Automatic (DraftKings)", "Upload DKSalaries.csv"], horizontal=True)
    dg, up = None, None
    if mode.startswith("Automatic"):
        c1, c2 = st.columns([3, 1])
        if c2.button("Find slates"):
            try:
                from beatnfl.sources import dk
                ss["slates"] = dk.discover(offline=offline)
            except StepError as e:
                st.error(e.as_text())
        if ss.get("slates"):
            from beatnfl.sources import dk
            main = dk.pick_main_slate(ss["slates"])
            labels = [s.label for s in ss["slates"]]
            pick = c1.selectbox("Slate", labels, index=labels.index(main.label))
            dg = ss["slates"][labels.index(pick)].draft_group_id
        else:
            c1.caption("Main slate is chosen automatically. Click *Find slates* to pick a different one.")
        st.caption(f"If DraftKings blocks the download, the newest DKSalaries*.csv in the inbox is used automatically.")
    else:
        up = st.file_uploader("DKSalaries.csv (DraftKings → NFL Classic contest → Export to CSV)", type="csv")
    paid = st.file_uploader("Paid projections CSV (optional — newest file in the inbox is used otherwise)", type="csv")
    pasted = st.text_area("…or paste a projections table (e.g. select Sharp's DFS table in your browser, copy, paste here)",
                          height=90, key="pasted_proj")
    use_model = st.checkbox("Include in-house model", True)

    if st.button("▶ Run", type="primary", disabled=(mode.startswith("Upload") and up is None)):
        status = st.status("Running…", expanded=True)
        r = pipeline.run_week(S, salary_file=up.getvalue() if up else None, draft_group_id=dg,
                              paid_file=paid.getvalue() if paid else (pasted.encode() if pasted.strip() else None), use_model=use_model, offline=offline,
                              progress=lambda m: status.write(m))
        status.update(label="Done", state="complete", expanded=False)
        ss["run"], ss["lineups"] = r, None
        st.rerun()

    r = ss.get("run")
    if r:
        show_steps(r.steps)
    if r and r.pool is not None and "proj" in r.pool:
        st.subheader("2 · Player pool")
        pool = r.pool
        cols = ["dk_id", "name", "pos", "team", "opp", "salary", "status", "injury_note", "proj", "value", "proj_paid",
                "proj_sleeper", "proj_model", "proj_spread", "pct_rostered_yahoo", "sources"]
        cols = [c for c in cols if c in pool]
        view = pool[cols].copy()
        view.insert(1, "Lock", view.dk_id.isin(ss.get("locks", [])))
        view.insert(2, "Exclude", view.dk_id.isin(ss.get("excludes", [])))
        f1, f2 = st.columns([1, 3])
        posf = f1.multiselect("Positions", ["QB", "RB", "WR", "TE", "DST"], [])
        if posf:
            view = view[view.pos.isin(posf)]
        f2.caption("Missing projections show blank (never 0) and are left out of lineups. "
                   "`proj_spread` = disagreement between sources.")
        ed = st.data_editor(view.sort_values("proj", ascending=False), hide_index=True, width="stretch",
                            disabled=[c for c in view.columns if c not in ("Lock", "Exclude")], height=420)
        ss["locks"] = ed[ed.Lock].dk_id.tolist() + [l for l in ss.get("locks", []) if l not in set(view.dk_id)]
        ss["excludes"] = ed[ed.Exclude].dk_id.tolist() + [l for l in ss.get("excludes", []) if l not in set(view.dk_id)]

        st.subheader("3 · Build lineups")
        c = st.columns(7)
        o = optimizer.OptSettings(
            n_lineups=c[0].number_input("Lineups", 1, 150, S.n_lineups),
            min_salary=c[1].number_input("Min salary", 40000, 50000, S.min_salary, 100),
            max_exposure=c[2].slider("Max exposure", 0.1, 1.0, S.max_exposure, 0.05),
            min_unique=c[3].number_input("Min changes", 1, 6, S.min_unique),
            stack_min=c[4].number_input("QB stack", 0, 3, S.stack_min),
            bring_back=c[5].number_input("Bring-back", 0, 2, S.bring_back),
            randomness=c[6].slider("Randomness", 0.0, 0.4, S.randomness, 0.02),
            locks=ss.get("locks", []), excludes=ss.get("excludes", []))
        name = st.text_input("Name for this set", f"{r.season}_wk{r.week}" if r.week else "lineups")
        if st.button("⚙ Build, save & export", type="primary"):
            with st.spinner("Optimizing…"):
                try:
                    ss["lineups"] = pipeline.build_lineups(r, o, name=name)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Lineup build failed: {e}")
        out = ss.get("lineups")
        if out:
            res = out["result"]
            msg = f"Requested **{res.requested}** · generated **{len(res.lineups)}** · {res.status} · {res.elapsed}s"
            (st.success if res.status == "complete" else st.warning if res.lineups else st.error)(msg)
            for d in res.diagnostics:
                st.write("•", d)
            if res.lineups:
                if out["set_id"]:
                    st.info(f"Saved as set #{out['set_id']} — reopen any time under *Saved lineups*. "
                            f"Files written to `{EXPORTS}`.")
                st.dataframe(export.lineup_table(res.lineups, r.pool), hide_index=True, width="stretch")
                d1, d2 = st.columns(2)
                for f, col, lab in zip(out["files"], (d1, d2), ("⬇ DraftKings upload CSV", "⬇ Readable lineups CSV")):
                    col.download_button(lab, Path(f).read_bytes(), file_name=Path(f).name, mime="text/csv")
                with st.expander("Exposure"):
                    st.dataframe(export.exposure(res.lineups, r.pool), hide_index=True)

# ------------------------------------------------------------------ SAVED
with tab_saved:
    sets = storage.list_lineup_sets()
    if sets.empty:
        st.info("No saved lineup sets yet.")
    else:
        st.dataframe(sets, hide_index=True, width="stretch")
        sid = st.selectbox("Open set", sets.id.tolist(), format_func=lambda i: f"#{i} · {sets.set_index('id').at[i, 'name']}")
        d = storage.load_lineup_set(int(sid))
        _, pool = storage.get_projection_run(d["projection_run_id"])
        bad = [i + 1 for i, lu in enumerate(d["lineups"]) if optimizer.validate(lu, pool)]
        st.caption(f"{len(d['lineups'])} lineups · created {d['created_utc']} UTC · revision {d['revision']} · "
                   + ("all valid against the saved slate" if not bad else f"invalid: {bad}"))
        st.dataframe(export.lineup_table(d["lineups"], pool), hide_index=True, width="stretch")
        c1, c2, c3 = st.columns(3)
        try:
            c1.download_button("⬇ DraftKings upload CSV", export.dk_csv_text(d["lineups"], pool),
                               file_name=f"DK_upload_{d['name']}.csv", mime="text/csv")
        except ValueError as e:
            c1.error(str(e))
        new = c2.text_input("Rename", d["name"])
        if c2.button("Rename") and new != d["name"]:
            try:
                storage.rename_lineup_set(int(sid), new, d["revision"])
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
        if c3.button("Delete set"):
            ss["confirm_del"] = int(sid)
        if ss.get("confirm_del") == int(sid):
            if c3.button("Confirm delete — can't be undone"):
                storage.delete_lineup_set(int(sid))
                ss.pop("confirm_del")
                st.rerun()

# ------------------------------------------------------------------ MY YAHOO TEAM
with tab_yahoo:
    from beatnfl.sources import yahoo
    st.caption("Read-only: shows your Yahoo roster with BeatNFL projections and start/sit suggestions. "
               "It never changes your lineup on Yahoo.")
    if not (S.yahoo_client_id and S.yahoo_client_secret):
        st.markdown(
            "**One-time setup**\n\n"
            "1. Go to [developer.yahoo.com/apps/create](https://developer.yahoo.com/apps/create/) and sign in.\n"
            "2. Application type **Installed Application**, Redirect URI **oob**, API permissions **Fantasy Sports → Read**. Create the app.\n"
            "3. Copy the **Client ID** and **Client Secret** into *Settings* in the sidebar and click **Save settings**.")
    elif not yahoo.connected():
        st.markdown("**Connect your Yahoo account**")
        st.link_button("1 · Sign in to Yahoo and approve", yahoo.auth_url(S.yahoo_client_id))
        code = st.text_input("2 · Paste the code Yahoo shows you", key="yahoo_code")
        if st.button("Connect", disabled=not code.strip()):
            try:
                yahoo.connect(S.yahoo_client_id, S.yahoo_client_secret, code)
                st.success("Connected to Yahoo.")
                st.rerun()
            except StepError as e:
                st.error(e.as_text())
    else:
        try:
            if "yteams" not in ss:
                ss["yteams"] = yahoo.my_teams(S.yahoo_client_id, S.yahoo_client_secret)
            teams = ss["yteams"]
            if not teams:
                st.warning("No NFL fantasy teams found on this Yahoo account for the current season.")
            else:
                from beatnfl.sources import nflverse as nv
                sched = nv.schedules(offline=offline)
                up = sched[sched.home_score.isna() & (sched.game_type == "REG")]
                season_y = int(up.season.iloc[0]) if len(up) else nv.current_season()
                c1, c2 = st.columns([3, 1])
                labels = [t["name"] for t in teams]
                pick = c1.selectbox("Team", labels)
                yweek = c2.number_input("Week", 1, 18, int(up.week.iloc[0]) if len(up) else 1, key="yweek")
                team = teams[labels.index(pick)]
                if st.button("Load roster & projections", type="primary"):
                    with st.spinner("Reading your roster and running the model…"):
                        from beatnfl.models import train
                        from beatnfl.names import norm_name
                        from beatnfl.sources import sleeper
                        rows = yahoo.roster(team["team_key"], int(yweek), S.yahoo_client_id, S.yahoo_client_secret)
                        slots = yahoo.roster_slots(team["league_key"], S.yahoo_client_id, S.yahoo_client_secret)
                        pm, dm, meta = train.get_models(offline=offline)
                        try:
                            spp, _ = sleeper.players(offline=offline)
                        except StepError:
                            spp = None
                        fc, warns = train.forecast_week(season_y, int(yweek), spp, pm, offline=offline)
                        wk = sched[(sched.season == season_y) & (sched.week == int(yweek))]
                        playing = set(wk.home_team) | set(wk.away_team)
                        fc = fc.assign(nn=fc.name.map(norm_name))
                        by_nt = {(r.nn, r.team): r for r in fc.itertuples()}
                        uniq = fc.drop_duplicates("nn", keep=False).set_index("nn")
                        for r in rows:
                            nn = norm_name(r["name"])
                            f = by_nt.get((nn, r["team"]))
                            if f is None and nn in uniq.index:
                                f = uniq.loc[nn]
                            r["proj"] = round(float(f.p_dk_pts), 1) if f is not None and pd.notna(f.p_dk_pts) else None
                            r["opp"] = f.opp if f is not None else ""
                            r["on_bye"] = r["team"] not in playing and r["pos"] not in ("K", "DEF")
                            for k in ("p_pass_yds", "p_rush_yds", "p_rec", "p_rec_yds", "p_any_td"):
                                r[k] = (round(float(getattr(f, k)), 2) if f is not None and pd.notna(getattr(f, k)) else None)
                        ss["yroster"], ss["yslots"] = rows, slots
                        for w in warns:
                            st.warning(w)
                rows, slots = ss.get("yroster"), ss.get("yslots")
                if rows:
                    df = pd.DataFrame(rows)
                    show = df[["slot", "name", "pos", "team", "opp", "status", "injury_note", "proj", "p_pass_yds",
                               "p_rush_yds", "p_rec", "p_rec_yds", "p_any_td"]].rename(columns={
                        "slot": "Yahoo slot", "proj": "Proj (PPR+bonus)", "p_pass_yds": "Pass yds", "p_rush_yds": "Rush yds",
                        "p_rec": "Rec", "p_rec_yds": "Rec yds", "p_any_td": "TD chance"})
                    st.dataframe(show, hide_index=True, width="stretch")
                    model_slots = [(s_, n) for s_, n in slots if s_ not in ("K", "DEF")]
                    sug = yahoo.suggest(rows, model_slots)
                    cur = {r["player_key"] for r in rows if r["slot"] not in ("BN", "IR", "IR+", "NA", "K", "DEF")}
                    best = {s_["player_key"] for s_ in sug if s_["player_key"]}
                    st.markdown("**Suggested starters** (kickers and defenses not projected — keep your choice)")
                    st.dataframe(pd.DataFrame(sug).drop(columns=["player_key"]), hide_index=True)
                    ins = [r for r in rows if r["player_key"] in best - cur]
                    outs = [r for r in rows if r["player_key"] in cur - best]
                    if ins or outs:
                        parts = []
                        if ins:
                            parts.append("start " + ", ".join(f"{r['name']} ({r['proj']})" for r in ins))
                        if outs:
                            parts.append("bench " + ", ".join(
                                f"{r['name']} ({r['proj'] if r['proj'] is not None else 'no proj'})" for r in outs))
                        st.warning("Changes vs your current Yahoo lineup: " + " · ".join(parts) +
                                   ". Make any changes on Yahoo yourself.")
                    else:
                        st.success("Your current Yahoo starters match the model's picks.")
                    missing = [r["name"] for r in rows if r["proj"] is None and r["pos"] not in ("K", "DEF")]
                    if missing:
                        st.caption("No projection (bye, injured/out, or too little history): " + ", ".join(missing))
            if st.button("Disconnect Yahoo"):
                yahoo.disconnect()
                ss.pop("yteams", None)
                st.rerun()
        except StepError as e:
            st.error(e.as_text())

# ------------------------------------------------------------------ RESEARCH
with tab_research:
    st.caption("Information only. The game model has **not** beaten closing lines historically — see Evidence. "
               "Treat model numbers as a second opinion, not a pick.")
    from beatnfl.sources import nflverse as nv
    try:
        sched = nv.schedules(offline=offline)
        upcoming = sched[sched.home_score.isna() & (sched.game_type == "REG")]
        default = (int(upcoming.season.iloc[0]), int(upcoming.week.iloc[0])) if len(upcoming) else (nv.current_season(), 1)
    except StepError as e:
        st.error(e.as_text())
        default = (nv.current_season(), 1)
    c1, c2 = st.columns(2)
    season = c1.number_input("Season", 2015, 2035, default[0])
    week = c2.number_input("Week", 1, 18, default[1])
    lines = None
    if S.odds_api_key:
        try:
            from beatnfl.sources import odds
            lines = odds.consensus(odds.game_lines(S.odds_api_key, offline=offline))
        except StepError as e:
            st.warning(e.as_text())
    st.subheader("Games")
    try:
        from beatnfl import research
        st.dataframe(research.games_table(int(season), int(week), lines, offline=offline), hide_index=True,
                     width="stretch")
    except StepError as e:
        st.error(e.as_text())

    st.subheader("Matchups")
    st.caption("Team efficiency from free NFL play-by-play: offense vs the opponent's defense, recent games weighted "
               "more. Pass/Run edge = offense EPA per play + opponent's EPA allowed (higher favors the offense). "
               "Tested as model inputs, these didn't improve forecasts beyond what betting lines already carry (see "
               "Evidence), so they're shown for reading, not fed into projections.")
    if st.button("Load matchups"):
        with st.spinner("Reading play-by-play (first time downloads ~20 MB per season)…"):
            try:
                from beatnfl import research as _r
                ss["mu"] = _r.matchups(int(season), int(week), offline=offline)
            except StepError as e:
                st.error(e.as_text())
    if ss.get("mu") is not None:
        st.dataframe(ss["mu"].sort_values("Pass edge", ascending=False), hide_index=True, width="stretch")

    st.subheader("Player forecasts & props")
    if st.button("Forecast every player this week"):
        with st.spinner("Forecasting…"):
            try:
                from beatnfl.models import train
                from beatnfl.sources import sleeper
                pm, dm, meta = train.get_models(offline=offline)
                try:
                    spp, _ = sleeper.players(offline=offline)
                except StepError:
                    spp = None
                fc, warns = train.forecast_week(int(season), int(week), spp, pm, lines, offline=offline)
                from beatnfl.sources import yahoo as _y
                if S.yahoo_client_id and S.yahoo_client_secret and _y.connected() and not offline:
                    try:
                        lk = _y.first_league(S.yahoo_client_id, S.yahoo_client_secret)
                        yp = pd.DataFrame(_y.league_players(lk, S.yahoo_client_id, S.yahoo_client_secret))
                        from beatnfl.names import norm_name
                        yp["nn"] = yp.name.map(norm_name)
                        fc["nn"] = fc.name.map(norm_name)
                        fc = fc.merge(yp[["nn", "team", "pct_rostered", "pct_change", "status_raw"]].drop_duplicates(["nn", "team"]),
                                      on=["nn", "team"], how="left").rename(columns={
                            "pct_rostered": "yahoo_pct_rostered", "pct_change": "yahoo_pct_change", "status_raw": "yahoo_status"})
                    except StepError as e:
                        st.warning("Yahoo data skipped: " + e.what)
                ss["fc"], ss["pm"] = fc, pm
                for w in warns:
                    st.warning(w)
            except StepError as e:
                st.error(e.as_text())
    fc = ss.get("fc")
    if fc is not None:
        cols_fc = ["name", "position", "team", "opp", "injury", "p_dk_pts", "p_pass_yds", "p_pass_att", "p_rush_yds",
                   "p_rush_att", "p_rec", "p_rec_yds", "p_any_td", "yahoo_status", "yahoo_pct_rostered", "yahoo_pct_change"]
        show = fc[[c for c in cols_fc if c in fc]].rename(columns=lambda c: c.replace("p_", "", 1) if c.startswith("p_") else c)
        st.dataframe(show.sort_values("dk_pts", ascending=False).round(2), hide_index=True, width="stretch")
        if not S.odds_api_key:
            st.info("Add a free Odds API key in Settings to compare these against sportsbook prop lines.")
        else:
            from beatnfl import research
            mk = st.multiselect("Prop markets", list(__import__("beatnfl.sources.odds", fromlist=["x"]).PROP_MARKETS),
                                research.CORE_PROP_MARKETS)
            n_games = fc.game_id.nunique()
            st.caption(f"Estimated cost: about {len(mk) * n_games} Odds API credits ({len(mk)} markets × {n_games} games). "
                       "Free tier = 500/month.")
            if st.button("Load props"):
                with st.spinner("Pulling props…"):
                    try:
                        board, info = research.props_board(S.odds_api_key, int(season), int(week), fc, ss["pm"], mk,
                                                           offline=offline)
                        ss["board"] = board
                    except StepError as e:
                        st.error(e.as_text())
            b = ss.get("board")
            if b is not None:
                me = st.slider("Show where |model − book| probability ≥", 0.0, 0.3, 0.05, 0.01)
                st.dataframe(b[b.edge.abs() >= me], hide_index=True, width="stretch")
                st.caption("book_p_over = no-vig probability from that book's over/under prices. Edges shown are model "
                           "opinions; their track record against real lines appears in Evidence once games finish.")

# ------------------------------------------------------------------ EVIDENCE
with tab_evidence:
    from beatnfl.config import HOME as _H
    files = sorted(glob.glob(str(_H / "evidence" / "backtest_*.json")))
    bundled = Path(__file__).resolve().parent / "evidence" / "backtest_baseline.json"
    if not files and bundled.exists():
        files = [str(bundled)]
    if st.button("Re-run historical tests (≈3 min)"):
        from beatnfl import backtest
        with st.spinner("Testing…"):
            backtest.run(offline=offline, progress=lambda m: None)
        st.rerun()
    if files:
        ev = json.loads(Path(files[-1]).read_text())
        st.caption(f"{ev['label']} · generated {ev['generated']} · {ev['train_rule']}")
        st.markdown("**DK points — mean absolute error (lower is better)**")
        pl = pd.DataFrame(ev["players"])
        st.dataframe(pl[pl.season.astype(str) == "all"][["pos", "n", "mae_model", "mae_ewm", "mae_naive", "bias_model"]],
                     hide_index=True)
        st.markdown("**Decision test — average actual points of each week's top-N picks**")
        st.dataframe(pd.DataFrame(ev["picks"]), hide_index=True)
        st.markdown("**Props — stat forecasts and probability quality** (lines here are baseline-derived, not sportsbook)")
        st.dataframe(pd.DataFrame(ev["props"]).drop(columns=["calibration"], errors="ignore"), hide_index=True)
        st.markdown("**DST**")
        st.dataframe(pd.DataFrame(ev["dst"]), hide_index=True)
        g = ev["games"]
        st.markdown(f"**Games vs closing line** ({g['seasons']}, {g['n_games']} games; break-even at -110 = 52.4%)")
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(g["ats"]), hide_index=True)
        c2.dataframe(pd.DataFrame(g["totals"]), hide_index=True)
        st.caption(f"Margin MAE: {g['mae_margin']} · Total MAE: {g['mae_total']}")
    sc = sorted(glob.glob(str(_H / "evidence" / "sleeper_compare_*.json")))
    if sc:
        c = json.loads(Path(sc[-1]).read_text())
        if c.get("n"):
            st.markdown(f"**Sleeper vs in-house model** ({c['n']:,} player-games, {c['seasons']}) — "
                        f"lowest error at model weight **{c['best_model_weight']}** (rest Sleeper)")
            st.dataframe(pd.DataFrame(c["by_pos"]), hide_index=True)
    else:
        st.caption("Run `python -m beatnfl backtest --sleeper` on your computer to test Sleeper's published projections "
                   "against the model and pick blend weights from evidence.")
    mab = Path(__file__).resolve().parent / "evidence" / "matchup_ab_test.json"
    if mab.exists():
        m = json.loads(mab.read_text())
        st.markdown(f"**Matchup stats as model inputs** — {m['verdict']}")
        st.dataframe(pd.DataFrame(m["rows"]), hide_index=True)
    st.subheader("Live grading (true pregame forecasts)")
    if st.button("Grade frozen forecasts"):
        from beatnfl import backtest, research
        gr = backtest.grade_frozen(offline=offline)
        st.dataframe(gr if not gr.empty else pd.DataFrame({"note": ["No finished slates with pregame runs yet."]}),
                     hide_index=True)
        st.write(research.grade_props(offline=offline))

# ------------------------------------------------------------------ STATUS
with tab_status:
    if st.button("Test every source now"):
        from beatnfl import doctor
        lines_out = []
        doctor.run(lines_out.append)
        st.code("\n".join(lines_out))
    stt = storage.data_status()
    st.dataframe(stt if not stt.empty else pd.DataFrame({"note": ["No runs yet."]}), hide_index=True,
                 width="stretch")
    if LOG_PATH.exists():
        with st.expander("Technical log (last 60 lines)"):
            st.code("\n".join(LOG_PATH.read_text().splitlines()[-60:]))
