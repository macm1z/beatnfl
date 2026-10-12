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


APP_ROOT = Path(__file__).resolve().parent
if not (APP_ROOT / "beatnfl").exists():  # launched through a wrapper script
    import beatnfl as _bn
    APP_ROOT = Path(_bn.__file__).resolve().parent.parent
ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}


def show_steps(steps):
    """One compact checklist. Problems are listed first with their fix; routine steps sit underneath."""
    fails = [x for x in steps if x.status == "fail"]
    warns = [x for x in steps if x.status == "warn"]
    head = (f"❌ {len(fails)} step(s) failed" if fails else
            f"⚠️ Ready, with {len(warns)} thing(s) to check" if warns else "✅ Everything loaded")
    with st.container(border=True):
        st.markdown(f"**{head}**")
        for x in fails + warns:
            st.markdown(f"{ICON[x.status]} **{x.step}** — {x.message}" + (f"  \n  ➜ {x.fix}" if x.fix else ""))
        rest = [x for x in steps if x.status in ("ok", "skip")]
        if rest:
            with st.expander(f"Details ({len(rest)} steps)", expanded=not (fails or warns)):
                for x in rest:
                    st.markdown(f"{ICON[x.status]} **{x.step}** — {x.message}" + (f"  \n  ➜ {x.fix}" if x.fix else ""))


def tidy(df: pd.DataFrame, rename: dict, ints=(), one=(), two=(), pct=(), money=()) -> tuple[pd.DataFrame, dict]:
    """Readable table: friendly column names, empty columns dropped, sensible number formats.
    Returns (frame, column_config) for st.data_editor; read-only tables go through show()."""
    d = df[[c for c in rename if c in df]].rename(columns=rename).copy()
    fmts = {}
    for cols, fmt in ((ints, "%d"), (one, "%.1f"), (two, "%.2f"), (money, "$%d")):
        for c in cols:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                fmts[c] = fmt
    for c in pct:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce") * 100
            fmts[c] = "%d%%"
    d = d[[c for c in d.columns if c == "dk_id" or not d[c].isna().all()]]
    for c in d.columns:
        if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            d[c] = d[c].astype(object).where(d[c].notna(), "").replace({"None": "", "nan": "", "<NA>": ""})
    cfg = {c: st.column_config.NumberColumn(c, format=f) for c, f in fmts.items() if c in d.columns}
    d.attrs["fmts"] = {c: f for c, f in fmts.items() if c in d.columns}
    return d, cfg


def show(d: pd.DataFrame, cfg: dict | None = None, **kw):
    """Read-only table, already in a sensible order: formatted numbers, blank where a stat doesn't apply."""
    fmts = d.attrs.get("fmts", {})
    out = d.copy()
    for c in out.columns:
        f = fmts.get(c)
        if f:
            out[c] = [("" if pd.isna(v) else (f % v)) for v in out[c]]
        elif out[c].dtype.kind == "f":
            out[c] = ["" if pd.isna(v) else f"{v:g}" for v in out[c]]
    st.dataframe(out, hide_index=True, width=kw.pop("width", "stretch"), **kw)


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
        names = {"dk_id": "dk_id", "name": "Player", "pos": "Pos", "team": "Team", "opp": "Opp", "salary": "Salary",
                 "status": "Status", "injury_note": "Injury", "proj": "Proj", "value": "Pts/$1K", "proj_paid": "Yours",
                 "proj_sleeper": "Sleeper", "proj_model": "Model", "pct_rostered_yahoo": "Yahoo %", "sources": "Sources"}
        view, cfg = tidy(pool, names, one=("Proj", "Yours", "Sleeper", "Model"), two=("Pts/$1K",), money=("Salary",),
                         ints=("Yahoo %",))
        view.insert(1, "Lock", view.dk_id.isin(ss.get("locks", [])))
        view.insert(2, "Exclude", view.dk_id.isin(ss.get("excludes", [])))
        f1, f2 = st.columns([1, 3])
        posf = f1.multiselect("Positions", ["QB", "RB", "WR", "TE", "DST"], [])
        if posf:
            view = view[view.Pos.isin(posf)]
        f2.caption("Blank projection = no source covered that player; they're left out of lineups. "
                   "Yours / Sleeper / Model are the individual sources behind **Proj**.")
        cfg["dk_id"] = None  # hidden; kept for locks/excludes
        ed = st.data_editor(view.sort_values("Proj", ascending=False), hide_index=True, width="stretch", height=420,
                            column_config=cfg, disabled=[c for c in view.columns if c not in ("Lock", "Exclude")])
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
                            "Copies of both files are also in your BeatNFL/exports folder.")
                st.dataframe(export.lineup_table(res.lineups, r.pool), hide_index=True, width="stretch", column_config={"Salary": st.column_config.NumberColumn("Salary", format="$%d")})
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
        sv = sets.copy()
        sv["Created"] = pd.to_datetime(sv.created_utc, utc=True).dt.tz_convert("America/New_York").dt.strftime("%a %b %-d, %-I:%M %p")
        st.dataframe(sv.rename(columns={"id": "#", "name": "Name", "n": "Lineups", "slate": "Slate", "status": "Result"})[
            ["#", "Name", "Created", "Lineups", "Slate", "Result"]], hide_index=True, width="stretch")
        sid = st.selectbox("Open set", sets.id.tolist(), format_func=lambda i: f"#{i} · {sets.set_index('id').at[i, 'name']}")
        d = storage.load_lineup_set(int(sid))
        _, pool = storage.get_projection_run(d["projection_run_id"])
        bad = [i + 1 for i, lu in enumerate(d["lineups"]) if optimizer.validate(lu, pool)]
        st.caption(f"{len(d['lineups'])} lineups · created "
                   f"{pd.Timestamp(d['created_utc']).tz_convert('America/New_York'):%a %b %-d, %-I:%M %p} · "
                   + ("all valid against the saved slate" if not bad else f"invalid: {bad}"))
        st.dataframe(export.lineup_table(d["lineups"], pool), hide_index=True, width="stretch", column_config={"Salary": st.column_config.NumberColumn("Salary", format="$%d")})
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
                season_y, wk_default = nv.upcoming_week(sched)
                c1, c2 = st.columns([3, 1])
                labels = [t["name"] for t in teams]
                pick = c1.selectbox("Team", labels)
                yweek = c2.number_input("Week", 1, 18, wk_default, key="yweek")
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
                    ytbl, cfg = tidy(df, {"slot": "Yahoo slot", "name": "Player", "pos": "Pos", "team": "Team", "opp": "Opp",
                                          "status": "Status", "injury_note": "Injury", "proj": "Proj pts",
                                          "p_pass_yds": "Pass yds", "p_rush_yds": "Rush yds", "p_rec": "Rec",
                                          "p_rec_yds": "Rec yds", "p_any_td": "TD chance"},
                                     one=("Proj pts", "Rec"), ints=("Pass yds", "Rush yds", "Rec yds"), pct=("TD chance",))
                    show(ytbl)
                    model_slots = [(s_, n) for s_, n in slots if s_ not in ("K", "DEF")]
                    sug = yahoo.suggest(rows, model_slots)
                    cur = {r["player_key"] for r in rows if r["slot"] not in ("BN", "IR", "IR+", "NA", "K", "DEF")}
                    best = {s_["player_key"] for s_ in sug if s_["player_key"]}
                    st.markdown("**Suggested starters** (kickers and defenses not projected — keep your choice)")
                    sg, sgc = tidy(pd.DataFrame(sug), {"slot": "Slot", "player": "Start", "proj": "Proj pts"}, one=("Proj pts",))
                    sg["Start"] = sg["Start"].replace({"— nobody projected —": "— no healthy, projected player —"})
                    show(sg)
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
        default = nv.upcoming_week(sched)
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
        gt = research.games_table(int(season), int(week), lines, offline=offline)
        def _fav(game, m):
            if pd.isna(m):
                return ""
            away, home = game.split(" @ ")
            return f"{home if m > 0 else away} -{abs(m):.1f}" if abs(m) >= 0.05 else "Pick'em"
        gt["Model spread"] = [_fav(g, m) for g, m in zip(gt.Game, gt["Model margin (home)"])]
        gv, gc = tidy(gt, {"Game": "Game", "Market": "Spread", "Model spread": "Model spread", "Market total": "Total",
                           "Model total": "Model total", "P(home covers)": "Home covers %", "Lines from": "Lines from"},
                      one=("Total", "Model total"), pct=("Home covers %",))
        show(gv)
        st.caption("Model spread/total = the game model alone. Home covers % blends the model with the market, "
                   "weighted 75% to the market.")
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
        mu = ss["mu"].sort_values("Pass edge", ascending=False).copy()
        mu["Success rate"] = mu["Success rate"] * 100
        cfgm = {c: st.column_config.NumberColumn(c, format="%.2f") for c in ("Off EPA/play", "Opp def EPA allowed", "Pass edge", "Run edge")}
        cfgm.update({c: st.column_config.NumberColumn(c, format="%.0f") for c in
                     ("Success rate", "Explosive %", "Pass rate over exp. %", "Red zone TD %", "Sack rate %", "Opp sack rate %", "Plays/game")})
        cfgm["Success rate"] = st.column_config.NumberColumn("Success %", format="%.0f")
        st.dataframe(mu, hide_index=True, width="stretch", column_config=cfgm)

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
                ss["fc"], ss["pm"], ss["fc_week"] = fc, pm, (int(season), int(week))
                for w in warns:
                    st.warning(w)
            except StepError as e:
                st.error(e.as_text())
    fc = ss.get("fc")
    if fc is None:
        st.caption("Click the button to forecast every player in this week's games.")
    else:
        names = {"name": "Player", "position": "Pos", "team": "Team", "opp": "Opp", "injury": "Status",
                 "p_dk_pts": "DK pts", "p_pass_yds": "Pass yds", "p_pass_att": "Pass att", "p_rush_yds": "Rush yds",
                 "p_rush_att": "Rush att", "p_rec": "Rec", "p_rec_yds": "Rec yds", "p_any_td": "TD chance"}
        if "yahoo_pct_rostered" in fc and fc.yahoo_pct_rostered.notna().any():
            names.update({"yahoo_status": "Yahoo status", "yahoo_pct_rostered": "Yahoo %", "yahoo_pct_change": "Yahoo % chg"})
        fv, fcfg = tidy(fc.sort_values("p_dk_pts", ascending=False), names, one=("DK pts", "Rec", "Yahoo % chg"),
                        ints=("Pass yds", "Pass att", "Rush yds", "Rush att", "Rec yds", "Yahoo %"), pct=("TD chance",))
        show(fv)

    st.subheader("Props")
    if not S.odds_api_key:
        st.warning("**Props need a free Odds API key** — that's where the sportsbook lines come from.\n\n"
                   "1. Sign up at [the-odds-api.com](https://the-odds-api.com/) (free plan, key arrives by email).\n"
                   "2. Paste it into **Settings** in the sidebar and click **Save settings**.\n\n"
                   "Forecasts above, lineups and the prop checker in the test page work without it.")
    else:
        from beatnfl import research
        from beatnfl.sources import odds as _odds
        mk = st.multiselect("Prop markets", list(_odds.PROP_MARKETS), research.CORE_PROP_MARKETS,
                            format_func=lambda m: research.STAT_NAMES.get(_odds.PROP_MARKETS.get(m, m), m))
        n_games = int(fc.game_id.nunique()) if fc is not None else 14
        st.caption(f"Each load costs about {len(mk) * n_games} Odds API credits ({len(mk)} markets × {n_games} games). "
                   "Free plan = 500/month, so about one full load per week. Books post most player props "
                   "2–4 days before kickoff — loading earlier returns little or nothing.")
        if st.button("Load props"):
            with st.spinner("Pulling props…"):
                try:
                    if ss.get("fc") is None or ss.get("fc_week") != (int(season), int(week)):
                        from beatnfl.models import train
                        pm, dm, meta = train.get_models(offline=offline)
                        ss["fc"], _ = train.forecast_week(int(season), int(week), None, pm, lines, offline=offline)
                        ss["pm"], ss["fc_week"] = pm, (int(season), int(week))
                    board, info = research.props_board(S.odds_api_key, int(season), int(week), ss["fc"], ss["pm"], mk,
                                                       offline=offline)
                    ss["board"] = board
                    st.success(f"{len(board)} prop lines from {info['games']} games.")
                except StepError as e:
                    ss["board"] = None
                    st.error(e.as_text())
        b = ss.get("board")
        if b is not None and not b.empty:
            c1, c2, c3 = st.columns(3)
            games = ["All games"] + sorted(b.game.unique())
            pick_g = c1.selectbox("Game", games, key="props_game")
            books = [research.CONSENSUS] + sorted({research.book_name(x) for x in b.book.unique()})
            pick_b = c2.selectbox("Sportsbook", books, key="props_book")
            pick_s = c3.selectbox("Sort by", ["Biggest difference", "Sportsbook", "Player"], key="props_sort")
            only = st.toggle("Only show props where the model disagrees with the book", value=False, key="props_only")
            view = research.props_view(b, pick_g, pick_b, pick_s)
            if only and not view.empty:
                view = view[view.Lean.str.contains("Over|Under|Yes")]
            if view.empty:
                st.info("No props match these filters.")
            else:
                st.dataframe(view, hide_index=True, width="stretch", height=min(700, 38 + 35 * len(view)))
            st.caption("**Model %** = model's chance of the Over (or Yes). **Book %** = the book's chance with its margin "
                       "removed; Anytime TD is quoted on one side only, so its Book % still includes the margin and runs "
                       "high. **Lean** shows where the model differs by 4+ points; ★ = 8+ points. Information only.")

# ------------------------------------------------------------------ EVIDENCE
with tab_evidence:
    from beatnfl.config import HOME as _H
    files = sorted(glob.glob(str(_H / "evidence" / "backtest_*.json")))
    bundled = APP_ROOT / "evidence" / "backtest_baseline.json"
    if not files and bundled.exists():
        files = [str(bundled)]
    if st.button("Re-run historical tests (≈3 min)"):
        from beatnfl import backtest
        with st.spinner("Testing…"):
            backtest.run(offline=offline, progress=lambda m: None)
        st.rerun()
    if files:
        ev = json.loads(Path(files[-1]).read_text())
        st.caption(f"Historical test, generated {ev['generated'][:10]}: each season {ev['seasons'][0]}–{ev['seasons'][-1]} "
                   "is forecast by a model trained only on earlier seasons, using only games before each forecast.")
        st.markdown("**DK points — average miss per player-game (lower is better)**")
        pl = pd.DataFrame(ev["players"])
        t, _ = tidy(pl[pl.season.astype(str) == "all"], {"pos": "Pos", "n": "Player-games", "mae_model": "Model",
                    "mae_ewm": "Recent form", "mae_naive": "Season average", "bias_model": "Model bias"},
                    ints=("Player-games",), two=("Model", "Recent form", "Season average", "Model bias"))
        show(t)
        st.markdown("**Decision test — each week's top-N picks by method, average actual DK points**")
        t, _ = tidy(pd.DataFrame(ev["picks"]), {"pos": "Pos", "top_n": "Top N", "weeks": "Weeks", "avg_pts_model": "Model picks",
                    "avg_pts_naive": "Season-avg picks", "avg_pts_ewm": "Recent-form picks", "avg_pts_oracle": "Perfect hindsight",
                    "model_beats_naive_weeks": "Model better in"},
                    ints=("Top N", "Weeks"), one=("Model picks", "Season-avg picks", "Recent-form picks", "Perfect hindsight"))
        t["Model better in"] = t["Model better in"].astype(str) + " of weeks"
        show(t)
        st.markdown("**Prop stats** — lines in this test come from recent form, not sportsbooks; real lines are graded as you use Props")
        pr = pd.DataFrame(ev["props"])
        td = pr[pr.stat == "any_td"]
        pr = pr[pr.stat != "any_td"].assign(stat=lambda d: d.stat.map(lambda x: {"pass_yds": "Pass yds", "pass_att": "Pass att",
                 "pass_cmp": "Completions", "rush_yds": "Rush yds", "rush_att": "Rush att", "rec": "Receptions", "rec_yds": "Rec yds"}.get(x, x)))
        t, _ = tidy(pr, {"stat": "Stat", "n": "Games", "mae_model": "Miss: model", "mae_baseline": "Miss: recent form",
                    "brier_model": "Prob. score: model", "brier_coinflip": "Prob. score: 50/50", "confident_hit_rate": "Confident calls right"},
                    ints=("Games",), two=("Miss: model", "Miss: recent form"), pct=("Confident calls right",))
        t.attrs["fmts"].update({"Prob. score: model": "%.3f", "Prob. score: 50/50": "%.3f"})
        show(t)
        if len(td):
            st.caption(f"Anytime TD probability score: model {td.brier_model.iloc[0]:.3f} vs recent-form {td.brier_baseline.iloc[0]:.3f} "
                       "(lower is better). Probability score = Brier score.")
        st.markdown("**Defenses (DST)**")
        t, _ = tidy(pd.DataFrame(ev["dst"]), {"season": "Season", "n": "Games", "mae_model": "Miss: model", "mae_ewm": "Miss: recent form",
                    "rank_corr_model": "Ranking accuracy: model", "rank_corr_ewm": "Ranking accuracy: recent form"},
                    two=("Miss: model", "Miss: recent form", "Ranking accuracy: model", "Ranking accuracy: recent form"))
        t["Season"] = t["Season"].astype(str)
        show(t)
        g = ev["games"]
        st.markdown(f"**Games vs closing line** — {g['n_games']:,} games, {g['seasons']}. Break-even at -110 odds is 52.4%.")
        c1, c2 = st.columns(2)
        for col, key, lab in ((c1, "ats", "Spread"), (c2, "totals", "Total")):
            t = pd.DataFrame(g[key]).assign(edge=lambda d: d.edge_pts.map(lambda x: f"{x}+ pts"))
            t, _ = tidy(t, {"edge": f"{lab}: model differs by", "n": "Games", "win_rate": "Model side won"}, ints=("Games",),
                        pct=("Model side won",))
            with col:
                show(t)
        st.caption(f"Average miss on the final margin: market {g['mae_margin']['spread_line']:.2f}, model "
                   f"{g['mae_margin']['model_spread']:.2f} points. On the total: market {g['mae_total']['total_line']:.2f}, "
                   f"model {g['mae_total']['model_total']:.2f}. The market is more accurate.")
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
    mab = APP_ROOT / "evidence" / "matchup_ab_test.json"
    if mab.exists():
        m = json.loads(mab.read_text())
        st.markdown(f"**Matchup stats as model inputs** — {m['verdict']}")
        t = pd.DataFrame(m["rows"]).assign(Season=lambda d: d.Season.astype(str))
        st.dataframe(t, hide_index=True)
    st.subheader("Live grading (true pregame forecasts)")
    if st.button("Grade frozen forecasts"):
        from beatnfl import backtest, research
        gr = backtest.grade_frozen(offline=offline)
        st.dataframe(gr if not gr.empty else pd.DataFrame({"note": ["No finished slates with pregame runs yet."]}),
                     hide_index=True)
        st.write(research.grade_props(offline=offline))

# ------------------------------------------------------------------ STATUS
with tab_status:
    st.markdown("**Something not working?** Click below, then copy the block (copy icon, top right) and paste it to Claude. "
                "It contains no keys or passwords.")
    if st.button("Copy diagnostics", type="primary"):
        import platform
        from beatnfl import doctor, update as _u
        rep = [f"BeatNFL {__version__} · commit {(_u.installed() or 'unknown')[:7]} · {platform.platform()} · Python {platform.python_version()}",
               "Keys set: " + ", ".join(f"{k}={'yes' if getattr(S, k) else 'no'}" for k in
                                          ("odds_api_key", "yahoo_client_id", "yahoo_client_secret", "github_token")),
               "", "== Source checks =="]
        doctor.run(rep.append)
        stt_ = storage.data_status()
        if not stt_.empty:
            rep += ["", "== Last results per step =="] + [f"{r.source}: {r.status} · {r.message}" for r in stt_.itertuples()]
        r_ = ss.get("run")
        if r_:
            rep += ["", "== Last run on this page =="] + [x.as_text() for x in r_.steps]
        if LOG_PATH.exists():
            rep += ["", "== Log (last 40 lines) =="] + LOG_PATH.read_text().splitlines()[-40:]
        st.code("\n".join(rep), language=None)
    if st.button("Test every source now"):
        from beatnfl import doctor
        lines_out = []
        doctor.run(lines_out.append)
        st.code("\n".join(lines_out))
    st.caption(f"Data folder: `{HOME}` · Drop DKSalaries.csv or projection CSVs in `{INBOX}` · Exports: `{EXPORTS}`")
    stt = storage.data_status()
    if stt.empty:
        st.info("No runs yet — press Run on the This week tab.")
    else:
        stt = stt.assign(When=pd.to_datetime(stt.last_attempt_utc, utc=True).dt.tz_convert("America/New_York").dt.strftime("%a %-I:%M %p"),
                         Result=stt.status.map({"ok": "✅ ok", "warn": "⚠️ check", "skip": "⏭️ skipped", "fail": "❌ failed",
                                                "failed": "❌ failed"}).fillna(stt.status))
        st.dataframe(stt.rename(columns={"source": "Step", "message": "Last message"})[["Step", "Result", "When", "Last message"]],
                     hide_index=True, width="stretch")
    if LOG_PATH.exists():
        with st.expander("Technical log (last 60 lines)"):
            st.code("\n".join(LOG_PATH.read_text().splitlines()[-60:]))
