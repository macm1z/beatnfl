"""BeatNFL — Streamlit app. Start with:  python -m beatnfl app   (or the BeatNFL app / Start BeatNFL shortcut)."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _drop_stale_modules():
    """After an update the server process may still hold the previous version's modules in memory
    (Streamlit re-reads app.py on every page load, but not the modules it imports). If the version on
    disk differs from the one loaded, forget the old modules so the new code is imported below."""
    import re
    loaded = sys.modules.get("beatnfl")
    if loaded is None:
        return
    init = Path(loaded.__file__)
    try:
        disk = re.search(r'__version__ = "([^"]+)"', init.read_text()).group(1)
    except Exception:  # noqa: BLE001
        return
    if disk != getattr(loaded, "__version__", None):
        for name in [m for m in sys.modules if m == "beatnfl" or m.startswith("beatnfl.")]:
            del sys.modules[name]


_drop_stale_modules()

from beatnfl import __version__, export, optimizer, pipeline, storage  # noqa: E402
from beatnfl.config import EXPORTS, HOME, INBOX, LOG_PATH, Settings, ensure_dirs  # noqa: E402
TZ = "America/New_York"   # all times shown in Eastern (NFL time), wherever the computer is
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
STATUS_TXT = {"Q": "Questionable", "D": "Doubtful", "O": "Out", "IR": "IR", "OUT": "Out", "P": "Probable"}

st.markdown("""<style>
.stTabs [role="tablist"] { gap: 8px; flex-wrap: wrap; border-bottom: none; }
.stTabs [data-testid="stTab"] { border: 1px solid rgba(139,148,158,.45); border-radius: 8px; padding: 5px 14px;
  height: auto; margin: 0; }
.stTabs [data-testid="stTab"][aria-selected="true"] { border-color: #3FB950; background: rgba(63,185,80,.12); }
.stTabs [role="tablist"] > div:not([data-testid="stTab"]) { display: none; }
</style>""", unsafe_allow_html=True)


def step_line(x) -> str:
    return f"{ICON[x.status]} **{x.step}** — {x.message}" + (f"  \n  ➜ {x.fix}" if x.fix else "")


def show_steps(steps):
    """Full checklist (Load data page). Problems first with their fix; routine steps underneath."""
    fails = [x for x in steps if x.status == "fail"]
    warns = [x for x in steps if x.status == "warn"]
    head = (f"❌ {len(fails)} step(s) failed" if fails else
            f"⚠️ Ready, with {len(warns)} thing(s) to check" if warns else "✅ Everything loaded")
    with st.container(border=True):
        st.markdown(f"**{head}**")
        for x in fails + warns:
            st.markdown(step_line(x))
        rest = [x for x in steps if x.status in ("ok", "skip")]
        if rest:
            with st.expander(f"Details ({len(rest)} steps)", expanded=not (fails or warns)):
                for x in rest:
                    st.markdown(step_line(x))


def show_problems(steps):
    """Prediction pages: only what needs attention, folded away."""
    bad = [x for x in steps if x.status in ("fail", "warn")]
    if bad:
        n_f = sum(x.status == "fail" for x in bad)
        with st.expander(f"{'❌' if n_f else '⚠️'} {len(bad)} thing(s) to check from the last refresh"):
            for x in bad:
                st.markdown(step_line(x))
            st.caption("Full log: *Load data* tab.")


def tidy(df: pd.DataFrame, rename: dict, ints=(), one=(), two=(), pct=(), money=(), helps=None) -> tuple[pd.DataFrame, dict]:
    """Readable table: friendly column names, empty columns dropped, numbers kept numeric (so sorting works)
    and formatted through column_config. Returns (frame, column_config)."""
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
    helps = helps or {}
    cfg = {c: st.column_config.NumberColumn(c, format=f, help=helps.get(c)) for c, f in fmts.items() if c in d.columns}
    for c, h in helps.items():
        if c in d.columns and c not in cfg:
            cfg[c] = st.column_config.Column(c, help=h)
    return d, cfg


def show(d: pd.DataFrame, cfg: dict | None = None, **kw):
    """Read-only table. Numbers stay numbers, so clicking a column header sorts correctly."""
    st.dataframe(d, hide_index=True, width=kw.pop("width", "stretch"), column_config=cfg or {}, **kw)


def slate_banner():
    r = ss.get("run")
    if r and r.slate:
        sl = r.slate
        verified = "IDs verified" if sl.ids_verified else "IDs not yet verified vs official CSV"
        st.caption(f"**Slate:** {sl.slate.label if sl.slate else ''} · {r.season or '?'} week {r.week or '?'} · "
                   f"{verified} · loaded {ss.get('run_at', '')}")
    else:
        st.caption("No slate loaded yet — click **⟳ Refresh**.")


def week_picker(prefix: str):
    """Season/week for Games and Props — the two pages stay on the same week."""
    other = "p" if prefix == "g" else "g"

    def sync():
        ss[f"{other}_season"], ss[f"{other}_week"] = ss[f"{prefix}_season"], ss[f"{prefix}_week"]
        ss["week_auto"] = (int(ss[f"{prefix}_season"]), int(ss[f"{prefix}_week"])) == ss.get("current_week")

    def back_to_current():
        ss["week_auto"] = True
        ss["g_season"], ss["g_week"] = ss["p_season"], ss["p_week"] = ss["current_week"]
    c1, c2, c3, _ = st.columns([1, 1, 1.3, 2.7], vertical_alignment="bottom")
    c1.number_input("Season", 2015, 2035, key=f"{prefix}_season", on_change=sync)
    c2.number_input("Week", 1, 22, key=f"{prefix}_week", on_change=sync)
    if ss.get("week_auto", True):
        c3.caption("✓ current week")
    else:
        c3.button("↺ Current week", key=f"{prefix}_now", on_click=back_to_current)
    return int(ss[f"{prefix}_season"]), int(ss[f"{prefix}_week"])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏈 BeatNFL")
    st.caption(f"v{__version__} · research & predictions only — no bets, no contest entry")
    with st.expander("Settings"):
        key = st.text_input("The Odds API key (optional)", S.odds_api_key, type="password",
                            help="Free key at the-odds-api.com — powers live lines and props.")
        from beatnfl.net import odds_quota as _oq
        _q = _oq()
        if _q.get("remaining") is not None:
            st.caption(f"Odds API credits left this month: **{int(_q['remaining']):,}** (used {int(_q.get('used', 0)):,})")
        fpkey = st.text_input("FantasyPros API key (optional)", S.fantasypros_api_key, type="password",
                              help="Production key from secure.fantasypros.com/api-keys (MVP/HOF). Fills the 'Yours' "
                                   "projection automatically on every Refresh.")
        bpkey = st.text_input("BettingPros API key (optional)", S.bettingpros_api_key, type="password",
                              help="Your BettingPros Consumer API key (personal use). Adds BettingPros' prop projections "
                                   "and star ratings to Props, and builds their graded track record in Evidence.")
        if S.bettingpros_api_key:
            from beatnfl.sources import bettingpros as _bpu
            _u = _bpu.usage()
            st.caption(f"BettingPros requests today: **{_u['used']}** of {_u['limit']}"
                       + (f" · membership: {_u['tier']}" if _u.get("tier") else "") + " · resets 8 PM Eastern")
        akey = st.text_input("Anthropic API key (for the Ask tab)", S.anthropic_api_key, type="password",
                             help="platform.claude.com/settings/keys. Billed per question by Anthropic (separate from a "
                                  "Claude subscription); each answer shows its cost.")
        from beatnfl import ask as _ask
        _mk = list(_ask.MODELS)
        amodel = st.selectbox("Ask model", _mk, index=_mk.index(S.ask_model) if S.ask_model in _mk else 0,
                              format_func=lambda m: _ask.MODELS[m][0])
        yid = st.text_input("Yahoo Client ID (optional)", S.yahoo_client_id, type="password")
        ysec = st.text_input("Yahoo Client Secret (optional)", S.yahoo_client_secret, type="password")
        st.caption("Blend weights for the **Combined** projection" +
                   (" — *tested per-position weights from Evidence are in use; these apply only where none exist*"
                    if S.pos_weights else ""))
        c1, c2, c3 = st.columns(3)
        wp = c1.number_input("Yours", 0.0, 1.0, S.w_paid, 0.05, help="Your projections: FantasyPros API (key in Settings), or a pasted table / CSV")
        ws = c2.number_input("Sleeper", 0.0, 1.0, S.w_sleeper, 0.05)
        wm = c3.number_input("Model", 0.0, 1.0, S.w_model, 0.05)
        from beatnfl import research as _rsb
        _bk_all = list(dict.fromkeys(list(_rsb.BOOK_NAMES) + list(S.my_books)))
        mybooks = st.multiselect("My sportsbooks", _bk_all, [b for b in S.my_books if b in _bk_all],
                                 format_func=_rsb.book_name,
                                 help="Where you can bet. Best odds, bet slips and the Picks price are taken from these "
                                      "books only (offshore books are left out by default).")
        ubl = st.checkbox("Use sportsbook prop lines in the model", S.use_book_lines,
                          help="Moves the model's projection toward the books' player lines (≈5 Odds API credits per game, "
                               "reused for 3 hours). Tested on 2025: lower miss on unseen weeks.")
        qh = st.slider("Questionable haircut", 0.5, 1.0, S.questionable_haircut, 0.05,
                       help="Questionable players' Combined projection is multiplied by this.")
        if st.button("Save settings"):
            S.odds_api_key, S.w_paid, S.w_sleeper, S.w_model, S.questionable_haircut = key, wp, ws, wm, qh
            S.fantasypros_api_key = fpkey.strip()
            S.anthropic_api_key, S.ask_model = akey.strip(), amodel
            S.bettingpros_api_key = bpkey.strip()
            S.use_book_lines = bool(ubl)
            S.my_books = list(mybooks) or list(_rsb.DEFAULT_BOOKS)
            S.yahoo_client_id, S.yahoo_client_secret = yid.strip(), ysec.strip()
            S.save()
            st.success("Saved. Click ⟳ Refresh to apply new weights.")
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
    from beatnfl import schedule as _sched
    if _sched.supported():
        if "tz_synced" not in ss:
            ss["tz_synced"] = True
            _note = _sched.keep_in_sync()
            if _note:
                st.caption(_note)
        with st.expander("Automatic checks"):
            st.caption("Runs in the background while your Mac is awake and notifies you: Wednesday 7 PM picks list · "
                       "game-day lineup check after inactives post · closing prices before kickoffs · BettingPros "
                       "track record each evening (with a BettingPros key).")
            _on = st.toggle("On", value=_sched.installed(), key="auto_checks")
            if _on != _sched.installed():
                st.info(_sched.install(_on))
    offline = st.toggle("Offline mode (use cached data only)", False)

# Games / Props / Picks follow the current NFL week automatically (rechecked every 10 minutes), unless you
# pick another week — then "↺ Current week" brings it back.
import time as _time  # noqa: E402
if "current_week" not in ss or _time.time() - ss.get("current_week_at", 0) > 600:
    from beatnfl.sources import nflverse as _nv
    try:
        _d = _nv.upcoming_week(_nv.schedules(offline=offline))
    except Exception:  # noqa: BLE001
        _d = ss.get("current_week") or (_nv.current_season(), 1)
    ss["current_week"], ss["current_week_at"] = (int(_d[0]), int(_d[1])), _time.time()
if ss.get("week_auto", True) or "g_season" not in ss:
    ss["week_auto"] = True
    ss["g_season"], ss["g_week"] = ss["p_season"], ss["p_week"] = ss["current_week"]


def _slate_choice():
    """Draft group picked on the Load data page (None = main slate chosen automatically)."""
    sl = ss.get("slates")
    pick = ss.get("slate_pick")
    if sl and pick:
        labels = [s.label for s in sl]
        if pick in labels:
            return sl[labels.index(pick)].draft_group_id
    return None


# ------------------------------------------------------------------ header: slate + Refresh
h1, h2 = st.columns([6, 1], vertical_alignment="center")
with h1:
    slate_banner()
upload_mode = str(ss.get("slate_mode", "Automatic")).startswith("Upload")
refresh = h2.button("⟳ Refresh", type="primary", width="stretch",
                    disabled=upload_mode and ss.get("up_file") is None,
                    help="Reload slate, salaries, injuries, projections and lines using the choices on the Load data tab.")
if refresh:
    up, paid, pasted = ss.get("up_file"), ss.get("paid_file"), ss.get("pasted_proj") or ""
    status = st.status("Refreshing…", expanded=True)
    r = pipeline.run_week(S, salary_file=up.getvalue() if (upload_mode and up) else None,
                          draft_group_id=None if upload_mode else _slate_choice(),
                          paid_file=paid.getvalue() if paid else (pasted.encode() if pasted.strip() else None),
                          use_model=ss.get("use_model", True), offline=offline, progress=lambda m: status.write(m))
    status.update(label="Done", state="complete", expanded=False)
    ss["run"], ss["lineups"], ss["saved_info"] = r, None, None
    ss["run_at"] = pd.Timestamp.now(tz=TZ).strftime("%a %-I:%M %p")
    st.rerun()

tab_week, tab_ask, tab_saved, tab_games, tab_props, tab_picks, tab_parlay, tab_yahoo, tab_evidence, tab_load, tab_status = st.tabs(
    ["This week", "Ask", "Saved lineups", "Games", "Props", "Picks", "Parlay", "My Yahoo team", "Evidence", "Load data", "Data status"])

# ------------------------------------------------------------------ THIS WEEK
with tab_week:
    r = ss.get("run")
    if not r:
        st.info("Click **⟳ Refresh** (top right) to load this week's DraftKings slate, injuries and projections. "
                "To use a different slate, an uploaded salary file or your own projections, set that up on the "
                "**Load data** tab first.")
    else:
        show_problems(r.steps)
        try:
            al = pipeline.lineup_alerts(r)
        except Exception:  # noqa: BLE001
            al = pd.DataFrame()
        if not al.empty:
            bad = al[al.status.isin(["O", "IR", "D"])].drop_duplicates("dk_id")
            q = al[al.status == "Q"].drop_duplicates("dk_id")
            if len(bad):
                st.error("**Ruled out but in your saved lineups:** " + ", ".join(
                    f"{x.player} ({STATUS_TXT.get(x.status, x.status)}, set #{x.set}, {x.lineups} lineups)" for x in bad.itertuples())
                    + ". Rebuild, or swap them on DraftKings before lock.")
            if len(q):
                st.warning("Questionable in your saved lineups: " + ", ".join(f"{x.player} (set #{x.set})" for x in q.itertuples())
                           + ". Check again after inactives post (~90 minutes before kickoff).")
    if r and r.pool is not None and "proj" in r.pool:
        st.subheader("Player pool")
        pool = r.pool.copy()
        pool["injury"] = [(STATUS_TXT.get(str(s), str(s)) + (f" – {n}" if n and str(n) not in ("nan", "None") else ""))
                          if s and str(s) not in ("nan", "None") else "" for s, n in
                          zip(pool.get("status", ""), pool.get("injury_note", pd.Series("", index=pool.index)))]
        names = {"dk_id": "dk_id", "name": "Player", "pos": "Pos", "team": "Team", "opp": "Opp", "salary": "Salary",
                 "injury": "Injury", "proj": "Combined", "value": "Pts/$1K", "proj_paid": "Yours",
                 "proj_sleeper": "Sleeper", "proj_model": "Model", "p_dk_floor": "Floor", "p_dk_ceiling": "Ceiling",
                 "proj_lines_adj": "Book lines",
                 "pct_rostered_yahoo": "Yahoo %"}
        w = (f"Yours {S.w_paid:.0%} · Sleeper {S.w_sleeper:.0%} · Model {S.w_model:.0%}" if not S.pos_weights else
             "tested per position: " + "; ".join(f"{ps} " + " ".join(
                 f"{ {'paid': 'Yours', 'sleeper': 'Sleeper', 'model': 'Model'}.get(k, k)} {v:.0%}" for k, v in ww.items())
                 for ps, ww in S.pos_weights.items()))
        helps = {
            "Combined": f"The projection lineups are built from: weighted average of the sources ({w}), re-spread over "
                        f"whichever sources cover the player. Questionable players × {S.questionable_haircut}. "
                        "Change weights in Settings (sidebar).",
            "Pts/$1K": "Combined projection ÷ salary × 1,000 — points per $1,000 of salary.",
            "Yours": "Your projections: FantasyPros via its API when the key is set in Settings (DraftKings points computed "
                     "from their projected stats), or a table you pasted / uploaded on the Load data tab.",
            "Sleeper": "Sleeper's free published projection, scored with DraftKings rules. How it compares to the "
                       "model historically: Evidence tab → Sleeper vs model.",
            "Model": "BeatNFL's own forecast. Gradient-boosted model trained on every NFL game since 2012 (nflverse): "
                     "recent usage (snaps, targets, carries, weighted toward recent games), touches freed by injured "
                     "teammates, team pass/run volume, the opponent's points allowed to the position, and the "
                     "betting-implied team total. Accuracy: Evidence tab.",
            "Floor": "A bad-day score for this player (the model's 15th percentile): about 1 game in 7 goes lower.",
            "Ceiling": "A big-day score for this player (the model's 85th percentile): about 1 game in 7 goes higher. "
                       "Each player gets his own range — boom-or-bust players have a wider one. Useful for tournaments.",
            "Book lines": "Points the sportsbooks' prop lines moved the Model number (already included in Model). "
                          "In the 2025 test, folding the lines in cut the model's average miss on unseen weeks.",
            "Injury": "Worst of DraftKings, Sleeper and Yahoo. Out / Doubtful / IR players are left out of lineups.",
            "Yahoo %": "Share of Yahoo leagues rostering the player.",
        }
        view, cfg = tidy(pool, names, one=("Combined", "Yours", "Sleeper", "Model", "Floor", "Ceiling", "Book lines"), two=("Pts/$1K",), money=("Salary",),
                         ints=("Yahoo %",), helps=helps)
        view.insert(1, "Lock", view.dk_id.isin(ss.get("locks", [])))
        view.insert(2, "Exclude", view.dk_id.isin(ss.get("excludes", [])))
        f1, f2, f3 = st.columns([2, 3, 2])
        posf = f1.multiselect("Positions", ["QB", "RB", "WR", "TE", "DST"], [], placeholder="All positions")
        teamf = f2.multiselect("Teams", sorted(view.Team.unique()), [], placeholder="All teams")
        q = f3.text_input("Find player", "", placeholder="Name…")
        if posf:
            view = view[view.Pos.isin(posf)]
        if teamf:
            view = view[view.Team.isin(teamf)]
        if q.strip():
            view = view[view.Player.str.contains(q.strip(), case=False, regex=False)]
        st.caption("**Combined** is the number lineups use; Yours / Sleeper / Model are the sources behind it "
                   "(hover a column name for details). Blank = no source covered that player, so they're left out.")
        cfg["Injury"] = st.column_config.Column("Injury", help=helps["Injury"], width="medium")
        cfg["dk_id"] = None  # hidden; kept for locks/excludes
        cfg["Lock"] = st.column_config.CheckboxColumn("Lock", help="Force into every lineup")
        cfg["Exclude"] = st.column_config.CheckboxColumn("Exclude", help="Keep out of every lineup")
        ed = st.data_editor(view.sort_values("Combined", ascending=False), hide_index=True, width="stretch", height=420,
                            column_config=cfg, disabled=[c for c in view.columns if c not in ("Lock", "Exclude")])
        ss["locks"] = ed[ed.Lock].dk_id.tolist() + [l for l in ss.get("locks", []) if l not in set(view.dk_id)]
        ss["excludes"] = ed[ed.Exclude].dk_id.tolist() + [l for l in ss.get("excludes", []) if l not in set(view.dk_id)]

        st.subheader("Build lineups")
        ct_keys = list(optimizer.CONTESTS)
        contest = st.radio("Contest", ct_keys, index=ct_keys.index(S.contest) if S.contest in ct_keys else 1,
                           format_func=optimizer.CONTESTS.get, horizontal=True, key="contest",
                           help="Sets the builder for the contest you're entering. Cash: highest projection, no stack "
                                "required. Single-entry tournament: highest projection with a QB stack. Both give your "
                                "best lineup first plus two backups that change 2+ players, in case of late news.")
        if contest != S.contest:
            S.contest = contest
            S.save()
        P = optimizer.preset(contest) if contest != "multi" else dict(
            n_lineups=S.n_lineups, max_exposure=S.max_exposure, min_unique=S.min_unique,
            stack_min=S.stack_min, bring_back=S.bring_back, randomness=S.randomness)
        c = st.columns(7)
        k = f"_{contest}"          # each contest type keeps its own control values
        o = optimizer.OptSettings(
            n_lineups=c[0].number_input("Lineups", 1, 150, P["n_lineups"], key="n" + k,
                                        help="Single entry: #1 is the one to enter; the rest are backups."),
            min_salary=c[1].number_input("Min salary", 40000, 50000, S.min_salary, 100, key="ms" + k),
            max_exposure=c[2].slider("Max exposure", 0.1, 1.0, float(P["max_exposure"]), 0.05, key="mx" + k,
                                     help="Most lineups any one player can appear in (share of the set)."),
            min_unique=c[3].number_input("Min changes", 1, 6, P["min_unique"], key="mu" + k,
                                         help="Players each lineup must differ by."),
            stack_min=c[4].number_input("QB stack", 0, 3, P["stack_min"], key="sk" + k,
                                        help="Pass-catchers from the QB's team."),
            bring_back=c[5].number_input("Bring-back", 0, 2, P["bring_back"], key="bb" + k,
                                         help="Players from the QB's opponent."),
            randomness=c[6].slider("Randomness", 0.0, 0.4, float(P["randomness"]), 0.02, key="rn" + k,
                                   help="Projection noise per lineup, for variety (multi-entry only; 0 = pure best)."),
            locks=ss.get("locks", []), excludes=ss.get("excludes", []))
        b1, b2, b3, b4 = st.columns([3, 1.3, 1.3, 4], vertical_alignment="bottom")
        name = b1.text_input("Name for this set", f"{r.season}_wk{r.week}" if r.week else "lineups")
        if b2.button("⚙ Build lineups", type="primary", width="stretch"):
            with st.spinner("Optimizing…"):
                try:
                    ss["lineups"], ss["saved_info"] = pipeline.build_lineups(r, o, name=name), None
                except Exception as e:  # noqa: BLE001
                    st.error(f"Lineup build failed: {e}")
        out = ss.get("lineups")
        can_save = bool(out and out["result"].lineups) and not ss.get("saved_info")
        if b3.button("💾 Save", width="stretch", disabled=not can_save,
                     help="Save this set (and write both CSVs to your BeatNFL/exports folder). Identical sets aren't saved twice."):
            try:
                ss["saved_info"] = pipeline.save_lineups(r, out, name)
            except Exception as e:  # noqa: BLE001
                st.error(f"Save failed: {e}")
        with b4:
            if out:
                res = out["result"]
                msg = f"Built **{len(res.lineups)}** of {res.requested} · {res.status} · {res.elapsed}s"
                si = ss.get("saved_info")
                if si and si.get("set_id"):
                    msg += (f" · already saved as set #{si['set_id']}" if si["duplicate"] else f" · saved as set #{si['set_id']}")
                elif res.lineups:
                    msg += " · not saved yet"
                (st.success if res.status == "complete" else st.warning if res.lineups else st.error)(msg)
        if out:
            res = out["result"]
            for d in res.diagnostics:
                st.write("•", d)
            if res.lineups:
                st.dataframe(export.lineup_table(res.lineups, r.pool), hide_index=True, width="stretch", column_config={
                    "Rank": st.column_config.NumberColumn("Rank", help="1 = highest Combined projection. Lineups are listed best-first."),
                    "Proj": st.column_config.NumberColumn("Proj", format="%.1f", help="Sum of the nine players' Combined projections."),
                    "Salary": st.column_config.NumberColumn("Salary", format="$%d")})
                d1, d2, _ = st.columns([1, 1, 2])
                try:
                    d1.download_button("⬇ DraftKings upload CSV", export.dk_csv_text(res.lineups, r.pool),
                                       file_name=f"DK_upload_{name}.csv", mime="text/csv", width="stretch", key="dl_week_dk")
                except ValueError as e:
                    d1.error(str(e))
                d2.download_button("⬇ Readable lineups CSV", export.readable_text(res.lineups, r.pool),
                                   file_name=f"Lineups_{name}.csv", mime="text/csv", width="stretch", key="dl_week_rd")
                with st.expander("Exposure"):
                    st.dataframe(export.exposure(res.lineups, r.pool), hide_index=True)

# ------------------------------------------------------------------ SAVED
with tab_saved:
    sets = storage.list_lineup_sets()
    if sets.empty:
        st.info("No saved lineup sets yet. Build lineups on *This week* and click **💾 Save**.")
    else:
        dups = storage.duplicate_sets()
        sv = sets.copy()
        sv["Created"] = pd.to_datetime(sv.created_utc, utc=True).dt.tz_convert(TZ).dt.strftime("%a %b %-d, %-I:%M %p")
        sv["Week"] = [f"{int(s_)} wk {int(w_)}" if pd.notna(s_) and pd.notna(w_) else "" for s_, w_ in zip(sv.season, sv.week)]
        sv["Duplicate of"] = sv.id.map(lambda i: f"#{dups[i]}" if i in dups else "")
        sv.insert(0, "Select", False)
        sv = sv.rename(columns={"id": "Set", "name": "Name", "n": "Lineups"})
        cols = ["Select", "Set", "Name", "Created", "Week", "Lineups", "Duplicate of"]
        if not dups:
            cols.remove("Duplicate of")
        st.caption("Tick sets to delete them. Click a name to rename it.")
        ed = st.data_editor(sv[cols], hide_index=True, width="stretch", key=f"sets_editor_{ss.get('sets_ver', 0)}",
                            disabled=[c for c in cols if c not in ("Select", "Name")],
                            column_config={"Select": st.column_config.CheckboxColumn("Select", width="small"),
                                           "Set": st.column_config.NumberColumn("Set", format="#%d", width="small")})
        renamed = [(int(i), n) for i, n, o_ in zip(ed.Set, ed.Name, sv.Name) if str(n).strip() and n != o_]
        for i, n in renamed:
            try:
                storage.rename_lineup_set(i, str(n).strip(), int(sets.set_index("id").at[i, "revision"]))
            except RuntimeError as e:
                st.error(str(e))
        if renamed:
            ss["sets_ver"] = ss.get("sets_ver", 0) + 1
            st.rerun()
        chosen = [int(i) for i in ed[ed.Select].Set]
        a1, a2, a3 = st.columns([1.4, 1.6, 3])
        if a1.button(f"🗑 Delete selected ({len(chosen)})", disabled=not chosen, width="stretch"):
            ss["confirm_del"] = chosen
        if dups and a2.button(f"🧹 Remove {len(dups)} duplicate set(s)", width="stretch",
                              help="Deletes sets whose lineups are identical to an older saved set; the oldest copy is kept."):
            ss["confirm_del"] = sorted(dups)
        if ss.get("confirm_del"):
            ids = ss["confirm_del"]
            a3.warning(f"Delete {len(ids)} set(s): " + ", ".join(f"#{i}" for i in ids) + "? This can't be undone.")
            k1, k2, _ = st.columns([1.4, 1, 3.6])
            if k1.button("Yes, delete", type="primary", width="stretch"):
                for i in ids:
                    storage.delete_lineup_set(int(i))
                ss.pop("confirm_del")
                ss["sets_ver"] = ss.get("sets_ver", 0) + 1
                st.rerun()
            if k2.button("Cancel", width="stretch"):
                ss.pop("confirm_del")
                st.rerun()

        st.divider()
        ids = sets.id.tolist()
        o1, o2 = st.columns([2, 3])
        sid = o1.selectbox("Open set", ids, format_func=lambda i: f"#{i} · {sets.set_index('id').at[i, 'name']}")
        d = storage.load_lineup_set(int(sid))
        _, pool = storage.get_projection_run(d["projection_run_id"])
        idxp = pool.set_index("dk_id")
        players = sorted({idxp.at[x, "name"] for lu in d["lineups"] for x in lu if x in idxp.index})
        has = o2.multiselect("Only lineups with", players, placeholder="Any player")
        bad = [i + 1 for i, lu in enumerate(d["lineups"]) if optimizer.validate(lu, pool)]
        order = sorted(range(len(d["lineups"])), key=lambda i: -float(idxp.loc[d["lineups"][i]].proj.sum()))
        lus = [d["lineups"][i] for i in order]
        tbl = export.lineup_table(lus, pool)
        if has:
            keep = [all(any(idxp.at[x, "name"] == p for x in lu) for p in has) for lu in lus]
            tbl = tbl[keep]
        st.caption(f"{len(d['lineups'])} lineups · created "
                   f"{pd.Timestamp(d['created_utc']).tz_convert(TZ):%a %b %-d, %-I:%M %p} · "
                   + ("all valid against the saved slate" if not bad else f"invalid: {bad}")
                   + (f" · showing {len(tbl)} with {', '.join(has)}" if has else ""))
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={
            "Rank": st.column_config.NumberColumn("Rank", help="1 = highest projection when saved."),
            "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "Salary": st.column_config.NumberColumn("Salary", format="$%d")})
        e1, e2, _ = st.columns([1, 1, 2])
        try:
            e1.download_button("⬇ DraftKings upload CSV", export.dk_csv_text(lus, pool),
                               file_name=f"DK_upload_{d['name']}.csv", mime="text/csv", width="stretch", key="dl_saved_dk")
        except ValueError as e:
            e1.error(str(e))
        e2.download_button("⬇ Readable lineups CSV", export.readable_text(lus, pool),
                           file_name=f"Lineups_{d['name']}.csv", mime="text/csv", width="stretch", key="dl_saved_rd")

        st.divider()
        st.subheader("How did these lineups do?")
        if st.button("📊 Grade against actual results", key="grade_set",
                     help="After the games are final: each lineup's real DraftKings points vs projection, every player's "
                          "real score, and the best lineup that was possible on this slate in hindsight."):
            from beatnfl import grading
            with st.spinner("Reading final stats…"):
                try:
                    ss["graded"] = (int(sid), grading.grade_set(int(sid), offline=offline))
                except StepError as e:
                    st.error(e.as_text())
        gd = ss.get("graded")
        if gd and gd[0] == int(sid):
            g_ = gd[1]
            sm = g_["summary"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Average actual", sm["avg_actual"], f"{sm['avg_actual'] - sm['avg_proj']:+.1f} vs projected"
                      if sm["avg_actual"] is not None else None)
            m2.metric("Best lineup", sm["best"])
            m3.metric("Worst lineup", sm["worst"])
            m4.metric("Beat projection", f"{sm['beat_proj']} of {sm['graded']}")
            m5.metric("Best possible", sm["hindsight_best"], help="The highest-scoring lineup anyone could have built on this "
                      "slate with these salaries, knowing the results.")
            if sm["graded"] < sm["n"]:
                st.caption(f"{sm['n'] - sm['graded']} lineup(s) include games that aren't final yet.")
            L = g_["lineups"].copy()
            lv = pd.DataFrame({"Lineup": L.lineup, "Projected": L.proj, "Actual": L.actual, "Difference": (L.actual - L.proj).round(1),
                               **{f"P{k + 1}": [f"{pl[k][0]} {pl[k][3] if pl[k][3] is not None else '–'}" for pl in L.players]
                                  for k in range(9)}})
            lv = lv.rename(columns={f"P{k + 1}": c for k, c in enumerate(["QB", "RB", "RB2", "WR", "WR2", "WR3", "TE", "FLEX", "DST"])})
            show(lv.sort_values("Actual", ascending=False), {
                "Projected": st.column_config.NumberColumn("Projected", format="%.1f"),
                "Actual": st.column_config.NumberColumn("Actual", format="%.1f"),
                "Difference": st.column_config.NumberColumn("Difference", format="%+.1f")})
            st.caption("Each player cell shows his actual DraftKings points.")
            st.markdown("**Players you used**")
            P = g_["players"]
            show(P.rename(columns={"player": "Player", "pos": "Pos", "team": "Team", "salary": "Salary", "proj": "Projected",
                                   "actual": "Actual", "diff": "Difference", "lineups": "In lineups"}), {
                "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                "Projected": st.column_config.NumberColumn("Projected", format="%.1f"),
                "Actual": st.column_config.NumberColumn("Actual", format="%.1f"),
                "Difference": st.column_config.NumberColumn("Difference", format="%+.1f")})
            if g_["hindsight"]:
                st.markdown("**The best possible lineup — how we saw those players before kickoff**")
                ht = g_["hindsight"].get("table")
                if ht is not None and not ht.empty:
                    show(ht.rename(columns={"player": "Player", "pos": "Pos", "salary": "Salary", "actual": "Actual",
                                            "projected": "Our projection", "pos_rank": "Rank at position",
                                            "value_rank": "Value rank", "in_lineups": "In your lineups"}), {
                        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
                        "Rank at position": st.column_config.NumberColumn("Rank at position", help="Where our projection ranked him among his position's players (1 = highest)."),
                        "Value rank": st.column_config.NumberColumn("Value rank", help="Rank by projected points per dollar at his position.")})
                    st.caption("High ranks that weren't in your lineups = the settings left them out. Low ranks = the "
                               "projections didn't see the big game coming (common — most hindsight lineups are full of surprises).")
            if st.button("🔁 Which projection source would have done best?", key="cmp_src",
                         help="Rebuilds the same number of lineups with the same settings from each source alone "
                              "(as it stood before kickoff) and grades them on the real results."):
                from beatnfl import grading
                with st.spinner("Rebuilding and grading…"):
                    try:
                        ss["cmp_src"] = (int(sid), grading.compare_sources(int(sid), offline=offline))
                    except StepError as e:
                        st.error(e.as_text())
            cs = ss.get("cmp_src")
            if cs and cs[0] == int(sid) and not cs[1].empty:
                show(cs[1].rename(columns={"source": "Projections used", "lineups": "Lineups", "avg_projected": "Avg projected",
                                           "avg_actual": "Avg actual", "best": "Best", "worst": "Worst",
                                           "players_projected": "Players covered"}), {
                    "Avg projected": st.column_config.NumberColumn("Avg projected", format="%.1f"),
                    "Avg actual": st.column_config.NumberColumn("Avg actual", format="%.1f")})
                st.caption("One week is mostly luck — the same check over several weeks shows which source to trust.")


# ------------------------------------------------------------------ shared: betting lines
def raw_lines():
    """Odds API game lines (cached on disk for an hour), or None without a key."""
    if not S.odds_api_key:
        return None
    try:
        from beatnfl.sources import odds
        return odds.game_lines(S.odds_api_key, offline=offline)
    except StepError as e:
        st.warning(e.as_text())
        return None


def consensus_lines(raw):
    if raw is None or raw.empty:
        return None
    from beatnfl.sources import odds
    return odds.consensus(raw)


def fp_blend(fc, season, week):
    """Mix FantasyPros' stat lines into the stat forecasts by the tested shares (if any)."""
    if fc is None or not (S.fantasypros_api_key and S.stat_weights):
        return fc
    try:
        from beatnfl import research as _r
        from beatnfl.sources import fantasypros as _fp
        fpdf, _ = _fp.projections(S.fantasypros_api_key, season, week, offline=offline)
        fc, n = _r.blend_fp_stats(fc, fpdf, S.stat_weights)
        ss["fp_blend_note"] = f"{n} players' stat forecasts include FantasyPros (" + ", ".join(
            f"{k.replace('_', ' ')} {v:.0%}" for k, v in S.stat_weights.items() if v) + ", from Evidence)."
    except StepError as e:
        ss["fp_blend_note"] = "FantasyPros stats not mixed in: " + e.what
    return fc


# ------------------------------------------------------------------ GAMES
with tab_games:
    st.caption("Information only. The game model has **not** beaten closing lines historically — see Evidence. "
               "Treat model numbers as a second opinion, not a pick.")
    season, week = week_picker("g")
    raw_all = raw_lines()
    from beatnfl import research
    raw, book_pick = raw_all, None
    if raw_all is not None and not raw_all.empty:
        bk_keys = sorted(raw_all.book.unique(), key=lambda k: (k != "draftkings", research.book_name(k)))
        labels_b = [research.book_name(k) for k in bk_keys] + ["All books (median)"]
        pick_b = st.selectbox("Sportsbook", labels_b, key="games_book",
                              help="Lines and chances from one sportsbook, or the median across every book the Odds API covers.")
        if pick_b != "All books (median)":
            book_pick = bk_keys[labels_b.index(pick_b)]
            raw = raw_all[raw_all.book == book_pick]
        if raw_all.retrieved_stale.any():
            st.warning("Live lines couldn't be refreshed just now — these are the last ones downloaded. Check your "
                       "Odds API credits (Settings) and connection.")
    from beatnfl.sources import odds as _odds_src
    g_started = st.toggle("Include games already started", False, key="games_started",
                          help="Off: only games that haven't kicked off — the ones you can still bet before the game.")
    try:
        gt = research.games_table(season, week, raw, offline=offline)
        n_gone = int(research.started(gt._kick).sum())
        if not g_started:
            gt = gt[~research.started(gt._kick).values]
        if n_gone and not g_started:
            st.caption(f"{n_gone} game{'s' if n_gone != 1 else ''} already kicked off this week — hidden.")
        if gt.empty:
            st.info(f"No games left to play in {season} week {week}." if n_gone else f"No games found for {season} week {week}.")
        else:
            src = ((f"{research.book_name(book_pick)} via the Odds API" if book_pick else "Odds API — median of all books")
                   if (gt._src == "odds_api").any() else
                   "nflverse schedule (add an Odds API key in Settings for live lines from every book)")
            if book_pick and (gt._src != "odds_api").any():
                src += f"; games {research.book_name(book_pick)} hasn't posted use the schedule's consensus line"
            st.subheader("Lines")
            show(gt.drop(columns=["_src", "_kick"]), {
                "Spread": st.column_config.Column("Spread", help="Both teams' point spreads. Minus = favourite."),
                "Moneyline": st.column_config.Column("Moneyline", help="Odds to win outright (American odds)."),
                "Total": st.column_config.NumberColumn("Total", format="%.1f", help="Combined points over/under line."),
                "Model spread": st.column_config.Column("Model spread", help="BeatNFL's game model alone (team ratings from results this season and last)."),
                "Model total": st.column_config.NumberColumn("Model total", format="%.1f")}, height=38 + 35 * len(gt))
            st.caption(f"Lines from: {src}.")
            if raw_all is not None and not raw_all.empty:
                with st.expander("Every sportsbook's line for one game"):
                    gsel = st.selectbox("Game", list(gt.Game), key="games_book_game")
                    a_, h_ = gsel.split(" @ ")
                    rb = raw_all[(raw_all.home == h_) & (raw_all.away == a_)]
                    if rb.empty:
                        st.caption("No live lines for this game.")
                    else:
                        def _o(v):
                            return "" if pd.isna(v) else (f"+{int(v)}" if v > 0 else str(int(v)))
                        out_ = []
                        for bk, g_ in rb.groupby("book"):
                            sp_ = g_[g_.market == "spreads"].drop_duplicates("side").set_index("side")
                            ml_ = g_[g_.market == "h2h"].drop_duplicates("side").set_index("side")
                            to_ = g_[g_.market == "totals"].drop_duplicates("side").set_index("side")
                            ts_ = pd.to_datetime(g_.source_ts, utc=True, errors="coerce").max()
                            out_.append({
                                "Book": research.book_name(bk),
                                "Spread": " · ".join(f"{t} {sp_.at[t, 'line']:+g} ({_o(sp_.at[t, 'price'])})" for t in (a_, h_)
                                                      if t in sp_.index) if len(sp_) else "",
                                "Moneyline": " · ".join(f"{t} {_o(ml_.at[t, 'price'])}" for t in (a_, h_) if t in ml_.index),
                                "Total": (f"{to_.at['Over', 'line']:g} (o {_o(to_.at['Over', 'price'])} / u "
                                          f"{_o(to_.at['Under', 'price']) if 'Under' in to_.index else ''})" if "Over" in to_.index else ""),
                                "Updated": ts_.tz_convert(TZ).strftime("%a %-I:%M %p") if pd.notna(ts_) else "",
                                "Open": _odds_src.bet_link(next((x for x in g_.get("event_link", pd.Series(dtype=object))
                                                                 if isinstance(x, str)), None))})
                        st.dataframe(pd.DataFrame(out_), hide_index=True, width="stretch", column_config={
                            "Open": st.column_config.LinkColumn("Open", display_text="Open ↗",
                                                                help="This game on the sportsbook's site.")})

            st.subheader("Chances: market vs model")
            gp = research.game_probs(season, week, raw, offline=offline)
            if not g_started:
                gp = gp[~research.started(gp._kick).values]
            from beatnfl.sources import odds as _odds_src
            gp = gp.join(research.game_best(gp, raw_all, S.my_books))
            gp["Bet slip"] = research.game_slips(gp, raw) if book_pick else gp["Best slip"]
            if research.GAME_LEAN is None:                  # the game model hasn't beaten the market — no recommendations
                gp = gp.drop(columns="Bet slip")
            else:
                gp.loc[~(gp.Difference >= research.GAME_LEAN), "Bet slip"] = None
            gp = gp.drop(columns="Best slip")
            f1, f2, f3 = st.columns(3)
            pg = f1.selectbox("Game", ["All games"] + list(gt.Game), key="games_game")
            pt = f2.selectbox("Bet type", ["All", "Spread", "Moneyline", "Total"], key="games_type")
            pshow = f3.selectbox("Show", ["Every side", "Only where model is higher than market"], key="games_show")
            v = gp
            if pg != "All games":
                v = v[v.Game == pg]
            if pt != "All":
                v = v[v["Bet type"] == pt]
            if pshow.startswith("Only"):
                v = v[v.Difference > 0].sort_values("Difference", ascending=False)
            if v.empty:
                st.info("Nothing matches these filters.")
            else:
                gsel = st.dataframe(v, hide_index=True, width="stretch", key="games_tbl", on_select="rerun",
                                    selection_mode="multi-row", column_config={
                    "Odds": st.column_config.NumberColumn("Odds", format="%+d", help="Median American odds across books (or the selected book's)."),
                    "Market %": st.column_config.NumberColumn("Market %", format="%.1f%%",
                                                              help="The sportsbooks' chance for this side with their margin removed."),
                    "Model %": st.column_config.NumberColumn("Model %", format="%.1f%%",
                                                             help="Market chance moved by how far BeatNFL's game model sits from the line "
                                                                  "(model weighted 25%, market 75%, because the model alone hasn't beaten closing lines)."),
                    "Difference": st.column_config.NumberColumn("Difference", format="%+.1f",
                                                                help="Model % − Market %, in percentage points. Click a column header to sort."),
                    "_kick": None,
                    "Best odds": st.column_config.NumberColumn("Best odds", format="%+d",
                                                               help="Highest payout for this side among your sportsbooks (Settings → My sportsbooks), at this line."),
                    "Best book": st.column_config.Column("Best book", help="Where the best odds are. '(at …)' = the most-quoted line, when the line shown is an average."),
                    "Bet slip": st.column_config.LinkColumn("Bet slip", display_text="Slip ↗", help="Only where the model is 4+ points above the market. Opens the sportsbook selected above (the best-odds book in the all-books view) with this bet added to the betslip. "
                                                     "You review and place it in the book yourself — BeatNFL never places bets.")},
                     height=min(700, 38 + 35 * len(v)))
                gpk = v.iloc[gsel.selection.rows] if gsel and gsel.selection.rows else v.iloc[0:0]
                if st.button(f"➕ Add {len(gpk)} to parlay" if len(gpk) else "➕ Add to parlay", disabled=gpk.empty,
                             key="games_to_parlay", help="Tick rows (left edge) first."):
                    from beatnfl import parlay as _pl
                    pl_ = ss.setdefault("parlay", [])
                    for r_ in gpk.to_dict("records"):
                        leg_ = _pl.game_leg(r_, raw_all)
                        if leg_["id"] not in {x["id"] for x in pl_}:
                            pl_.append(leg_)
                    st.success(f"Parlay has {len(pl_)} legs — open the **Parlay** tab.")
                st.caption("Spread, moneyline and total each have their own chance. Differences of a couple of points "
                           "are within normal noise. Click any column header to sort.")
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
                ss["mu"] = research.matchups(season, week, offline=offline)
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

# ------------------------------------------------------------------ PROPS
with tab_props:
    st.caption("Information only — player forecasts and how they compare with sportsbook prop lines.")
    season, week = week_picker("p")
    lines = consensus_lines(raw_lines())
    st.subheader("Player forecasts")
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
                fc, warns = train.forecast_week(season, week, spp, pm, lines, offline=offline)
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
                ss["fc"], ss["pm"], ss["fc_week"] = fp_blend(fc, season, week), pm, (season, week)
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
        g1, g2, g3 = st.columns([2, 3, 2])
        fpos = g1.multiselect("Positions", sorted(fv.Pos.unique()), [], placeholder="All positions", key="fc_pos")
        fteam = g2.multiselect("Teams", sorted(fv.Team.unique()), [], placeholder="All teams", key="fc_team")
        fq = g3.text_input("Find player", "", placeholder="Name…", key="fc_q")
        if fpos:
            fv = fv[fv.Pos.isin(fpos)]
        if fteam:
            fv = fv[fv.Team.isin(fteam)]
        if fq.strip():
            fv = fv[fv.Player.str.contains(fq.strip(), case=False, regex=False)]
        fv = fv[[c for c in fv.columns if not (pd.api.types.is_numeric_dtype(fv[c]) and fv[c].isna().all())]]
        show(fv, {k: v for k, v in fcfg.items() if k in fv.columns}, height=420)
        if ss.get("fp_blend_note") and S.stat_weights:
            st.caption(ss["fp_blend_note"])
        st.caption("Blank (None) = that stat isn't forecast for the player's position. Pick a position to see just its stats. "
                   "Click a column header to sort.")

    st.subheader("Props")
    if not S.odds_api_key:
        st.warning("**Props need a free Odds API key** — that's where the sportsbook lines come from.\n\n"
                   "1. Sign up at [the-odds-api.com](https://the-odds-api.com/) (free plan, key arrives by email).\n"
                   "2. Paste it into **Settings** in the sidebar and click **Save settings**.\n\n"
                   "Forecasts above and lineups work without it.")
    else:
        from beatnfl import research
        from beatnfl.sources import odds as _odds
        mk = st.multiselect("Prop markets", list(_odds.PROP_MARKETS), research.CORE_PROP_MARKETS,
                            format_func=lambda m: research.STAT_NAMES.get(_odds.PROP_MARKETS.get(m, m), m))
        n_games = int(fc.game_id.nunique()) if fc is not None and "game_id" in fc else 14
        st.caption(f"Each load costs about {len(mk) * n_games} Odds API credits ({len(mk)} markets × {n_games} games). "
                   "Free plan = 500/month, so about one full load per week. Books post most player props "
                   "2–4 days before kickoff — loading earlier returns little or nothing.")
        if st.button("Load props"):
            with st.spinner("Pulling props…"):
                try:
                    if ss.get("fc") is None or ss.get("fc_week") != (season, week):
                        from beatnfl.models import train
                        pm, dm, meta = train.get_models(offline=offline)
                        fc0, _ = train.forecast_week(season, week, None, pm, lines, offline=offline)
                        ss["fc"] = fp_blend(fc0, season, week)
                        ss["pm"], ss["fc_week"] = pm, (season, week)
                    board, info = research.props_board(S.odds_api_key, season, week, ss["fc"], ss["pm"], mk,
                                                       offline=offline)
                    ss["board"] = board
                    st.success(f"{len(board)} prop lines from {info['games']} games.")
                except StepError as e:
                    ss["board"] = None
                    st.error(e.as_text())
        b = ss.get("board")
        if b is not None and not b.empty:
            c1, c2, c3, c4 = st.columns(4)
            games = ["All games"] + sorted(b[~research.started(b.commence).values].game.unique()
                                           if "commence" in b and not ss.get("props_started") else b.game.unique())
            pick_g = c1.selectbox("Game", games, key="props_game")
            books = [research.CONSENSUS] + sorted({research.book_name(x) for x in b.book.unique()})
            pick_b = c2.selectbox("Sportsbook", books, key="props_book")
            pick_s = c3.selectbox("Sort by", ["Biggest difference", "Sportsbook", "Player"], key="props_sort")
            pick_f = c4.selectbox("Show", ["Every prop", "Only where model is higher (Over / Yes)",
                                           "Only where model disagrees (4+ pts)"], key="props_show")
            p_started = st.toggle("Include games already started", False, key="props_started",
                                  help="Off: only props for games that haven't kicked off.")
            gone = sorted(b[research.started(b.commence).values].game.unique()) if "commence" in b else []
            if gone and not p_started:
                st.caption(f"Hidden — already kicked off: {', '.join(gone)}.")
            view = research.props_view(b, pick_g, pick_b, pick_s, books=S.my_books, include_started=p_started)
            if not view.empty:
                if pick_f.startswith("Only where model is higher"):
                    view = view[view.Difference > 0]
                elif pick_f.startswith("Only where model disagrees"):
                    view = view[view.Lean.str.contains("Over|Under|Yes")]
            if view.empty:
                st.info("No props match these filters.")
            else:
                sel = st.dataframe(view, hide_index=True, width="stretch", key="props_tbl", on_select="rerun",
                                   selection_mode="multi-row", column_config={
                    "Odds": st.column_config.NumberColumn("Odds", format="%+d", help="Over (or Yes) price — median across books, or this book's."),
                    "Model median": st.column_config.NumberColumn("Model median", format="%.1f",
                                                                  help="The model's most likely result, corrected for how its past forecasts missed."),
                    "Model %": st.column_config.NumberColumn("Model %", format="%.0f%%", help="Model's chance of the Over (or Yes)."),
                    "Book %": st.column_config.NumberColumn("Book %", format="%.0f%%", help="The book's chance of the Over (or Yes) with its margin removed."),
                    "Difference": st.column_config.NumberColumn("Difference", format="%+.1f", help="Model % − Book %, in percentage points."),
                    "Lean": st.column_config.Column("Lean", help="Model differs by 4+ points (★ = 8+). No lean = no bet suggested."),
                    "Bet": st.column_config.Column("Bet", help="The side the model leans to (blank when it doesn't differ enough from the book)."),
                    "Bet odds": st.column_config.NumberColumn("Bet odds", format="%+d",
                                                              help="Best price for that side among your sportsbooks (all-books view) or this book's price."),
                    "Best book": st.column_config.Column("Best book", help="Where that price is."),
                    "Bet slip": st.column_config.LinkColumn("Bet slip", display_text="Slip ↗", help="Opens the book with this bet added to the betslip. You review and place it in the book yourself — BeatNFL never places bets."),
                    "_key": None, "_side": None},
                    height=min(700, 38 + 35 * len(view)))
                picked = view.iloc[sel.selection.rows] if sel and sel.selection.rows else view.iloc[0:0]
                a1, a2 = st.columns([1.3, 3], vertical_alignment="center")
                if a1.button(f"➕ Add {len(picked)} to parlay" if len(picked) else "➕ Add to parlay",
                             disabled=picked.empty, key="props_to_parlay"):
                    from beatnfl import parlay as _pl
                    pl_ = ss.setdefault("parlay", [])
                    for r_ in picked.to_dict("records"):       # (itertuples renames columns that start with "_")
                        leg_ = _pl.prop_leg(b, r_["_key"], r_["_side"] or "Over")
                        if leg_ and leg_["id"] not in {x["id"] for x in pl_}:
                            pl_.append(leg_)
                    st.success(f"Parlay has {len(pl_)} legs — open the **Parlay** tab.")
                a2.caption("Tick rows (left edge) to add them to a parlay. Props without a lean are added as Over — "
                           "switch the side on the Parlay tab.")
            st.caption("\\* Anytime TD is quoted on one side only, so its Book % still includes the margin and runs high. "
                       "**no forecast** = player ruled out or too little history. Click any column header to sort. "
                       "Information only.")

    st.subheader("BettingPros")
    if not S.bettingpros_api_key:
        st.caption("Add your BettingPros API key in Settings to see their prop projections and star ratings here.")
    else:
        from beatnfl import bp_record as _bpr
        from beatnfl.sources import bettingpros as _bps
        st.caption(f"Their projection, recommended side and 1–5★ rating for every prop on their board. Each load uses "
                   f"a few requests (you have **{_bps.remaining()}** left today) and is saved as a pre-game snapshot for "
                   "the track record in Evidence. Whether their stars actually win: Evidence → BettingPros track record.")
        if st.button("Load BettingPros props"):
            with st.spinner("Pulling BettingPros' board…"):
                try:
                    bdf, binfo = _bps.props(S.bettingpros_api_key, season, week, completed=False, offline=offline)
                    ss["bp"], ss["bp_week"] = bdf, (season, week)
                    n_s = _bpr.save_snapshot(bdf)
                    st.success(f"{len(bdf)} props ({binfo['pages']} pages, {binfo['requests_left']} requests left today)"
                               + (f" · saved for the track record" if n_s else ""))
                except StepError as e:
                    st.error(e.as_text())
        bdf = ss.get("bp")
        if bdf is not None and not bdf.empty and ss.get("bp_week") == (season, week):
            v = bdf.copy()
            fc_ = ss.get("fc")
            col_ = {"pass_yds": "p_pass_yds", "rush_yds": "p_rush_yds", "rec_yds": "p_rec_yds", "rec": "p_rec",
                    "pass_att": "p_pass_att", "rush_att": "p_rush_att", "pass_cmp": "p_pass_cmp"}
            v["model"] = np.nan
            if fc_ is not None and not fc_.empty:
                from beatnfl.names import norm_name as _nn
                f_ = fc_.assign(nn=fc_.name.map(_nn)).drop_duplicates("nn").set_index("nn")
                for st_, c_ in col_.items():
                    if c_ in f_:
                        m_ = v.stat == st_
                        v.loc[m_, "model"] = v.loc[m_, "nn"].map(f_[c_])
            v["agree"] = np.where(v.model.isna() | v.bp_side.isna() | v.line.isna(), "",
                                  np.where((v.model > v.line) == (v.bp_side == "over"), "agrees", "disagrees"))
            v["best"] = [(f"{int(o):+d}" if pd.notna(o) else "") for o in v.rec_odds]
            k1, k2, k3 = st.columns([1, 3, 2])
            min_s = k1.selectbox("Stars", ["Any", "3+", "4+", "5"], key="bp_stars")
            mks = k2.multiselect("Markets", sorted(v.market.dropna().unique()), [], placeholder="All markets", key="bp_mk")
            bq = k3.text_input("Find player", "", placeholder="Name…", key="bp_q")
            if min_s != "Any":
                v = v[pd.to_numeric(v.bp_stars, errors="coerce") >= float(min_s[0])]
            if mks:
                v = v[v.market.isin(mks)]
            if bq.strip():
                v = v[v.player.str.contains(bq.strip(), case=False, regex=False)]
            v = v.sort_values(["bp_stars", "bp_ev"], ascending=False, na_position="last")
            out_ = v.rename(columns={"player": "Player", "team": "Team", "market": "Market", "line": "Line",
                                     "bp_proj": "BP projection", "bp_side": "BP side", "bp_stars": "Stars",
                                     "bp_prob": "BP win %", "bp_ev": "BP EV %", "best": "Best odds",
                                     "model": "Our model (avg)", "agree": "Our model"})
            out_["BP win %"] = pd.to_numeric(out_["BP win %"], errors="coerce") * 100
            out_["BP EV %"] = pd.to_numeric(out_["BP EV %"], errors="coerce") * 100
            cols_ = ["Player", "Team", "Market", "Line", "BP projection", "BP side", "Stars", "BP win %", "BP EV %",
                     "Best odds", "Our model (avg)", "Our model"]
            show(out_[cols_], {
                "BP projection": st.column_config.NumberColumn("BP projection", format="%.1f"),
                "Stars": st.column_config.NumberColumn("Stars", format="%d★", help="BettingPros' 1–5 rating (Plus/Elite)."),
                "BP win %": st.column_config.NumberColumn("BP win %", format="%.0f%%", help="Their chance the recommended side wins."),
                "BP EV %": st.column_config.NumberColumn("BP EV %", format="%+.1f%%", help="Their expected return at the best listed price."),
                "Best odds": st.column_config.Column("Best odds", help="Best price for their side across the books they track — "
                                                     "not every book is available in your state."),
                "Our model (avg)": st.column_config.NumberColumn("Our model (avg)", format="%.1f",
                                                                 help="BeatNFL's average forecast (Forecast every player first)."),
                "Our model": st.column_config.Column("Our model", help="Whether our forecast is on the same side of the line.")},
                 height=min(700, 38 + 35 * len(out_)))
            if pd.to_numeric(bdf.bp_stars, errors="coerce").isna().all():
                st.caption("No star ratings in this response — ratings, win % and EV come with a Plus or Elite membership.")
            st.caption("Information only. Their record so far: Evidence → BettingPros track record.")

# ------------------------------------------------------------------ PICKS
with tab_picks:
    from beatnfl import research as _rs
    st.warning("**Experimental — no proven edge.** The 2025 backtest on real sportsbook lines found the model adds no "
               "reliable information beyond the books' prices. The one hint was **Unders where the model is 10+ points "
               "below the market** (2025: 53.8% won, ROI +2.1% ± 4.6 — could be luck). Picks logs exactly that rule so "
               "it can be tested on games it has never seen. Research only — don't treat these as bets.")
    st.caption("Rule being tested: Under, model 10+ points below the market, FantasyPros agrees (when your key is set), "
               "best price across books. Each pick is logged with that price and graded on the result and the closing line.")
    season_k, week_k = int(ss.get("p_season")), int(ss.get("p_week"))
    c1, c2 = st.columns([1.4, 3], vertical_alignment="center")
    find = c1.button("🔎 Find this week's picks", type="primary", width="stretch", disabled=not S.odds_api_key)
    c2.caption(f"{season_k} week {week_k} (change the week on the Props tab). Uses about 5 Odds API credits per game."
               if S.odds_api_key else "Needs an Odds API key in Settings.")
    if find:
        with st.spinner("Forecasting, pulling props from every book, comparing with FantasyPros…"):
            try:
                ss["picks"], ss["picks_info"] = _rs.build_midweek(S, season_k, week_k, offline=offline)
                ss.pop("pick_grades", None)
            except StepError as e:
                st.error(e.as_text())
    pk_ = ss.get("picks")
    if pk_ is not None:
        inf = ss.get("picks_info", {})
        if pk_.empty:
            st.info(f"No props met the bar for week {inf.get('week')} ({inf.get('fp_note')}). Books post most props "
                    "2–4 days before kickoff; try again Wednesday or Thursday.")
        else:
            st.success(f"{len(pk_)} props flagged · {inf.get('new', 0)} newly logged · {inf.get('fp_note')}")
            gone_ = _rs.started(pk_.commence).values if "commence" in pk_ else np.zeros(len(pk_), bool)
            if gone_.any():
                st.caption(f"{int(gone_.sum())} of these are for games that already kicked off — hidden here, still graded "
                           "in the track record.")
            pk_ = pk_[~gone_]
            show(_rs.picks_view(pk_), {
                "Best odds": st.column_config.NumberColumn("Best odds", format="%+d", help="Best price for this side among your sportsbooks (Settings → My sportsbooks)."),
                "Market %": st.column_config.NumberColumn("Market %", format="%.0f%%", help="All books' consensus chance for this side, margin removed."),
                "Model %": st.column_config.NumberColumn("Model %", format="%.0f%%", help="BeatNFL's chance for this side."),
                "FantasyPros": st.column_config.NumberColumn("FantasyPros", format="%.1f", help="FantasyPros' projection for the stat (TD: chance 0–1)."),
                "Model median": st.column_config.NumberColumn("Model median", format="%.1f"),
                "Edge at best price %": st.column_config.NumberColumn("Edge at best price %", format="%+.1f",
                    help="What the best price is worth against the market's own fair chance. Positive = this book pays more than the market thinks it should."),
                "Bet slip": st.column_config.LinkColumn("Bet slip", display_text="Slip ↗", help="Opens the book with the best price with this bet added to the betslip. "
                                                     "You review and place it in the book yourself — BeatNFL never places bets.")},
                 height=min(700, 38 + 35 * len(pk_)))

    st.subheader("Track record")
    st.caption("**CLV (closing-line value)** = how much better your logged price was than the market's last price before "
               "kickoff, in percentage points. Consistently positive CLV is the sign of a real edge — it shows up within "
               "weeks, long before win/loss records mean anything.")
    g1, g2, g3 = st.columns([1.2, 1.6, 2.2], vertical_alignment="bottom")
    if g1.button("Grade picks", width="stretch"):
        with st.spinner("Grading…"):
            ss["pick_grades"] = _rs.grade_picks(offline=offline)
    hrs = g3.number_input("Games starting within (hours)", 1, 96, 4)
    if g2.button("Record closing prices now", width="stretch", disabled=not S.odds_api_key,
                 help="Pulls current prices for logged picks whose games start within the hours set. "
                      "The automatic Mac check does this before each kickoff window."):
        try:
            r_ = _rs.capture_close(S.odds_api_key, within_h=float(hrs), offline=offline)
            st.success(f"Recorded prices for {r_['games']} game(s) ({r_['credits']} credits).")
        except StepError as e:
            st.error(e.as_text())
    gr = ss.get("pick_grades")
    if gr:
        tbl, sm = gr
        if not sm.get("n"):
            st.info("No picks logged yet — click **Find this week's picks**.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Logged", sm["n"], help="Picks logged so far")
            m2.metric("Record", f"{sm['won']}-{sm['lost']}-{sm['push']}" if sm["graded"] else "—",
                      help="Won-lost-push for finished games")
            m3.metric("Units", f"{sm['units']:+.2f}" if sm["graded"] else "—",
                      help="Profit at the logged price, 1 unit per pick" + (f" · ROI {sm['roi']:+.1f}%" if sm.get("roi") is not None else ""))
            m4.metric("Avg CLV", f"{sm['clv_avg']:+.1f} pts" if sm.get("clv_avg") is not None else "—",
                      help=(f"Beat the close {sm['beat_close']:.0f}% of the time ({sm['clv_n']} picks with a closing price)"
                            if sm.get("clv_n") else "No closing prices recorded yet"))
            if sm.get("props"):
                st.caption(f"**One per prop** (alternate lines of the same player/stat counted once): {sm['props_won']}-"
                           f"{sm['props'] - sm['props_won']}, ROI {sm['props_roi']:+.1f}% ± {sm['props_roi_se'] or 0:.1f}. "
                           "Same-game Unders win and lose together, so a couple of weeks is still mostly luck — the 2025 "
                           "full-season test of this rule was +2.1% ± 4.6.")
            v = pd.DataFrame({
                "Week": tbl.week, "Game": tbl.game, "Player": tbl.player,
                "Prop": [("Anytime TD" if s_ == "any_td" else f"{_rs.STAT_NAMES.get(s_, s_)} {l_:g}") for s_, l_ in zip(tbl.stat, tbl.line)],
                "Side": tbl.side, "Odds": tbl.price.round(), "Book": tbl.book.map(_rs.book_name),
                "Taken %": (tbl.taken_p * 100).round(1), "Close %": (tbl.close_p * 100).round(1), "CLV": tbl.clv,
                "Actual": tbl.actual, "Result": tbl.result, "Units": tbl.units.round(2)})
            show(v, {"Odds": st.column_config.NumberColumn("Odds", format="%+d"),
                     "Taken %": st.column_config.NumberColumn("Taken %", format="%.1f%%", help="Chance implied by the logged price"),
                     "Close %": st.column_config.NumberColumn("Close %", format="%.1f%%", help="Market's fair chance at the close"),
                     "CLV": st.column_config.NumberColumn("CLV", format="%+.1f", help="Close % − Taken %. Positive = beat the close."),
                     "Units": st.column_config.NumberColumn("Units", format="%+.2f")})

    st.caption("Automatic checks (Wednesday picks, game-day lineup check, closing prices) are in the sidebar.")

# ------------------------------------------------------------------ LOAD DATA
with tab_load:
    st.markdown("Choose where this week's data comes from, then click **⟳ Refresh** (top right).")
    st.radio("Slate source", ["Automatic (DraftKings)", "Upload DKSalaries.csv"], horizontal=True, key="slate_mode")
    if not str(ss.get("slate_mode", "Automatic")).startswith("Upload"):
        from beatnfl.sources import dk
        if "slates" not in ss and not offline:
            try:
                ss["slates"], ss["slates_err"] = dk.discover(offline=offline), None
            except StepError as e:
                ss["slates"], ss["slates_err"] = [], e.as_text()
        if ss.get("slates"):
            main = dk.pick_main_slate(ss["slates"])
            labels = [s_.label for s_ in ss["slates"]]
            if ss.get("slate_pick") not in labels:
                ss["slate_pick"] = main.label
            st.selectbox("Slate", labels, key="slate_pick",
                         help="All NFL Classic slates DraftKings has open. The main slate is picked by default.")
        else:
            if ss.get("slates_err"):
                st.warning("Couldn't list DraftKings slates right now — the main slate will still be tried on Refresh, "
                           "then the newest DKSalaries*.csv in your inbox.\n\n" + ss["slates_err"])
            if st.button("Retry DraftKings slate list"):
                ss.pop("slates", None)
                st.rerun()
        st.caption("If DraftKings blocks the download, the newest DKSalaries*.csv in the inbox is used automatically.")
    else:
        st.file_uploader("DKSalaries.csv (DraftKings → NFL Classic contest → Export to CSV)", type="csv", key="up_file")
    if S.fantasypros_api_key:
        st.caption("✅ FantasyPros API key set — your projections load automatically on Refresh. "
                   "A table pasted or uploaded below replaces them for that refresh.")
    st.file_uploader("Your projections CSV (optional — newest file in the inbox is used otherwise)", type="csv", key="paid_file")
    st.text_area("…or paste a projections table (e.g. copy FantasyPros' table in your browser, paste here)",
                 height=90, key="pasted_proj")
    st.checkbox("Include in-house model", value=True, key="use_model")
    r = ss.get("run")
    st.subheader("Last refresh")
    if r:
        show_steps(r.steps)
    else:
        st.caption("Nothing loaded yet.")


# ------------------------------------------------------------------ MY YAHOO TEAM
with tab_yahoo:
    from beatnfl.sources import yahoo
    st.caption("Read-only: shows your Yahoo roster with BeatNFL projections and start/sit suggestions. "
               "It never changes your lineup on Yahoo.")
    if not (S.yahoo_client_id and S.yahoo_client_secret):
        st.markdown(
            "**One-time setup**\n\n"
            "1. Go to [developer.yahoo.com/apps/create](https://developer.yahoo.com/apps/create/) and sign in.\n"
            "2. Redirect URI **https://localhost:8000**, API permissions **Fantasy Sports → Read**. Create the app.\n"
            "3. Copy the **Client ID** and **Client Secret** into *Settings* in the sidebar and click **Save settings**.")
    elif not yahoo.connected():
        st.markdown("**Connect your Yahoo account**")
        ru = st.text_input("Redirect URI — must match your Yahoo app exactly", S.yahoo_redirect_uri, key="yahoo_ru",
                           help="Open developer.yahoo.com/apps → your app and copy its Redirect URI here. If Yahoo says "
                                "'Something went wrong', these two don't match (or the Client ID is wrong).")
        if ru.strip() != S.yahoo_redirect_uri:
            S.yahoo_redirect_uri = ru.strip() or yahoo.DEFAULT_REDIRECT
            S.save()
        try:
            catch = yahoo.start_catcher(S.yahoo_client_id, S.yahoo_client_secret, S.yahoo_redirect_uri)
        except StepError as e:
            catch = ""
            st.caption(e.what)
        st.link_button("1 · Sign in to Yahoo and approve", yahoo.auth_url(S.yahoo_client_id, S.yahoo_redirect_uri))
        if catch == "connected" or yahoo.connected():
            st.success("Connected to Yahoo.")
            st.rerun()
        if catch:
            st.caption("BeatNFL finishes the sign-in by itself. After Yahoo, your browser may warn that the "
                       "**connection isn't private** — that's BeatNFL's own page on your computer: choose **Show details → "
                       "visit this website** (Safari) or **Advanced → Proceed to localhost** (Chrome). Then come back and "
                       "click **Check connection**.")
            if catch.startswith("error"):
                st.error(catch[7:] + " — check that the Redirect URI above matches your Yahoo app exactly, then click the "
                         "sign-in button again.")
            elif catch.startswith("port"):
                st.warning(f"Can't listen for Yahoo ({catch}). Use the copy-the-address method below.")
            st.button("Check connection", key="yahoo_check")
        else:
            st.caption("Click the button above. After you approve on Yahoo it " +
                       ("shows a code — paste it below." if S.yahoo_redirect_uri == "oob" else
                        "opens a page whose address starts with `localhost:8000/?code=` — copy the whole address."))
        code = st.text_input("2 · Or paste the code / the whole address here", key="yahoo_code")
        if st.button("Connect", disabled=not code.strip()):
            try:
                yahoo.connect(S.yahoo_client_id, S.yahoo_client_secret, code, S.yahoo_redirect_uri)
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
                    show(ytbl, cfg)
                    model_slots = [(s_, n) for s_, n in slots if s_ not in ("K", "DEF")]
                    sug = yahoo.suggest(rows, model_slots)
                    cur = {r["player_key"] for r in rows if r["slot"] not in ("BN", "IR", "IR+", "NA", "K", "DEF")}
                    best = {s_["player_key"] for s_ in sug if s_["player_key"]}
                    st.markdown("**Suggested starters** (kickers and defenses not projected — keep your choice)")
                    sg, sgc = tidy(pd.DataFrame(sug), {"slot": "Slot", "player": "Start", "proj": "Proj pts"}, one=("Proj pts",))
                    sg["Start"] = sg["Start"].replace({"— nobody projected —": "— no healthy, projected player —"})
                    show(sg, sgc)
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
            ss.pop("yteams", None)
            if st.button("Disconnect Yahoo", key="ydisc_err", help="Forget this sign-in so you can sign in again."):
                yahoo.disconnect()
                st.rerun()

# ------------------------------------------------------------------ EVIDENCE
with tab_evidence:
    st.subheader("Weekly review")
    st.caption("How last week's lineups, prop picks, game numbers and BettingPros ratings landed — one place, graded "
               "on the box scores. One week is mostly luck: the model is changed only when the multi-season tests "
               "below agree.")
    from beatnfl import review as _rv
    from beatnfl.sources import nflverse as _nvr
    try:
        _cs, _cw = _nvr.upcoming_week(_nvr.schedules(offline=offline))
    except StepError:
        _cs, _cw = 2026, 2
    r1, r2, r3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    rv_season = r1.number_input("Season", 2020, 2100, int(_cs), key="rv_season")
    rv_week = r2.number_input("Week", 1, 22, max(1, int(_cw) - 1), key="rv_week",
                              help="Defaults to last week. Box scores post the morning after the games.")
    if r3.button("📋 Review this week", type="primary"):
        with st.spinner("Grading everything for that week…"):
            ss["review"] = _rv.week_review(int(rv_season), int(rv_week), offline=offline)
    rv = ss.get("review")
    if rv:
        st.caption(f"{rv['season']} week {rv['week']} · {rv.get('games_final', '')}")
        d_ = rv.get("dfs", {})
        st.markdown("**DFS lineups**")
        if isinstance(d_.get("sets"), pd.DataFrame) and not d_["sets"].empty:
            show(d_["sets"])
            if isinstance(d_.get("sources"), pd.DataFrame) and not d_["sources"].empty:
                st.caption("The same lineups rebuilt from each projection source alone, scored on the real results:")
                show(d_["sources"].rename(columns={"source": "Projections used", "avg_projected": "Avg projected",
                                                   "avg_actual": "Avg actual", "players_projected": "Players covered"}))
        else:
            st.caption(d_.get("note") or d_.get("error", ""))
        p_ = rv.get("picks", {})
        st.markdown("**Prop picks**")
        if p_.get("summary"):
            show(pd.DataFrame([{"count": k, **v} for k, v in p_["summary"].items()]))
            if isinstance(p_.get("by_stat"), pd.DataFrame) and not p_["by_stat"].empty:
                show(p_["by_stat"])
        else:
            st.caption(p_.get("error") or "No graded picks for this week.")
        g_ = rv.get("games", {})
        st.markdown("**Game model vs the closing line** (lower miss is better)")
        if g_.get("n"):
            show(pd.DataFrame([{"measure": k, "value": str(v)} for k, v in g_.items() if k != "n"]))
        else:
            st.caption(g_.get("error") or "No finished games.")
        b_ = rv.get("bettingpros", {})
        st.markdown("**BettingPros ratings** (your pre-game snapshots)")
        st.caption(str(b_.get("verdict", b_.get("error", ""))))
        if isinstance(b_.get("by_stars"), pd.DataFrame) and not b_["by_stars"].empty:
            show(b_["by_stars"])
        with st.expander("Copy as text"):
            st.code(_rv.as_text(rv), language=None)
    st.divider()
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
        t, tc = tidy(pl[pl.season.astype(str) == "all"], {"pos": "Pos", "n": "Player-games", "mae_model": "Model",
                    "mae_ewm": "Recent form", "mae_naive": "Season average", "bias_model": "Model bias"},
                    ints=("Player-games",), two=("Model", "Recent form", "Season average", "Model bias"))
        show(t, tc)
        st.markdown("**Decision test — each week's top-N picks by method, average actual DK points**")
        t, tc = tidy(pd.DataFrame(ev["picks"]), {"pos": "Pos", "top_n": "Top N", "weeks": "Weeks", "avg_pts_model": "Model picks",
                    "avg_pts_naive": "Season-avg picks", "avg_pts_ewm": "Recent-form picks", "avg_pts_oracle": "Perfect hindsight",
                    "model_beats_naive_weeks": "Model better in"},
                    ints=("Top N", "Weeks"), one=("Model picks", "Season-avg picks", "Recent-form picks", "Perfect hindsight"))
        t["Model better in"] = t["Model better in"].astype(str) + " of weeks"
        show(t, tc)
        st.markdown("**Prop stats** — lines in this test come from recent form, not sportsbooks; real lines are graded as you use Props")
        pr = pd.DataFrame(ev["props"])
        td = pr[pr.stat == "any_td"]
        pr = pr[pr.stat != "any_td"].assign(stat=lambda d: d.stat.map(lambda x: {"pass_yds": "Pass yds", "pass_att": "Pass att",
                 "pass_cmp": "Completions", "rush_yds": "Rush yds", "rush_att": "Rush att", "rec": "Receptions", "rec_yds": "Rec yds"}.get(x, x)))
        t, tc = tidy(pr, {"stat": "Stat", "n": "Games", "mae_model": "Miss: model", "mae_baseline": "Miss: recent form",
                    "brier_model": "Prob. score: model", "brier_coinflip": "Prob. score: 50/50", "confident_hit_rate": "Confident calls right"},
                    ints=("Games",), two=("Miss: model", "Miss: recent form"), pct=("Confident calls right",))
        tc.update({c_: st.column_config.NumberColumn(c_, format="%.3f") for c_ in ("Prob. score: model", "Prob. score: 50/50")})
        show(t, tc)
        if len(td):
            st.caption(f"Anytime TD probability score: model {td.brier_model.iloc[0]:.3f} vs recent-form {td.brier_baseline.iloc[0]:.3f} "
                       "(lower is better). Probability score = Brier score.")
        st.markdown("**Defenses (DST)**")
        t, tc = tidy(pd.DataFrame(ev["dst"]), {"season": "Season", "n": "Games", "mae_model": "Miss: model", "mae_ewm": "Miss: recent form",
                    "rank_corr_model": "Ranking accuracy: model", "rank_corr_ewm": "Ranking accuracy: recent form"},
                    two=("Miss: model", "Miss: recent form", "Ranking accuracy: model", "Ranking accuracy: recent form"))
        t["Season"] = t["Season"].astype(str)
        show(t, tc)
        g = ev["games"]
        st.markdown(f"**Games vs closing line** — {g['n_games']:,} games, {g['seasons']}. Break-even at -110 odds is 52.4%.")
        c1, c2 = st.columns(2)
        for col, key, lab in ((c1, "ats", "Spread"), (c2, "totals", "Total")):
            t = pd.DataFrame(g[key]).assign(edge=lambda d: d.edge_pts.map(lambda x: f"{x}+ pts"))
            t, tc = tidy(t, {"edge": f"{lab}: model differs by", "n": "Games", "win_rate": "Model side won"}, ints=("Games",),
                        pct=("Model side won",))
            with col:
                show(t, tc)
        st.caption(f"Average miss on the final margin: market {g['mae_margin']['spread_line']:.2f}, model "
                   f"{g['mae_margin']['model_spread']:.2f} points. On the total: market {g['mae_total']['total_line']:.2f}, "
                   f"model {g['mae_total']['model_total']:.2f}. The market is more accurate.")
    st.markdown("**BettingPros track record — do their star ratings win at real prices?**")
    if not S.bettingpros_api_key:
        st.caption("Add your BettingPros API key in Settings to build this.")
    else:
        from beatnfl import bp_record as _bpr
        from beatnfl.sources import bettingpros as _bps
        st.caption("Used for props only (not DraftKings projections). Two records: **their archive** (finished weeks from BettingPros, ratings taken at face value) and "
                   "**your snapshots** (their board as the app saved it before kickoff — can't be revised later). Graded at "
                   "the best listed price and at the consensus price; break-even at -110 is 52.4%. Requests are limited per "
                   f"day (**{_bps.remaining()}** left today), so the archive grows a few weeks per day — the evening "
                   "automatic check does this for you.")
        if st.button("Collect finished weeks now (uses today's spare requests)"):
            with st.spinner("Downloading finished weeks and grading…"):
                try:
                    r_bp = _bpr.daily(S.bettingpros_api_key, offline=offline)
                    st.success(f"{r_bp['collect']['weeks_done']} finished weeks collected "
                               f"({len(r_bp['collect']['fetched_now'])} new) · {r_bp['collect']['requests_left']} requests left today"
                               + (" · stopped at today's limit — continues tomorrow" if r_bp["collect"]["stopped"] else ""))
                except StepError as e:
                    st.error(e.as_text())
        for nm_, lab_ in (("snapshots", "Your pre-game snapshots"), ("archive", "BettingPros' archive")):
            res_ = _bpr.load_result(nm_)
            if not res_:
                continue
            st.markdown(f"*{lab_}* (graded {res_.get('generated', '')}): {res_['verdict']}")
            inf_ = res_.get("info") or {}
            if inf_.get("n"):
                st.markdown(f"Does it know more than the line? **{inf_['verdict']}**")
            cfg_ = {"won_pct": "Won %", "roi_best_pct": "ROI at best price %", "roi_best_se": "± (luck)",
                    "roi_cons_pct": "ROI at consensus %", "roi_cons_se": "± (luck) ",
                    "their_win_prob_pct": "Their win % claim", "picks": "Picks"}
            if not res_["by_stars"].empty:
                t, tc = tidy(res_["by_stars"], {"stars": "Stars", **cfg_}, ints=("Stars", "Picks"),
                             one=("Won %", "ROI at best price %", "± (luck)", "ROI at consensus %", "± (luck) ", "Their win % claim"))
                show(t, tc)
            if not res_["by_market"].empty:
                with st.expander(f"{lab_} by market and by model"):
                    t, tc = tidy(res_["by_market"], {"market": "Market", **cfg_}, ints=("Picks",),
                                 one=("Won %", "ROI at best price %", "± (luck)", "ROI at consensus %", "± (luck) ", "Their win % claim"))
                    show(t, tc)
                    t, tc = tidy(res_["by_model"], {"model": "BettingPros model", **cfg_}, ints=("Picks",),
                                 one=("Won %", "ROI at best price %", "± (luck)", "ROI at consensus %", "± (luck) ", "Their win % claim"))
                    show(t, tc)
                    bs_ = inf_.get("by_stat")
                    if isinstance(bs_, pd.DataFrame) and not bs_.empty:
                        st.caption("Information test by market — accuracy of the Over chance (lower is better) and the "
                                   "weight BettingPros gets once the market price is known (real information ≈ 2× its ±).")
                        t, tc = tidy(bs_, {"market": "Market", "props": "Props", "brier_market": "Market accuracy",
                                           "brier_bp": "BettingPros accuracy", "bp_coef": "BP weight", "bp_se": "±",
                                           "bp_z": "z"}, ints=("Props",), two=("BP weight", "±", "z"))
                        show(t, tc)

    st.markdown("**Props on real sportsbook lines — does the Picks rule hold up?**")
    st.caption("Replays a past season with the model as it stood before that season: each prop's real Wednesday-evening "
               "price and closing price from every US book (Odds API historical data, paid plans). Picks = model differs "
               "from the midweek market by the threshold, taken at the best midweek price, graded on result and on "
               "closing-line value. Information only.")
    from beatnfl import props_backtest as _pb
    b1, b2, b3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    bt_season = b1.selectbox("Season", [2025, 2024, 2023], key="bt_season")
    bt_res = b2.number_input("Keep at least (credits)", 0, 1_000_000, 4000, 500, key="bt_reserve",
                             help="The test stops before your Odds API credits drop below this, so weekly use is never starved. "
                                  "Run it again after credits renew — finished parts are never paid for twice.")
    try:
        est = _pb.estimate(int(bt_season), offline=offline)
        _q = _oq()
        b3.caption(f"{est['games']} games · up to ~{est['credits_max']:,} credits (only markets books actually posted are charged)"
                   + (f" · you have {int(_q['remaining']):,}" if _q.get("remaining") is not None else ""))
    except StepError:
        pass
    if st.button("Run the props backtest", disabled=not S.odds_api_key):
        with st.spinner("Rebuilding the model for that season and pulling historical prices (≈10–20 min)…"):
            try:
                _pb.run(S.odds_api_key, int(bt_season), reserve=float(bt_res), offline=offline)
            except StepError as e:
                st.error(e.as_text())
    pbf = sorted(glob.glob(str(_H / "evidence" / "props_backtest_*.json")))
    for f_ in pbf[::-1]:
        r_ = _pb.load_result(f_)
        if not r_.get("n"):
            st.info(f"{r_.get('season')}: {r_.get('note', 'no data')}")
            continue
        head = (f"**{r_['season']}** · {r_['n']:,} props (one per player-game; alternate lines not double-counted), weeks "
                f"{min(r_['weeks'])}–{max(r_['weeks'])}" if r_.get("weeks") else f"**{r_['season']}**")
        if r_.get("status") == "paused":
            head += " · ⏸ paused by the credit limit — run again after credits renew to finish the season"
        st.markdown(head)
        mi, br = r_.get("model_info", {}), r_.get("brier", {})
        if mi:
            verdict = ("the model adds real information beyond the sportsbook price" if mi["z"] >= 2 else
                       "the model adds **no proven information** beyond the sportsbook price")
            st.markdown(f"Verdict: {verdict} (weight {mi['coef']:+.2f} ± {mi['se']:.2f}; real information needs about 2× its ± or more).")
        st.caption("Accuracy of the Over chance (Brier, lower is better; 0.2500 = calling everything 50/50): model "
                   f"{br.get('model', float('nan')):.4f} · market midweek {br.get('market_midweek', float('nan')):.4f}"
                   + (f" · market at close {br['market_close']:.4f}" if "market_close" in br else ""))
        pt = pd.DataFrame([{**{k: v for k, v in x.items() if k not in ("by_stat", "sides")},
                            "roi_under": (x.get("sides", {}).get("under") or {}).get("roi"),
                            "roi_over": (x.get("sides", {}).get("over") or {}).get("roi")} for x in r_["picks"]])
        t, tc = tidy(pt, {"threshold": "Model differs by (pts)", "n": "Picks", "win_rate": "Win %", "roi": "ROI %",
                          "roi_se": "± (luck)", "roi_under": "ROI Unders %", "roi_over": "ROI Overs %",
                          "clv_avg": "Avg CLV (pts)", "moved_toward": "Line moved toward %", "moved_away": "Line moved away %"},
                     ints=("Model differs by (pts)", "Picks"),
                     one=("Win %", "ROI %", "± (luck)", "ROI Unders %", "ROI Overs %", "Line moved toward %", "Line moved away %"),
                     two=("Avg CLV (pts)",))
        show(t, tc)
    st.caption("ROI within its ± is indistinguishable from luck. Avg CLV compares your price (with the book's margin) to "
               "the fair closing chance; 'Line moved toward' = how often the market moved to the pick's side by kickoff. "
               "Break-even at -110 is 52.4%.")

    st.markdown("**FantasyPros vs Sleeper vs model — which to trust, by position**")
    st.caption("Scores each source's published weekly projections against what actually happened (the model is "
               "trained only on earlier seasons), then finds the blend with the smallest miss for each position — and, "
               "for props, how much of FantasyPros' stat lines to mix into the model's stat forecasts.")
    if st.button("Run the source test (≈10–15 min)"):
        from beatnfl import backtest
        with st.spinner("Downloading past projections and testing…"):
            try:
                backtest.source_compare(S.fantasypros_api_key, current_weights={"paid": S.w_paid, "sleeper": S.w_sleeper,
                                                                                 "model": S.w_model}, offline=offline,
                                        progress=lambda m: None)
            except StepError as e:
                st.error(e.as_text())
    scf = sorted(glob.glob(str(_H / "evidence" / "source_compare_*.json")))
    if scf:
        c = json.loads(Path(scf[-1]).read_text())
        for n_ in c.get("notes", []):
            st.info(n_)
        if c.get("by_pos"):
            st.markdown(f"{c['n']:,} player-games, seasons {', '.join(map(str, c['seasons']))} · sources: "
                        f"{', '.join(c['sources'])} · FantasyPros weeks: {c.get('weeks_fantasypros', 0)}")
            ren = {"pos": "Pos", "n": "Player-games", "mae_paid": "Miss: FantasyPros", "mae_sleeper": "Miss: Sleeper",
                   "mae_model": "Miss: model", "mae_current_blend": "Miss: your weights", "mae_best_blend": "Miss: best blend",
                   "w_paid": "FantasyPros wt", "w_sleeper": "Sleeper wt", "w_model": "Model wt"}
            t, tc = tidy(pd.DataFrame(c["by_pos"]), ren, ints=("Player-games",),
                         two=tuple(v for k, v in ren.items() if k.startswith("mae")), pct=("FantasyPros wt", "Sleeper wt", "Model wt"))
            show(t, tc)
            if c.get("stats"):
                st.markdown("**Props: FantasyPros' stat lines vs the model's** (average miss; share = best mix)")
                t, tc = tidy(pd.DataFrame(c["stats"]), {"stat": "Stat", "n": "Player-games", "mae_model": "Miss: model",
                             "mae_fantasypros": "Miss: FantasyPros", "mae_blend": "Miss: mix", "w_fantasypros": "FantasyPros share"},
                             ints=("Player-games",), two=("Miss: model", "Miss: FantasyPros", "Miss: mix"), pct=("FantasyPros share",))
                show(t, tc)
            u1, u2, _ = st.columns([1.3, 1.3, 3])
            if u1.button("Use these weights", type="primary"):
                S.pos_weights = c.get("weights", {})
                S.stat_weights = c.get("stat_weights", {})
                S.save()
                st.success("Saved. Combined now uses these weights per position; Props mixes in FantasyPros' stats "
                           "where it tested better. Click ⟳ Refresh to apply.")
            if (S.pos_weights or S.stat_weights) and u2.button("Back to simple weights"):
                S.pos_weights, S.stat_weights = {}, {}
                S.save()
                st.success("Back to the single weights in Settings. Click ⟳ Refresh to apply.")
    elif S.pos_weights:
        st.caption("Tested per-position weights are in use.")
    with st.expander("Live check — your own frozen pregame projections, graded after the games"):
        if st.button("Grade live weights"):
            from beatnfl import backtest
            lw = backtest.live_weights(offline=offline)
            if lw.get("weeks", 0) < 3:
                st.info(f"{lw.get('weeks', 0)} finished week(s) graded so far — weights are suggested after 3. "
                        "Each Refresh before kickoff freezes the numbers for this.")
            else:
                t, tc = tidy(pd.DataFrame(lw["by_pos"]), {"pos": "Pos", "n": "Player-games", "mae_paid": "Miss: FantasyPros",
                             "mae_sleeper": "Miss: Sleeper", "mae_model": "Miss: model", "mae_best_blend": "Miss: best blend",
                             "w_paid": "FantasyPros wt", "w_sleeper": "Sleeper wt", "w_model": "Model wt"},
                             ints=("Player-games",), two=("Miss: FantasyPros", "Miss: Sleeper", "Miss: model", "Miss: best blend"),
                             pct=("FantasyPros wt", "Sleeper wt", "Model wt"))
                show(t, tc)
                ss["live_w"] = lw["weights"]
        if ss.get("live_w") and st.button("Use live weights"):
            S.pos_weights = ss["live_w"]
            S.save()
            st.success("Saved. Click ⟳ Refresh to apply.")
    blt = APP_ROOT / "evidence" / "book_lines_test.json"
    if blt.exists():
        m_ = json.loads(blt.read_text())
        st.markdown(f"**Sportsbook lines in the model** — {m_['verdict']}")
        st.dataframe(pd.DataFrame(m_["rows"]), hide_index=True)
        st.caption(m_["note"] + " For DK points the 'book line' column is the model with the lines folded in.")
    mvt = APP_ROOT / "evidence" / "model_variants.json"
    if mvt.exists():
        m_ = json.loads(mvt.read_text())
        st.markdown(f"**Model settings tested** — {m_['verdict']}")
        st.dataframe(pd.DataFrame(m_["rows"]), hide_index=True)
        st.dataframe(pd.DataFrame(m_["stats"]), hide_index=True)
    rgt = APP_ROOT / "evidence" / "ranges_test.json"
    if rgt.exists():
        m_ = json.loads(rgt.read_text())
        st.markdown(f"**Floor / ceiling ranges** — {m_['verdict']}")
        st.dataframe(pd.DataFrame(m_["rows"]).astype(str), hide_index=True)
    qab = APP_ROOT / "evidence" / "qb_weather_ab_test.json"
    if qab.exists():
        m_ = json.loads(qab.read_text())
        st.markdown(f"**Starting QB and weather as model inputs** — {m_['verdict']}")
        st.dataframe(pd.DataFrame(m_["rows"]), hide_index=True)
    sab = APP_ROOT / "evidence" / "snaps_ab_test.json"
    if sab.exists():
        m_ = json.loads(sab.read_text())
        st.markdown(f"**Snap share as a model input** — {m_['verdict']}")
        st.dataframe(pd.DataFrame(m_["rows"]), hide_index=True)
    iab = APP_ROOT / "evidence" / "injury_ab_test.json"
    if iab.exists():
        m_ = json.loads(iab.read_text())
        st.markdown(f"**Practice reports as model inputs** — {m_['verdict']}")
        st.dataframe(pd.DataFrame(m_["rows"]), hide_index=True)
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
                                          ("odds_api_key", "fantasypros_api_key", "anthropic_api_key", "bettingpros_api_key", "yahoo_client_id", "yahoo_client_secret", "github_token")),
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
        st.info("No runs yet — click ⟳ Refresh (top right).")
    else:
        stt = stt.assign(When=pd.to_datetime(stt.last_attempt_utc, utc=True).dt.tz_convert(TZ).dt.strftime("%a %-I:%M %p"),
                         Result=stt.status.map({"ok": "✅ ok", "warn": "⚠️ check", "skip": "⏭️ skipped", "fail": "❌ failed",
                                                "failed": "❌ failed"}).fillna(stt.status))
        st.dataframe(stt.rename(columns={"source": "Step", "message": "Last message"})[["Step", "Result", "When", "Last message"]],
                     hide_index=True, width="stretch")
    if LOG_PATH.exists():
        with st.expander("Technical log (last 60 lines)"):
            st.code("\n".join(LOG_PATH.read_text().splitlines()[-60:]))


# ------------------------------------------------------------------ ASK
def _ask_games(season, week):
    from beatnfl import research
    raw = raw_lines()
    if raw is not None and not raw.empty and (raw.book == "draftkings").any():
        raw = raw[raw.book == "draftkings"]
    gt = research.games_table(season, week, raw, offline=offline)
    gp = research.game_probs(season, week, raw, offline=offline)
    gt = gt.assign(started=research.started(gt._kick).values).drop(columns=["_src", "_kick"], errors="ignore")
    gp = gp[~research.started(gp._kick).values].drop(columns="_kick", errors="ignore")
    return gt, gp


def _ask_props():
    b = ss.get("board")
    if b is None or b.empty:
        return None
    try:
        from beatnfl import research
        return research.props_view(b)
    except Exception:  # noqa: BLE001
        return b


with tab_ask:
    from beatnfl import ask as _ask
    st.caption("Ask about this week in plain English — answers come from the app's own data (pool, projections, "
               "injuries, lines, props, your saved lineups and grades, Evidence). Information only.")
    if not S.anthropic_api_key:
        st.info("To turn this on: get an API key at **platform.claude.com/settings/keys**, paste it in **Settings → "
                "Anthropic API key** (sidebar) and click **Save settings**. It's billed per question by Anthropic "
                "(each answer shows its cost).")
    hist = ss.setdefault("ask_hist", [])
    st.caption("Try: *Best value RBs under $6K?* · *Why is Bijan projected so high?* · *Build my cash lineup with "
                "Bo Nix locked* · *How did my lineups do last week?* · *Which projection source has been most accurate?*")
    for i, h in enumerate(hist):
        with st.chat_message("user"):
            st.markdown(h["q"])
        with st.chat_message("assistant"):
            st.markdown(h["a"])
            b = h.get("built")
            if b and b["result"].lineups and ss.get("run") is not None:
                st.dataframe(export.lineup_table(b["result"].lineups, ss["run"].pool), hide_index=True, width="stretch")
                if st.button("Use on This week", key=f"ask_use_{i}",
                             help="Puts these lineups in This week's Build area, where you can Save and download the DK CSV."):
                    ss["lineups"], ss["saved_info"] = b, None
                    st.success("Done — open **This week** to save or download them.")
            meta = []
            if h.get("tools"):
                meta.append("looked up: " + ", ".join(dict.fromkeys(t.replace("_", " ") for t in h["tools"])))
            if h.get("cost") is not None:
                meta.append(f"≈ ${h['cost']:.3f}")
            if meta:
                st.caption(" · ".join(meta))
    if hist and st.button("Clear conversation", key="ask_clear"):
        ss["ask_hist"] = []
        st.rerun()
    q_ = st.chat_input("Ask about this week…", key="ask_q", disabled=not S.anthropic_api_key)
    if q_:
        r_ = ss.get("run")
        season_, week_ = ((r_.season, r_.week) if r_ is not None and r_.season else ss.get("current_week", (None, None)))
        ctx_ = _ask.Context(run=r_, season=season_, week=week_, offline=offline,
                            games=lambda: _ask_games(season_, week_), forecasts=lambda: ss.get("fc"), props=_ask_props,
                            bettingpros=lambda: ss.get("bp"))
        with st.status("Thinking…", expanded=False) as stt_:
            try:
                res_ = _ask.ask(S.anthropic_api_key, q_, ctx_, [(h["q"], h["a"]) for h in hist], model=S.ask_model,
                                progress=lambda m: stt_.update(label=m))
                hist.append({"q": q_, "a": res_["answer"], "tools": res_["tools_used"], "cost": res_["cost"],
                             "built": res_["built"]})
                stt_.update(label="Done", state="complete")
            except StepError as e:
                stt_.update(label="Couldn't answer", state="error")
                hist.append({"q": q_, "a": "⚠️ " + e.as_text(), "tools": [], "cost": None, "built": None})
        st.rerun()


# ------------------------------------------------------------------ PARLAY
with tab_parlay:
    from beatnfl import parlay as _pl
    st.caption("Build a parlay from **Props** and **Games**: tick rows there and click **➕ Add to parlay**. Information only — "
               "you place it in the sportsbook yourself.")
    legs = ss.setdefault("parlay", [])
    if not legs:
        st.info("No legs yet. On **Props** or **Games**, tick the rows you want (left edge of the table) and click "
                "**➕ Add to parlay**.")
    else:
        lt = _pl.legs_table(legs)
        lt.insert(0, "Remove", False)
        ed = st.data_editor(lt, hide_index=True, width="stretch", key=f"parlay_ed_{ss.get('parlay_v', 0)}",
                            disabled=["Leg", "Game", "Model %", "Market %", "Books"], column_config={
            "Side": st.column_config.SelectboxColumn("Side", options=["Over", "Under", "Yes", "Bet"],
                                                     help="Switch a prop between Over and Under (anytime TD is Yes only; game legs are fixed)."),
            "Model %": st.column_config.NumberColumn("Model %", format="%.0f%%", help="BeatNFL's chance for this leg."),
            "Market %": st.column_config.NumberColumn("Market %", format="%.0f%%", help="The books' chance with their margin removed."),
            "Books": st.column_config.NumberColumn("Books", help="How many sportsbooks offer this leg at this line.")})
        changed = False
        for lg, r_ in zip(list(legs), ed.to_dict("records")):
            if lg["kind"] == "prop" and lg["stat"] != "any_td" and r_["Side"] in ("Over", "Under") and r_["Side"] != lg["side"]:
                lg["side"], changed = r_["Side"], True
        if ed.Remove.any():
            ss["parlay"] = legs = [lg for lg, rm in zip(legs, ed.Remove) if not rm]
            changed = True
        if changed:
            ss["parlay_v"] = ss.get("parlay_v", 0) + 1
            st.rerun()
        gone_l = _pl.started_legs(legs)
        if gone_l:
            st.error("Already kicked off — books no longer take these before the game (live odds differ): "
                     + "; ".join(gone_l) + ". Tick Remove to drop them.")
        sg = _pl.same_game(legs)
        if sg:
            st.warning("Same-game legs (" + ", ".join(sg) + "): these results move together, and books price same-game "
                       "parlays with their own adjustment — the odds below are a guide; the book's betslip shows the real price.")
        pr = _pl.price(legs, S.my_books)
        if pr.empty:
            st.info("None of your sportsbooks offer these legs.")
        else:
            bp_ = _pl.best(pr)
            if bp_:
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{len(legs)}-leg parlay odds", f"{int(bp_['odds']):+d}")
                m2.metric("$10 pays", f"${bp_['pays10']:,.2f}")
                m3.metric("Best at", " = ".join(bp_["tied"]) if len(bp_["tied"]) <= 3 else bp_["name"])
            else:
                st.warning(f"No single sportsbook of yours offers all {len(legs)} legs, so there's no parlay price yet — "
                           "see **Not offered** below and remove or swap those legs.")
            st.markdown(f"**{len(legs)}-leg parlay at your sportsbooks** (best first)")
            show(pr.drop(columns="book"), {
                "Best": st.column_config.Column("Best", help="Highest payout. Tied books pay exactly the same — use whichever you prefer."),
                "Not offered": st.column_config.Column("Not offered", help="Legs this book doesn't offer — the parlay can't be placed there as is."),
                "Different line": st.column_config.Column("Different line", help="Legs this book hangs at a different line (e.g. 75.5 instead of 74.5) — the price uses that book's line."),
                "Parlay odds": st.column_config.NumberColumn("Parlay odds", format="%+d", help="All legs' odds multiplied (independent legs)."),
                "Pays on $10": st.column_config.NumberColumn("Pays on $10", format="$%.2f", help="Total return on a $10 bet, stake included."),
                "Market chance %": st.column_config.NumberColumn("Market chance %", format="%.1f%%",
                                                                 help="Chance all legs win, from the books' own no-vig prices."),
                "Model chance %": st.column_config.NumberColumn("Model chance %", format="%.1f%%", help="Chance all legs win by BeatNFL's numbers."),
                "Value vs market %": st.column_config.NumberColumn("Value vs market %", format="%+.1f%%",
                                                                   help="Expected return of these odds at the market's own chance. Negative = the book's margin, which grows with every leg."),
                "Value vs model %": st.column_config.NumberColumn("Value vs model %", format="%+.1f%%",
                                                                  help="Expected return at BeatNFL's chances. The model has no proven edge on props or games (Evidence).")})
            full = pr[pr["Parlay odds"].notna()]
            choices = list(full.book) or list(pr.book)
            bk = st.selectbox("Place at", choices, format_func=lambda k: _pl.book_name(k), key="parlay_book",
                              help="Books are listed best payout first; tied books are in your My sportsbooks order.")
            sl = _pl.slips(legs, bk)
            show(sl, {"Odds": st.column_config.NumberColumn("Odds", format="%+d"),
                      "Slip": st.column_config.LinkColumn("Slip", display_text="Add ↗",
                                                          help="Opens the book with this leg added to its betslip.")})
            st.caption(f"Open each leg's **Add ↗** in order — they collect in {_pl.book_name(bk)}'s betslip — then choose "
                       "**Parlay** there, check the price and place it yourself. A blank slip = this book gives no link for "
                       "that leg (or doesn't offer it); add it in the book directly. Each extra leg adds the book's margin "
                       "again, so parlays cost more than the same bets placed singly.")
        if st.button("Clear parlay", key="parlay_clear"):
            ss["parlay"], ss["parlay_v"] = [], ss.get("parlay_v", 0) + 1
            st.rerun()
