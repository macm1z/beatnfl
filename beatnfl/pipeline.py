"""The one weekly workflow. Each step reports ok / warn / fail with a plain-English fix,
and a failure in an optional step never throws away earlier work.

    run_week()  →  slate → injuries → schedule/lines → projections → blend → freeze
    build_lineups() → optimize → validate → save → export
"""

from __future__ import annotations

import logging
from datetime import datetime
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import export, optimizer, projections, storage
from .config import INBOX, LOG_PATH, Settings, ensure_dirs
from .errors import StepError, StepResult
from .sources import dk, odds, paidcsv, sleeper

log = logging.getLogger("beatnfl")


def _setup_log():
    ensure_dirs()
    if not log.handlers:
        h = logging.FileHandler(LOG_PATH)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)


@dataclass
class WeekRun:
    steps: list[StepResult] = field(default_factory=list)
    slate: dk.LoadedSlate | None = None
    slate_id: int | None = None
    pool: pd.DataFrame | None = None
    run_id: int | None = None
    season: int | None = None
    week: int | None = None
    lines: pd.DataFrame | None = None
    model_meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.pool is not None and not any(s.status == "fail" and s.step.startswith("1") for s in self.steps)

    def add(self, *a, **k):
        r = StepResult(*a, **k)
        self.steps.append(r)
        log.info(r.as_text().replace("\n", " "))
        storage.log_status(r.step, r.status, r.message)
        return r


def _err(run: WeekRun, step: str, e: Exception, status="fail"):
    if isinstance(e, StepError):
        msg = e.what + (f" {e.why}" if e.why else "")
        run.add(step, status, msg, e.fix, e.detail)
    else:
        log.error(traceback.format_exc())
        run.add(step, status, f"Unexpected error: {type(e).__name__}: {str(e)[:160]}",
                "This is a bug. The full detail is in ~/BeatNFL/beatnfl.log — send it over.", traceback.format_exc())


def newest_salary_file() -> Path | None:
    files = [p for p in INBOX.glob("*.csv") if p.name.lower().startswith("dksalaries")]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def run_week(settings: Settings | None = None, *, salary_file: Path | bytes | None = None, draft_group_id: str | None = None,
             paid_file: Path | bytes | None = None, use_model: bool = True, offline: bool = False,
             progress=lambda m: None) -> WeekRun:
    _setup_log()
    s = settings or Settings.load()
    run = WeekRun()

    # 1 ---------------------------------------------------------------- slate
    progress("Loading DraftKings slate…")
    try:
        if salary_file is not None:
            run.slate = dk.load_from_file(salary_file)
        else:
            try:
                run.slate = dk.load_auto(draft_group_id, offline=offline)
            except StepError as auto_err:
                f = newest_salary_file()
                if f is None:
                    raise
                run.slate = dk.load_from_file(f, label=f"{f.name}")
                run.add("1. Slate (automatic)", "warn",
                        f"Automatic DraftKings download failed ({(auto_err.why or auto_err.what)[:140]}) — "
                        f"used {f.name} from your inbox instead (saved {datetime.fromtimestamp(f.stat().st_mtime):%a %b %-d %-I:%M %p}).",
                        "", auto_err.detail)
        sl = run.slate
        starts = pd.to_datetime(sl.players.start_utc.replace("", None), utc=True, errors="coerce").dropna()
        if len(starts) and starts.max() < pd.Timestamp.now(tz="UTC"):
            raise StepError("1. Slate", f"This salary file is for a slate that has already been played "
                            f"(last game {starts.max().tz_convert('America/New_York'):%a %b %-d}).",
                            "It is an old DKSalaries.csv — the app will not swap in a different week silently.",
                            "Export this week's file from DraftKings (any NFL Classic contest → Export to CSV) and load it.")
        if len(starts) and starts.min() < pd.Timestamp.now(tz="UTC"):
            sl.warnings.append("Some games on this slate have already started — locked players are still in the pool.")
        msg = (f"{sl.slate.label if sl.slate else ''} — {len(sl.players)} players, {sl.players.game.nunique()} games "
               f"via {sl.route} ({sl.fetched_note}).")
        # verify JSON-route IDs against an official CSV in the inbox when one matches
        if sl.route == "json":
            f = newest_salary_file()
            if f is not None:
                try:
                    off, _ = dk.parse_salary_csv(f.read_bytes())
                    if set(off.dk_id) & set(sl.players.dk_id):
                        ok, issues = dk.verify_against_csv(sl.players, off)
                        sl.ids_verified = ok
                        sl.warnings.append("IDs verified against official CSV." if ok else "CSV check: " + "; ".join(issues))
                except StepError:
                    pass
            if not sl.ids_verified:
                sl.warnings.append("Export IDs come from the JSON feed and are not yet checked against DK's official CSV.")
        status = "warn" if any(w.startswith("STALE") for w in sl.warnings) else "ok"
        run.add("1. Slate", status, msg + (" " + " ".join(sl.warnings) if sl.warnings else ""))
    except Exception as e:
        _err(run, "1. Slate", e)
        return run

    pool = run.slate.players.copy()

    # 2 ---------------------------------------------------------------- injuries & roles
    progress("Checking injuries and depth charts…")
    sp_players = None
    try:
        sp_players, note = sleeper.players(offline=offline)
        run.add("2. Injuries & roles", "ok" if "STALE" not in note else "warn", f"Sleeper player status loaded ({note}).")
    except Exception as e:
        _err(run, "2. Injuries & roles", e, status="warn")
        run.steps[-1].message += " Using DraftKings' own injury tags only."
    pool = projections.attach_status(pool, sp_players)

    # 2b ------------------------------------------------------------- Yahoo injuries (only if connected)
    from .sources import yahoo
    if s.yahoo_client_id and s.yahoo_client_secret and yahoo.connected() and not offline:
        progress("Checking Yahoo injury tags…")
        try:
            lk = yahoo.first_league(s.yahoo_client_id, s.yahoo_client_secret)
            yp = pd.DataFrame(yahoo.league_players(lk, s.yahoo_client_id, s.yahoo_client_secret)) if lk else pd.DataFrame()
            if yp.empty:
                run.add("2b. Injuries (Yahoo)", "warn", "Yahoo returned no players.", "Check the My Yahoo team tab.")
            else:
                pool, n_changed = projections.attach_external_status(pool, yp, "yahoo")
                run.add("2b. Injuries (Yahoo)", "ok", f"{len(yp)} Yahoo players checked; {n_changed} slate players got a "
                        "more serious status from Yahoo.")
        except Exception as e:
            _err(run, "2b. Injuries (Yahoo)", e, status="warn")
    n_out = int(pool.out.sum())
    n_q = int((pool.status == "Q").sum())
    step2 = next(x for x in run.steps if x.step.startswith("2."))
    step2.message += f" {n_out} ruled out, {n_q} questionable (all sources combined)."

    # 3 ---------------------------------------------------------------- schedule & lines
    progress("Matching slate to the NFL schedule and betting lines…")
    try:
        from .models.train import locate_week
        from .sources import nflverse as nv
        run.season, run.week, _ = locate_week(pool, nv.schedules(offline=offline))
        run.add("3. Schedule", "ok", f"Slate is {run.season} week {run.week}.")
    except Exception as e:
        _err(run, "3. Schedule", e, status="warn")
        try:
            st = sleeper.state(offline=offline)
            run.season, run.week = int(st["season"]), int(st["week"])
            run.steps[-1].message += f" Falling back to Sleeper's current week ({run.season} wk {run.week})."
        except Exception:
            pass
    if s.odds_api_key:
        try:
            gl = odds.game_lines(s.odds_api_key, offline=offline)
            run.lines = odds.consensus(gl)
            storage.save_quotes(gl.assign(stat="", player=""), "game", run.season or 0, run.week or 0)
            run.add("3b. Betting lines", "ok", f"Live lines for {len(run.lines)} games (used for implied team totals).")
        except Exception as e:
            _err(run, "3b. Betting lines", e, status="warn")
            run.steps[-1].message += " Using nflverse lines instead."
    else:
        run.add("3b. Betting lines", "skip", "No Odds API key — using the lines published in the nflverse schedule.",
                "Optional: add a free key in Settings for fresher lines and props.")

    # 4 ---------------------------------------------------------------- projections
    sources: dict[str, pd.DataFrame] = {}
    src_meta = {}
    progress("Getting Sleeper projections…")
    if run.season and run.week:
        try:
            spj, note = sleeper.projections(run.season, run.week, offline=offline)
            m, n = projections.match_external(pool, spj, "proj_sleeper")
            sources["sleeper"] = m
            src_meta["sleeper"] = note
            run.add("4a. Sleeper projections", "ok" if n else "warn", f"{n} slate players matched ({note}).")
        except Exception as e:
            _err(run, "4a. Sleeper projections", e, status="warn")
    progress("Reading your projections…")
    try:
        pf = paid_file
        name = "upload"
        if pf is None and s.fantasypros_api_key and run.season and run.week:
            # FantasyPros API: used unless you uploaded or pasted a table for this run
            from .sources import fantasypros
            fp, info = fantasypros.projections(s.fantasypros_api_key, run.season, run.week, offline=offline)
            m, n = projections.match_external(pool, fp, "proj_paid")
            sources["paid"] = m
            src_meta["paid"] = {"file": "FantasyPros API", "tier": info.get("tier"), "projection": "DK points from FantasyPros stat lines"}
            limited = info.get("limited") or info.get("short")
            active = len(pool[~pool.out])
            st = "ok" if n >= 0.5 * active and not limited else "warn"
            msg = f"FantasyPros API: {n} of {len(pool)} slate players matched (DK points computed from their projected stats)."
            fix = ""
            if limited:
                msg += f" Your key returned sample data only ({'; '.join(info['short']) or 'free tier'})."
                fix = ("Activate production access for your key (MVP/HOF member page at secure.fantasypros.com/api-keys), "
                       "then Refresh.")
            elif st == "warn":
                fix = "Low match rate — FantasyPros may not have posted this week's projections yet; Refresh later."
            run.add("4b. FantasyPros projections", st, msg, fix)
        else:
            if pf is None:
                f = paidcsv.newest_in(INBOX)
                pf, name = (f.read_bytes(), f.name) if f else (None, "")
            elif isinstance(pf, Path):
                pf, name = pf.read_bytes(), pf.name
            if pf is None:
                run.add("4b. Your projections", "skip", "No FantasyPros API key, pasted table or projections file.",
                        "Add your FantasyPros API key in Settings, or paste a projections table on the Load data tab.")
            else:
                pj, mapping = paidcsv.parse(pf, name)
                m, n = projections.match_external(pool, pj, "proj_paid")
                sources["paid"] = m
                src_meta["paid"] = {"file": name, **mapping}
                st = "ok" if n >= 0.5 * len(pool[~pool.out]) else "warn"
                run.add("4b. Your projections", st, f"{name}: {n} of {len(pool)} slate players matched "
                        f"(projection column: '{mapping['projection']}').",
                        "" if st == "ok" else "Low match rate — check the file is for this slate and DraftKings scoring.")
    except Exception as e:
        _err(run, "4b. Your projections", e, status="warn")
    if use_model:
        progress("Running the in-house model…")
        try:
            from .models import train
            pm, dm, meta = train.get_models(offline=offline, progress=progress)
            run.model_meta = meta
            res, warns, info = train.forecast(pool, sp_players, pm, dm, run.lines, offline=offline)
            sources["model"] = res[["dk_id", "proj_model"]].dropna()
            src_meta["model"] = meta
            run.add("4c. In-house model", "ok" if not warns else "warn",
                    f"{len(sources['model'])} players projected (model {meta['tag']}). " + " ".join(warns + info.get("notes", [])))
            run._model_detail = res
        except Exception as e:
            _err(run, "4c. In-house model", e, status="warn")

    if not sources:
        run.add("5. Blend", "fail", "No projection source worked, so lineups can't be built.",
                "Drop a paid projection CSV into the inbox, or fix the failed source above and run again.")
        run.pool = projections.blend(pool, s, {})
        return run

    # 5 ---------------------------------------------------------------- blend + coverage
    pool = projections.blend(pool, s, sources)
    if hasattr(run, "_model_detail"):
        extra = run._model_detail.drop(columns=["proj_model", "player_id", "team"], errors="ignore")
        pool = pool.merge(extra, on="dk_id", how="left")
    active = pool[~pool.out]
    missing = active[active.proj.isna()].sort_values("salary", ascending=False)
    cov = 1 - len(missing) / max(1, len(active))
    msg = f"Projections for {len(active) - len(missing)} of {len(active)} active players ({cov:.0%})."
    if len(missing):
        msg += " No projection (excluded from lineups): " + ", ".join(f"{r['name']} ${r.salary:,}" for _, r in missing.head(6).iterrows())
    big_missing = missing[missing.salary >= 5000]
    run.add("5. Blend", "warn" if len(big_missing) else "ok", msg,
            "Add a paid projection file to cover them." if len(big_missing) else "")
    run.pool = pool

    # 6 ---------------------------------------------------------------- freeze
    try:
        run.slate_id = storage.save_slate(run.slate, run.season, run.week)
        run.run_id = storage.save_projection_run(run.slate_id, pool, run.model_meta.get("tag", "none"), src_meta)
        run.add("6. Saved", "ok", f"Slate snapshot #{run.slate_id} and projection run #{run.run_id} frozen for later grading.")
    except Exception as e:
        _err(run, "6. Saved", e, status="warn")
    return run


def build_lineups(run: WeekRun, opt: optimizer.OptSettings, name: str = "", save=False, export_files=False) -> dict:
    """Optimize and validate. Nothing is saved unless asked — use save_lineups() for that."""
    res = optimizer.build(run.pool, opt)
    out = {"result": res, "opt": opt, "set_id": None, "files": [], "duplicate": False}
    if not res.lineups:
        return out
    bad = [i for i, lu in enumerate(res.lineups) if optimizer.validate(lu, run.pool)]
    if bad:
        res.diagnostics.append(f"{len(bad)} lineups failed independent validation and were removed.")
        res.lineups = [lu for i, lu in enumerate(res.lineups) if i not in bad]
    proj = run.pool.set_index("dk_id").proj
    res.lineups = sorted(res.lineups, key=lambda lu: -float(proj.loc[lu].sum()))   # rank 1 = highest projection
    if save or export_files:
        out.update(save_lineups(run, out, name, export_files=export_files))
    return out


def save_lineups(run: WeekRun, out: dict, name: str = "", export_files=True) -> dict:
    """Save a built set (skipped if an identical set is already saved) and write the two CSVs to exports."""
    res = out["result"]
    label = name or (f"{run.season}_wk{run.week}" if run.week else "slate")
    r = {"set_id": None, "files": [], "duplicate": False}
    if run.slate_id:
        before = storage.find_same_set(run.slate_id, res.lineups)
        r["set_id"] = before or storage.save_lineup_set(label, run.slate_id, run.run_id, vars(out["opt"]) if out.get("opt") else {},
                                                        res.lineups, run.pool, res.status, res.diagnostics)
        r["duplicate"] = before is not None
    if export_files:
        r["files"] = [export.dk_upload(res.lineups, run.pool, label), export.readable(res.lineups, run.pool, label)]
    return r
