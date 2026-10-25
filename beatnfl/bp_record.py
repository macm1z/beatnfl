"""BettingPros track record — have their prop ratings actually won, at real prices?

Two sources of graded props:
  1. BettingPros' archive: finished weeks from the Consumer API (`event_status=completed`) carry their
     projection, recommended side, win probability, expected value, star rating, the lines and the
     player's actual stat. Caveat: we can't verify these ratings are exactly what was shown before
     kickoff (an archive may be rebuilt), so this is their own record, taken at face value.
  2. Snapshots the app saves whenever it loads the current week's BettingPros board (before kickoff).
     Graded against the actual results once the games are final. Slower to build, but nobody can
     revise it — this is the trustworthy record.

Each pick is graded at the recommended side's best listed price (what BettingPros advertises; a
single bettor usually can't get the best price everywhere) and at the consensus price. Results are
split by star rating, market and BettingPros' model (bet_signals / projection), with ROI ± one
standard error. Break-even at -110 is 52.4%.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HOME
from .errors import StepError
from .props_backtest import model_information
from .sources import bettingpros as bp

DIR = HOME / "evidence" / "bettingpros"
RESERVE = 5          # requests kept back each day for the Props tab


def _dec(american) -> float:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(a) or a == 0:
        return np.nan
    return 1 + (a / 100 if a > 0 else 100 / -a)


def week_file(season: int, week: int) -> Path:
    return DIR / f"archive_{season}_w{week:02d}.csv"


def snap_file(season: int) -> Path:
    return DIR / f"snapshots_{season}.csv"


def save_snapshot(df: pd.DataFrame) -> int:
    """Append the current (pre-kickoff) board to the season's snapshot file. Returns rows added."""
    if df is None or df.empty or "season" not in df:
        return 0
    DIR.mkdir(parents=True, exist_ok=True)
    d = df.assign(fetched_utc=pd.Timestamp.now(tz="UTC").isoformat())
    n = 0
    for season, g in d.groupby("season"):
        f = snap_file(int(season))
        g.to_csv(f, mode="a", header=not f.exists(), index=False)
        n += len(g)
    return n


def collect(key: str, seasons: list[int], weeks=range(1, 19), offline=False, progress=lambda m: None,
            current: tuple[int, int] | None = None) -> dict:
    """Download finished weeks into the archive, oldest first, within today's allowance (resumable)."""
    DIR.mkdir(parents=True, exist_ok=True)
    done, fetched, stopped = [], [], None
    for s in seasons:
        for w in weeks:
            if current and (s, w) >= current:
                break
            f = week_file(s, w)
            if f.exists():
                done.append((s, w))
                continue
            if bp.remaining() <= RESERVE and not offline:
                stopped = "allowance"
                break
            try:
                df, info = bp.props(key, s, w, completed=True, offline=offline, progress=progress,
                                    max_requests=max(0, bp.remaining() - RESERVE))
            except StepError as e:
                if "allowance" in e.what:
                    stopped = "allowance"
                    break
                raise
            if info["partial"]:
                stopped = "allowance"
                break
            df.to_csv(f, index=False)          # an empty week is saved too, so it isn't asked for again
            fetched.append((s, w))
        if stopped:
            break
    return {"weeks_done": len(done) + len(fetched), "fetched_now": fetched, "stopped": stopped,
            "requests_left": bp.remaining()}


IDS = ["bp_player_id", "market_id", "event_id"]


def _ids(d: pd.DataFrame) -> pd.DataFrame:
    for c in IDS:
        if c in d:
            d[c] = d[c].astype(str).str.replace(r"\.0$", "", regex=True)
    return d


def load_archive() -> pd.DataFrame:
    fs = sorted(DIR.glob("archive_*.csv"))
    frames = []
    for f in fs:
        try:
            frames.append(pd.read_csv(f))
        except (pd.errors.EmptyDataError, ValueError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """One graded row per recommended pick: won (1/0), profit per 1 unit at best and at consensus price."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[(df.scored.astype(bool)) & ~(df.push.astype(bool)) & df.bp_side.isin(["over", "under"])
           & df.actual.notna() & df.rec_line.notna()].copy()
    if d.empty:
        return d
    over = d.bp_side == "over"
    d = d[d.actual != d.rec_line]                      # exact hit on the line = push
    over = d.bp_side == "over"
    d["won"] = np.where(over, d.actual > d.rec_line, d.actual < d.rec_line).astype(int)
    dec_best = d.rec_odds.map(_dec)
    dec_cons = d.rec_cons_odds.map(_dec)
    d["profit_best"] = np.where(d.won == 1, dec_best - 1, -1.0)
    d["profit_cons"] = np.where(d.won == 1, dec_cons - 1, -1.0)
    d.loc[dec_best.isna(), "profit_best"] = np.nan
    d.loc[dec_cons.isna(), "profit_cons"] = np.nan
    d["stars"] = pd.to_numeric(d.bp_stars, errors="coerce")
    return d


def _summ(g: pd.DataFrame) -> dict:
    n = len(g)
    pb, pc = g.profit_best.dropna(), g.profit_cons.dropna()
    return {"picks": n, "won_pct": round(100 * g.won.mean(), 1) if n else np.nan,
            "roi_best_pct": round(100 * pb.mean(), 1) if len(pb) else np.nan,
            "roi_best_se": round(100 * pb.std(ddof=1) / math.sqrt(len(pb)), 1) if len(pb) > 1 else np.nan,
            "roi_cons_pct": round(100 * pc.mean(), 1) if len(pc) else np.nan,
            "roi_cons_se": round(100 * pc.std(ddof=1) / math.sqrt(len(pc)), 1) if len(pc) > 1 else np.nan,
            "their_win_prob_pct": round(100 * pd.to_numeric(g.bp_prob, errors="coerce").mean(), 1)
            if pd.to_numeric(g.bp_prob, errors="coerce").notna().any() else np.nan}


def summarize(graded: pd.DataFrame) -> dict:
    """Tables by star rating, by market and by model, plus a plain verdict."""
    if graded is None or graded.empty:
        return {"verdict": "No graded BettingPros picks yet.", "by_stars": pd.DataFrame(), "by_market": pd.DataFrame(),
                "by_model": pd.DataFrame(), "n": 0}
    rated = graded[graded.stars.notna()]
    out = {"n": len(graded), "n_rated": len(rated)}
    out["by_stars"] = pd.DataFrame([{"stars": int(s), **_summ(g)} for s, g in rated.groupby("stars")])
    out["by_market"] = pd.DataFrame([{"market": m, **_summ(g)} for m, g in graded.groupby("market")]).sort_values(
        "picks", ascending=False) if "market" in graded else pd.DataFrame()
    out["by_model"] = pd.DataFrame([{"model": m, **_summ(g)} for m, g in graded.groupby(graded.bp_signal.fillna("unknown"))])
    top = rated[rated.stars >= 4]
    if rated.empty:
        out["verdict"] = ("BettingPros returned graded props but no star ratings for these weeks — their archive "
                          "doesn't keep the ratings, so the record has to be built from the app's own pre-game "
                          "snapshots going forward.")
    else:
        s = _summ(top) if len(top) else None
        if not s or s["picks"] < 300:
            out["verdict"] = (f"{len(top)} graded 4–5★ picks so far — too few to judge (need several hundred). "
                              "Keep collecting.")
        else:
            lo = s["roi_cons_pct"] - 2 * s["roi_cons_se"]
            hi = s["roi_cons_pct"] + 2 * s["roi_cons_se"]
            if lo > 0:
                out["verdict"] = (f"4–5★ picks: {s['picks']} graded, won {s['won_pct']}%, ROI {s['roi_cons_pct']:+}% ± "
                                  f"{s['roi_cons_se']} at consensus prices — profitable beyond luck (95% range "
                                  f"{lo:+.1f}% to {hi:+.1f}%).")
            elif hi < 0:
                out["verdict"] = (f"4–5★ picks: {s['picks']} graded, won {s['won_pct']}%, ROI {s['roi_cons_pct']:+}% ± "
                                  f"{s['roi_cons_se']} at consensus prices — losing money after the books' cut.")
            else:
                out["verdict"] = (f"4–5★ picks: {s['picks']} graded, won {s['won_pct']}%, ROI {s['roi_cons_pct']:+}% ± "
                                  f"{s['roi_cons_se']} at consensus prices — not distinguishable from break-even "
                                  f"(95% range {lo:+.1f}% to {hi:+.1f}%).")
    return out


def _implied(a) -> float:
    d = _dec(a)
    return 1 / d if d == d else np.nan


def _mi(d: pd.DataFrame) -> dict:
    try:
        return model_information(d)
    except np.linalg.LinAlgError:          # e.g. every price identical — nothing to separate
        return {}


def info_test(df: pd.DataFrame) -> dict:
    """Does BettingPros' probability tell us anything the sportsbooks' price doesn't?
    For every graded prop where their pick is at the consensus line: did the Over hit, vs the market's
    no-vig chance of the Over (consensus prices, both sides) and BettingPros' chance of the Over.
    Scored two ways: accuracy of each chance (Brier, lower is better) and the weight BettingPros' view gets
    once the market price is known (logistic; ≈2× its ± or more = real information). Same test the
    BeatNFL model failed on 2025 lines."""
    if df is None or df.empty:
        return {"n": 0, "verdict": "No graded props yet."}
    d = df[df.scored.astype(bool) & ~df.push.astype(bool) & df.actual.notna() & df.line.notna()
           & df.bp_side.isin(["over", "under"]) & pd.to_numeric(df.bp_prob, errors="coerce").notna()
           & df.over_cons_odds.notna() & df.under_cons_odds.notna()].copy()
    d = d[(d.rec_line.isna()) | (d.rec_line == d.line)]
    d = d[d.actual != d.line]
    if d.empty:
        return {"n": 0, "verdict": "No graded props with BettingPros win chances and both consensus prices yet."}
    po, pu = d.over_cons_odds.map(_implied), d.under_cons_odds.map(_implied)
    d["p_over"] = po / (po + pu)
    bp_p = pd.to_numeric(d.bp_prob, errors="coerce").clip(0.01, 0.99)
    d["model_p_over"] = np.where(d.bp_side == "over", bp_p, 1 - bp_p)
    d["over"] = (d.actual > d.line).astype(int)
    d = d[d.p_over.notna()]
    out = {"n": int(len(d)),
           "brier_market": round(float(((d.p_over - d.over) ** 2).mean()), 4),
           "brier_bp": round(float(((d.model_p_over - d.over) ** 2).mean()), 4),
           "info": _mi(d)}
    out["by_stat"] = pd.DataFrame([{"market": m, "props": len(g),
                                    "brier_market": round(float(((g.p_over - g.over) ** 2).mean()), 4),
                                    "brier_bp": round(float(((g.model_p_over - g.over) ** 2).mean()), 4),
                                    **{f"bp_{k}": v for k, v in _mi(g).items() if k != "n"}}
                                   for m, g in d.groupby(d.market.fillna("?")) if len(g) >= 50]).sort_values(
        "props", ascending=False) if len(d) >= 50 else pd.DataFrame()
    i = out["info"]
    if not i:
        out["verdict"] = f"{len(d)} graded props — need at least 200 to test (keep collecting)."
    elif i["z"] >= 2 and out["brier_bp"] < out["brier_market"]:
        out["verdict"] = (f"BettingPros adds real information beyond the line ({len(d):,} props; weight {i['coef']:+.2f} ± "
                          f"{i['se']:.2f}; accuracy {out['brier_bp']:.4f} vs market {out['brier_market']:.4f}).")
    else:
        out["verdict"] = (f"BettingPros adds no proven information beyond the line ({len(d):,} props; weight "
                          f"{i['coef']:+.2f} ± {i['se']:.2f}; accuracy {out['brier_bp']:.4f} vs market "
                          f"{out['brier_market']:.4f} — lower is better).")
    return out


def snapshot_record(key: str, offline=False, progress=lambda m: None) -> pd.DataFrame:
    """Grade the app's own pre-game snapshots against finished results (uses the archive when present,
    otherwise downloads the finished week). Keeps each pick's LAST snapshot before kickoff."""
    fs = sorted(DIR.glob("snapshots_*.csv"))
    if not fs:
        return pd.DataFrame()
    S = _ids(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True))
    S = S.sort_values("fetched_utc").drop_duplicates(["bp_player_id", "market_id", "event_id"], keep="last")
    res = []
    for (s, w), g in S.groupby(["season", "week"]):
        f = week_file(int(s), int(w))
        if f.exists():
            A = pd.read_csv(f)
        else:
            try:
                A, info = bp.props(key, int(s), int(w), completed=True, offline=offline, progress=progress)
                if info["partial"]:
                    continue
                A.to_csv(f, index=False)
            except StepError:
                continue
        if A.empty:
            continue
        A = _ids(A.copy())
        keep = ["bp_player_id", "market_id", "event_id", "scored", "push", "actual"]
        m = g.drop(columns=[c for c in ("scored", "push", "actual") if c in g]).merge(
            A[keep].drop_duplicates(["bp_player_id", "market_id", "event_id"]), on=["bp_player_id", "market_id", "event_id"])
        res.append(m)
    return grade(pd.concat(res, ignore_index=True)) if res else pd.DataFrame()


def save_result(name: str, summ: dict):
    DIR.mkdir(parents=True, exist_ok=True)
    def conv(v):
        if isinstance(v, pd.DataFrame):
            return v.to_dict("records")
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    js = {k: conv(v) for k, v in summ.items()}
    js["generated"] = time.strftime("%Y-%m-%d %H:%M")
    (DIR / f"record_{name}.json").write_text(json.dumps(js, default=str))


def load_result(name: str) -> dict | None:
    f = DIR / f"record_{name}.json"
    if not f.exists():
        return None
    js = json.loads(f.read_text())
    for k in ("by_stars", "by_market", "by_model"):
        js[k] = pd.DataFrame(js.get(k) or [])
    if isinstance(js.get("info"), dict):
        js["info"]["by_stat"] = pd.DataFrame(js["info"].get("by_stat") or [])
    return js


def current_week(offline=False) -> tuple[int, int]:
    from .sources import nflverse as nv
    return nv.upcoming_week(nv.schedules(offline=offline))


def snapshot_now(key: str, offline=False, progress=lambda m: None) -> dict:
    """Save this week's BettingPros board (pre-kickoff ratings) for the forward record."""
    season, week = current_week(offline)
    df, info = bp.props(key, season, week, completed=False, offline=offline, progress=progress)
    return {"season": season, "week": week, "rows": save_snapshot(df), **info}


def daily(key: str, offline=False, progress=lambda m: None) -> dict:
    """The scheduled routine: grow the archive with today's spare allowance, then re-grade both records."""
    season, week = current_week(offline)
    c = collect(key, [season - 1, season], current=(season, week), offline=offline, progress=progress)
    A = load_archive()
    arch = summarize(grade(A))
    arch["info"] = info_test(A)
    save_result("archive", arch)
    G = snapshot_record(key, offline=offline, progress=progress)
    snap = summarize(G)
    snap["info"] = info_test(G)
    save_result("snapshots", snap)
    return {"collect": c, "archive": arch["verdict"], "snapshots": snap["verdict"], "season": season, "week": week}
