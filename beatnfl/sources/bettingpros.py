"""BettingPros Consumer API (read-only, personal use) — player props with BettingPros' projections and ratings.

Key: requested once on bettingpros.com; it follows your membership. Sent as the `x-api-key` header.
  Free / Plus: 100 requests a day, up to 20 props per page · Elite: 1,000 a day, up to 50 per page.
  Plus and Elite add the paid fields: win probability, expected value and the 1–5 star bet rating.
  The daily allowance resets at midnight UTC (8 PM Eastern in summer, 7 PM in winter).

Every response is cached on disk (completed weeks for good, current boards for an hour), and the app
counts the calls it makes per UTC day so it stops before the allowance runs out instead of failing midway.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from ..config import HOME
from ..errors import StepError
from ..names import loose_key, norm_name
from ..net import fetch_json

BASE = "https://api.bettingpros.com/v3"
STEP = "BettingPros"
USAGE = HOME / "bettingpros_usage.json"
DAILY = {"elite": 1000, "goat": 1000}          # every other tier: 100
CONSENSUS_BOOK = 0

# BettingPros market slug → BeatNFL stat name (matched by keywords; unknown markets keep their own label)
STAT_WORDS = [("passing-yards", "pass_yds"), ("passing-touchdowns", "pass_td"), ("passing-attempts", "pass_att"),
              ("completions", "pass_cmp"), ("interceptions", "pass_int"), ("rushing-yards", "rush_yds"),
              ("rushing-attempts", "rush_att"), ("receiving-yards", "rec_yds"), ("receptions", "rec"),
              ("anytime", "any_td"), ("touchdown-scorer", "any_td"), ("longest-reception", "long_rec"),
              ("longest-rush", "long_rush"), ("rushing-receiving-yards", "rush_rec_yds"),
              ("passing-rushing-yards", "pass_rush_yds")]


# ------------------------------------------------------------------ allowance
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def usage() -> dict:
    """{"date", "used", "tier", "limit"} for the current UTC day."""
    try:
        u = json.loads(USAGE.read_text())
    except Exception:  # noqa: BLE001
        u = {}
    if u.get("date") != _today():
        u = {"date": _today(), "used": 0, "tier": u.get("tier"), "page_limit": u.get("page_limit")}
    u["limit"] = DAILY.get(str(u.get("tier") or "").lower(), 100)
    return u


def _note(used_delta=0, tier=None, page_limit=None, exhausted=False):
    u = usage()
    u["used"] = u["limit"] if exhausted else u["used"] + used_delta
    if tier:
        u["tier"] = tier
    if page_limit:
        u["page_limit"] = page_limit
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        USAGE.write_text(json.dumps(u))
    except OSError:
        pass


def remaining() -> int:
    u = usage()
    return max(0, u["limit"] - u["used"])


# ------------------------------------------------------------------ requests
def _need_key(key: str):
    if not key:
        raise StepError(STEP, "No BettingPros API key set.", "",
                        "Paste your BettingPros consumer API key in Settings (sidebar) and click Save settings.")


def get(key: str, path: str, params: dict, max_age_h: float = 1.0, offline=False) -> dict:
    _need_key(key)
    from ..net import _key as _ck  # noqa: PLC0415 — same cache key the downloader uses
    from ..config import CACHE
    cached = (CACHE / f"{_ck(BASE + path, params)}.bin").exists()
    if not offline and remaining() <= 0 and not cached:
        u = usage()
        raise StepError(STEP, f"Today's BettingPros allowance is used up ({u['used']}/{u['limit']} requests).",
                        "It resets at midnight UTC (8 PM Eastern).", "Try again after the reset — everything "
                        "already downloaded stays cached and is not requested again.")
    try:
        data, meta = fetch_json(BASE + path, params=params, headers={"x-api-key": key}, step=STEP,
                                max_age_h=max_age_h, offline=offline, retries=2)
    except StepError as e:
        d = e.detail or ""
        if "HTTP 403" in d:
            raise StepError(STEP, "BettingPros refused the API key.", "The key is wrong, not approved yet, or deactivated.",
                            "Check the key in your BettingPros account, paste it again in Settings, then retry.",
                            detail=d) from None
        if "HTTP 429" in d and "Limit Exceeded" in d:
            _note(exhausted=True)
            raise StepError(STEP, "Today's BettingPros allowance is used up.", "It resets at midnight UTC (8 PM Eastern).",
                            "Try again after the reset; downloaded data stays cached.", detail=d) from None
        if "HTTP 400" in d:
            raise StepError(STEP, "BettingPros rejected the request.", "A parameter wasn't accepted.",
                            "Copy diagnostics (Data status tab) and send them.", detail=d) from None
        raise
    ca = data.get("consumer_api") or {}
    _note(0 if meta.from_cache else 1, ca.get("tier"), ca.get("page_limit"))
    return data


def markets(key: str, offline=False) -> dict[int, dict]:
    """NFL markets: id → {slug, label, stat}. One request, cached for a week."""
    data = get(key, "/markets", {"sport": "NFL"}, max_age_h=24 * 7, offline=offline)
    out = {}
    for m in data.get("markets", []):
        slug = str(m.get("slug") or "")
        stat = next((s for w, s in STAT_WORDS if w in slug), slug)
        # "rushing-receiving-yards" also contains "receiving-yards": prefer the longest keyword match
        hits = [(len(w), s) for w, s in STAT_WORDS if w in slug]
        if hits:
            stat = max(hits)[1]
        out[int(m["id"])] = {"slug": slug, "label": (m.get("meta") or {}).get("short_label") or slug, "stat": stat}
    return out


def books(key: str, offline=False) -> dict[int, str]:
    data = get(key, "/books", {}, max_age_h=24 * 7, offline=offline)
    return {int(b["id"]): b.get("display_name") or b.get("name") or str(b["id"]) for b in data.get("books", [])}


def _flat(p: dict, mk: dict) -> dict:
    part = p.get("participant") or {}
    pl = part.get("player") or {}
    pr = p.get("projection") or {}
    ov, un = p.get("over") or {}, p.get("under") or {}
    sc = p.get("scoring") or {}
    m = mk.get(int(p.get("market_id") or -1), {})
    side = (pr.get("recommended_side") or "").lower() or None
    rec = ov if side == "over" else un if side == "under" else {}
    name = part.get("name") or ""
    return {
        "bp_player_id": part.get("id"), "player": name, "pos": pl.get("position"), "team": pl.get("team"),
        "event_id": p.get("event_id"), "market_id": p.get("market_id"), "market": m.get("label"),
        "stat": m.get("stat"), "line": pr.get("consensus_line") or ov.get("consensus_line") or un.get("consensus_line"),
        "over_line": ov.get("line"), "over_odds": ov.get("odds"), "over_book": ov.get("book"),
        "over_cons_odds": ov.get("consensus_odds"),
        "under_line": un.get("line"), "under_odds": un.get("odds"), "under_book": un.get("book"),
        "under_cons_odds": un.get("consensus_odds"),
        "bp_proj": pr.get("value"), "bp_side": side, "bp_prob": pr.get("probability"),
        "bp_ev": pr.get("expected_value"), "bp_stars": pr.get("bet_rating"), "bp_diff": pr.get("diff"),
        "bp_fair_odds": pr.get("fair_odds"), "bp_signal": pr.get("signal_source"),
        "mkt_proj": pr.get("market_value"), "mkt_side": pr.get("market_recommended_side"),
        "mkt_prob": pr.get("market_probability"),
        "rec_line": rec.get("line"), "rec_odds": rec.get("odds"), "rec_book": rec.get("book"),
        "rec_cons_odds": rec.get("consensus_odds"),
        "scored": bool(sc.get("is_scored")), "push": bool(sc.get("is_push")), "actual": sc.get("actual"),
        "nn": norm_name(name), "lkey": loose_key(name, pl.get("position") or ""),
    }


def props(key: str, season: int, week: int, completed: bool | None = None, offline=False, max_requests: int | None = None,
          progress=lambda m: None) -> tuple[pd.DataFrame, dict]:
    """Every NFL prop on BettingPros' board for one week (all pages), one row per player-market.
    completed=True asks for finished games (graded, with actual results); False for upcoming; None for both.
    Stops early (info["partial"]=True) if the daily allowance would run out — call again tomorrow to finish;
    pages already downloaded come from the cache."""
    mk = markets(key, offline=offline)
    params = {"sport": "NFL", "season": int(season), "week": int(week), "limit": 50, "include_injured": "true",
              "ev_threshold": "false", "include_selections": "false"}
    if completed is not None:
        params["event_status"] = "completed" if completed else "upcoming"
    age = 24 * 365 if completed else 1.0
    rows, page, info = [], 1, {"pages": 0, "partial": False, "requests_left": remaining()}
    start_used = usage()["used"]
    while True:
        q = {**params, "page": page}
        if max_requests is not None and usage()["used"] - start_used >= max_requests:
            info["partial"] = True
            break
        try:
            data = get(key, "/props", q, max_age_h=age, offline=offline)
        except StepError as e:
            if "allowance" in e.what and rows:
                info["partial"] = True
                break
            raise
        rows += [_flat(p, mk) for p in data.get("props", [])]
        info["pages"] += 1
        pg = data.get("_pagination") or {}
        total = int(pg.get("total_pages") or 1)
        progress(f"BettingPros {season} week {week}: page {page} of {total}")
        if page >= total or not data.get("props"):
            break
        page += 1
    info["requests_left"] = remaining()
    info["tier"] = usage().get("tier")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["bp_player_id", "market_id", "event_id"]).reset_index(drop=True)
        df["season"], df["week"] = int(season), int(week)
    return df, info
