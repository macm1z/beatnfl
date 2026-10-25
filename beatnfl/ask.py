"""Ask — questions about this week's data, answered by Claude from the app's own numbers.

Claude (Anthropic API, your key in Settings) is given read-only tools over what the app already
has loaded: the player pool and its projection sources, games and lines, props, saved lineups and
their grades, the Evidence results and the data-status log. It can also build lineups on request
(not saved, nothing entered). Answers say which numbers they used; when the data doesn't cover a
question the answer says so instead of guessing. Information only — no bets, no contest entry.

Key: platform.claude.com/settings/keys (pick the Default workspace). Cost: billed by Anthropic per use (tokens). Each answer shows its approximate cost.
"""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .errors import StepError
from .names import norm_name

API = "https://api.anthropic.com/v1/messages"
VERSION = "2023-06-01"
STEP = "Ask"
MODELS = {  # id → (label, $ per million input tokens, $ per million output tokens)
    "claude-sonnet-5-5": ("Sonnet 5.5 — best answers", 2.0, 10.0),
    "claude-haiku-5-5": ("Haiku 5.5 — cheapest", 0.10, 0.50),
    "claude-opus-5-5": ("Opus 5.5 — most thorough", 4.0, 20.0),
}
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ROUNDS = 10
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"

SYSTEM = """You are the analyst inside BeatNFL, a personal NFL research app. The user builds DraftKings NFL Classic \
lineups — usually ONE lineup for single-entry contests or cash games (50/50, double-up, head-to-head) — and looks \
at game lines and player props for information. The app never places bets or enters contests, and neither do you.

How to answer
- Use the tools to get numbers; never invent a projection, salary, line, injury or result. If the tools don't have \
it, say exactly what's missing and where in the app to load it (e.g. "Props tab → Load props", "⟳ Refresh").
- Be direct and brief: lead with the answer, then the few numbers that support it. Short tables are fine.
- Name the numbers you used (e.g. "Combined 18.4 = FantasyPros 19.1, Sleeper 17.0, model 18.6").
- Projections: "Combined" is what lineups use (weighted blend of FantasyPros, Sleeper and the BeatNFL model; the \
model is already moved toward sportsbook prop lines). Floor/Ceiling = model's 15th/85th percentile.
- Be honest about uncertainty. The Evidence results show what has and hasn't been proven: props and game picks have \
NOT shown a proven edge over the sportsbooks; say so when relevant rather than hyping a pick. For BettingPros star \
ratings, quote their graded track record (bettingpros tool) — a rating is not evidence by itself.
- For lineup requests use build_lineups (contest "cash" or "se" for single entry unless asked otherwise). Lineups you \
build are not saved; tell the user they can use the "Use on This week" button under your answer to save/export them.
- Times are US Eastern. Money in dollars. This week is {season} week {week}."""


# ------------------------------------------------------------------ context the app hands in
@dataclass
class Context:
    run: object = None                                  # pipeline.WeekRun (pool, steps, season, week)
    season: int | None = None
    week: int | None = None
    games: Callable[[], tuple] | None = None            # → (lines table, chances table)
    forecasts: Callable[[], pd.DataFrame | None] | None = None
    props: Callable[[], pd.DataFrame | None] | None = None
    bettingpros: Callable[[], pd.DataFrame | None] | None = None
    offline: bool = False
    built: dict | None = None                          # last build_lineups result during this question (pipeline dict)


# ------------------------------------------------------------------ helpers
def _records(df: pd.DataFrame, cols=None, limit=40) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df[[c for c in (cols or df.columns) if c in df.columns]].head(limit)
    out = []
    for r in d.to_dict("records"):
        row = {}
        for k, v in r.items():
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v in ("", "nan", "None")):
                continue
            if isinstance(v, (np.floating, float)):
                v = round(float(v), 2)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, (np.bool_, bool)):
                v = bool(v)
            elif not isinstance(v, (int, str)):
                v = str(v)
            row[k] = v
        out.append(row)
    return out


def _pool(ctx: Context) -> pd.DataFrame | None:
    r = ctx.run
    return None if r is None or getattr(r, "pool", None) is None else r.pool


POOL_COLS = ["name", "pos", "team", "opp", "salary", "proj", "value", "proj_paid", "proj_sleeper", "proj_model",
             "p_dk_floor", "p_dk_ceiling", "proj_lines_adj", "status", "injury_note", "out", "start_utc"]
POOL_NAMES = {"proj": "combined", "proj_paid": "fantasypros_or_yours", "proj_sleeper": "sleeper", "proj_model": "model",
              "p_dk_floor": "floor", "p_dk_ceiling": "ceiling", "proj_lines_adj": "book_lines_moved_model_by",
              "value": "pts_per_1k", "start_utc": "kickoff_utc"}


def _find(df: pd.DataFrame, name: str, col="name") -> pd.DataFrame:
    """Rows whose name matches (exact normalised, then contains, then closest spelling)."""
    if df is None or df.empty or not name:
        return pd.DataFrame()
    nn = df[col].astype(str).map(norm_name)
    q = norm_name(name)
    hit = df[nn == q]
    if hit.empty:
        hit = df[nn.str.contains(q, regex=False)]
    if hit.empty:
        close = difflib.get_close_matches(q, nn.unique().tolist(), n=3, cutoff=0.75)
        hit = df[nn.isin(close)]
    return hit


def _ids_for(pool: pd.DataFrame, names: list[str]) -> tuple[list, list]:
    ids, missing = [], []
    for n in names or []:
        h = _find(pool, n)
        if h.empty and n.upper() in set(pool.team):            # "DEN" → Denver DST
            h = pool[(pool.pos == "DST") & (pool.team == n.upper())]
        (ids.append(h.dk_id.iloc[0]) if len(h) else missing.append(n))
    return ids, missing


# ------------------------------------------------------------------ tools
def t_pool_players(ctx, position=None, team=None, min_salary=None, max_salary=None, sort_by="proj", limit=25,
                   include_out=False):
    p = _pool(ctx)
    if p is None:
        return {"error": "No slate loaded. Click ⟳ Refresh (top right) first."}
    d = p.copy()
    if not include_out and "out" in d:
        d = d[~d.out.astype(bool)]
    if position:
        d = d[d.pos.isin([x.strip().upper() for x in str(position).split(",")])]
    if team:
        d = d[d.team.isin([x.strip().upper() for x in str(team).split(",")])]
    if min_salary:
        d = d[d.salary >= float(min_salary)]
    if max_salary:
        d = d[d.salary <= float(max_salary)]
    key = {"proj": "proj", "value": "value", "ceiling": "p_dk_ceiling", "floor": "p_dk_floor", "salary": "salary"}.get(
        sort_by, "proj")
    if key not in d:
        key = "proj"
    d = d.sort_values(key, ascending=False, na_position="last")
    return {"count_matching": int(len(d)), "sorted_by": key,
            "players": _records(d.rename(columns=POOL_NAMES), [POOL_NAMES.get(c, c) for c in POOL_COLS],
                                limit=min(int(limit or 25), 60))}


def t_player(ctx, name):
    out = {}
    p = _pool(ctx)
    if p is not None:
        h = _find(p, name)
        if len(h):
            drop = {"key", "dk_status", "game", "proj_raw", "n_sources", "lkey"}
            out["dk_pool"] = _records(h.drop(columns=[c for c in drop if c in h]).rename(columns=POOL_NAMES), limit=3)
    fc = ctx.forecasts() if ctx.forecasts else None
    if fc is not None and not fc.empty:
        h = _find(fc, name)
        if len(h):
            cols = [c for c in fc.columns if c.startswith(("p_", "fp_")) or c in ("name", "position", "team", "opp", "injury")]
            out["stat_forecast"] = _records(h, cols, limit=2)
    b = ctx.props() if ctx.props else None
    if b is not None and not b.empty:
        col = "player" if "player" in b else "name" if "name" in b else None
        if col:
            h = _find(b, name, col)
            if len(h):
                out["prop_lines"] = _records(h, limit=20)
    if not out:
        return {"error": f"No player matching '{name}' in the loaded data (DK pool"
                         + (", forecasts" if fc is not None else "") + (", props" if b is not None else "") + ")."}
    if fc is None:
        out["note"] = "Detailed stat forecasts aren't loaded (Props tab → Forecast every player this week)."
    if b is None:
        out["note_props"] = "Prop lines aren't loaded (Props tab → Load props)."
    return out


def t_games(ctx):
    if not ctx.games:
        return {"error": "Games aren't available."}
    try:
        lines, probs = ctx.games()
    except StepError as e:
        return {"error": e.as_text()}
    return {"lines": _records(lines, limit=20), "chances_market_vs_model": _records(probs, limit=80),
            "note": "Model chances are a second opinion only — the game model hasn't beaten closing lines historically."}


def t_props(ctx, player=None, game=None, limit=30):
    b = ctx.props() if ctx.props else None
    if b is None or b.empty:
        return {"error": "Prop lines aren't loaded this session. Props tab → Load props (uses Odds API credits)."}
    d = b
    if player:
        col = "player" if "player" in d else "name"
        d = _find(d, player, col)
    if game and "game" in d:
        d = d[d.game.astype(str).str.contains(str(game).upper(), regex=False)]
    return {"rows": _records(d, limit=min(int(limit or 30), 80)), "count": int(len(d))}


def t_bettingpros(ctx, player=None, min_stars=None, market=None, limit=30):
    b = ctx.bettingpros() if ctx.bettingpros else None
    out = {}
    if b is None or b.empty:
        out["board"] = "BettingPros props aren't loaded this session (Props tab → Load BettingPros props)."
    else:
        d = b
        if player:
            d = _find(d, player, "player")
        if min_stars:
            d = d[pd.to_numeric(d.bp_stars, errors="coerce") >= float(min_stars)]
        if market:
            d = d[d.market.astype(str).str.contains(str(market), case=False, regex=False)]
        d = d.sort_values(["bp_stars", "bp_ev"], ascending=False, na_position="last")
        out["props"] = _records(d, ["player", "team", "market", "line", "bp_proj", "bp_side", "bp_stars", "bp_prob",
                                    "bp_ev", "rec_odds", "rec_line", "mkt_proj"], limit=min(int(limit or 30), 60))
        out["count"] = int(len(d))
    try:
        from . import bp_record
        rec = {k: bp_record.load_result(k) for k in ("snapshots", "archive")}
        out["their_track_record"] = {k: {"verdict": v["verdict"], "by_stars": _records(v["by_stars"]),
                                         "beyond_the_line": (v.get("info") or {}).get("verdict")} for k, v in rec.items() if v}
    except Exception:  # noqa: BLE001
        pass
    return out


def t_saved_sets(ctx):
    from . import storage
    s = storage.list_lineup_sets()
    return {"sets": _records(s, limit=40)}


def t_saved_set(ctx, set_id, grade=False):
    from . import grading, storage
    try:
        d = storage.load_lineup_set(int(set_id))
        _, pool = storage.get_projection_run(d["projection_run_id"])
    except Exception as e:  # noqa: BLE001
        return {"error": f"Couldn't open set #{set_id}: {e}"}
    idx = pool.set_index("dk_id")
    lus = [{"lineup": i, "projected": round(float(idx.loc[lu].proj.sum()), 1), "salary": int(idx.loc[lu].salary.sum()),
            "players": [f"{idx.at[x, 'pos']} {idx.at[x, 'name']} ({idx.at[x, 'team']}) ${int(idx.at[x, 'salary'])} "
                        f"proj {idx.at[x, 'proj']:.1f}" for x in lu]} for i, lu in enumerate(d["lineups"], 1)]
    out = {"name": d.get("name"), "lineups": lus[:20], "n_lineups": len(lus)}
    if grade:
        try:
            g = grading.grade_set(int(set_id), offline=ctx.offline)
            out["grade_summary"] = g["summary"]
            out["players_actual"] = _records(g["players"], limit=40)
            if g.get("hindsight"):
                out["best_possible_lineup"] = _records(g["hindsight"]["table"], limit=9)
        except StepError as e:
            out["grade_error"] = e.as_text()
    return out


def t_build_lineups(ctx, contest="se", n=3, lock=None, exclude=None, min_salary=None):
    from . import optimizer
    p = _pool(ctx)
    if p is None:
        return {"error": "No slate loaded. Click ⟳ Refresh first."}
    locks, m1 = _ids_for(p, lock or [])
    excl, m2 = _ids_for(p, exclude or [])
    if m1 or m2:
        return {"error": "Couldn't find in this slate: " + ", ".join(m1 + m2)}
    s = optimizer.preset(contest if contest in optimizer.PRESETS else "se", n_lineups=max(1, min(int(n or 3), 20)))
    o = optimizer.OptSettings(locks=locks, excludes=excl, min_salary=int(min_salary or 49000), **s)
    from . import pipeline
    out = pipeline.build_lineups(ctx.run, o)          # validated, best projection first
    r = out["result"]
    if not r.lineups:
        return {"error": "No valid lineup: " + "; ".join(r.diagnostics)}
    ctx.built = out
    idx = p.set_index("dk_id")
    return {"status": r.status, "notes": r.diagnostics, "lineups": [
        {"lineup": i, "projected": round(float(idx.loc[lu].proj.sum()), 1), "salary": int(idx.loc[lu].salary.sum()),
         "players": [f"{idx.at[x, 'pos']} {idx.at[x, 'name']} ({idx.at[x, 'team']}) ${int(idx.at[x, 'salary'])} "
                     f"proj {idx.at[x, 'proj']:.1f}" for x in lu]} for i, lu in enumerate(r.lineups, 1)]}


def t_evidence(ctx, topic=None):
    files = sorted(EVIDENCE_DIR.glob("*.json"))
    names = [f.stem for f in files]
    if not topic:
        out = {}
        for f in files:
            try:
                v = json.loads(f.read_text()).get("verdict")
            except Exception:  # noqa: BLE001
                v = None
            out[f.stem] = v if isinstance(v, str) else "(see details)"
        return {"available": out, "how": "Call again with topic=<name> for the numbers."}
    m = difflib.get_close_matches(topic, names, n=1, cutoff=0.3) or [n for n in names if topic in n]
    if not m:
        return {"error": f"No evidence file like '{topic}'. Available: {', '.join(names)}"}
    txt = (EVIDENCE_DIR / f"{m[0]}.json").read_text()
    return {"topic": m[0], "data": txt[:9000] + (" …(truncated)" if len(txt) > 9000 else "")}


def t_data_status(ctx):
    r = ctx.run
    if r is None:
        return {"error": "Nothing loaded yet — click ⟳ Refresh."}
    return {"season": r.season, "week": r.week,
            "steps": [{"step": s.step, "status": s.status, "message": s.message, "fix": s.fix} for s in r.steps]}


TOOLS = {
    "pool_players": (t_pool_players, "This week's DraftKings player pool with salaries, Combined projection and its "
                     "sources, floor/ceiling, injury status. Filter and sort to answer value/ranking questions.", {
        "position": {"type": "string", "description": "QB, RB, WR, TE, DST (comma-separated for several)"},
        "team": {"type": "string", "description": "Team abbreviation(s), e.g. DEN or DEN,KC"},
        "min_salary": {"type": "number"}, "max_salary": {"type": "number"},
        "sort_by": {"type": "string", "enum": ["proj", "value", "ceiling", "floor", "salary"]},
        "limit": {"type": "integer", "description": "Max rows (default 25, max 60)"},
        "include_out": {"type": "boolean", "description": "Include players ruled out (default false)"}}, []),
    "player": (t_player, "Everything loaded about one player: DK salary/projection sources/injury, the model's stat "
               "forecasts (yards, receptions, TD chance) and any prop lines.", {
        "name": {"type": "string"}}, ["name"]),
    "games": (t_games, "This week's games: spreads, moneylines, totals (DraftKings by default) and market-vs-model "
              "chances for each side.", {}, []),
    "props": (t_props, "Player prop lines loaded on the Props tab, with the model's forecast and chances.", {
        "player": {"type": "string"}, "game": {"type": "string", "description": "e.g. KC or DEN@KC"},
        "limit": {"type": "integer"}}, []),
    "bettingpros": (t_bettingpros, "BettingPros' props for this week (their projection, recommended side, 1–5 star "
                    "rating, win probability, EV, best price) plus their graded track record so far.", {
        "player": {"type": "string"}, "min_stars": {"type": "number"}, "market": {"type": "string"},
        "limit": {"type": "integer"}}, []),
    "saved_sets": (t_saved_sets, "List the user's saved lineup sets (id, name, week, count).", {}, []),
    "saved_set": (t_saved_set, "One saved lineup set's lineups; grade=true adds real results after the games "
                  "(actual points, each player's score, best possible lineup in hindsight).", {
        "set_id": {"type": "integer"}, "grade": {"type": "boolean"}}, ["set_id"]),
    "build_lineups": (t_build_lineups, "Build DraftKings Classic lineups from this week's pool (not saved). contest: "
                      "'cash' (highest projection, no stack), 'se' (single-entry tournament, QB stack), 'multi'.", {
        "contest": {"type": "string", "enum": ["cash", "se", "multi"]},
        "n": {"type": "integer", "description": "Lineups (default 3: best + 2 backups; max 20)"},
        "lock": {"type": "array", "items": {"type": "string"}, "description": "Player names (or team abbr. for a DST) to force in"},
        "exclude": {"type": "array", "items": {"type": "string"}, "description": "Player names to keep out"},
        "min_salary": {"type": "integer"}}, []),
    "evidence": (t_evidence, "The app's historical test results (what improved projections, what didn't, model "
                 "accuracy, props/games backtests). No topic = list with verdicts.", {
        "topic": {"type": "string"}}, []),
    "data_status": (t_data_status, "What loaded this week and any problems (each Refresh step's status and message).",
                    {}, []),
}


def tool_specs() -> list[dict]:
    return [{"name": n, "description": d, "input_schema": {"type": "object", "properties": props, "required": req}}
            for n, (_, d, props, req) in TOOLS.items()]


def run_tool(ctx: Context, name: str, args: dict) -> str:
    fn = TOOLS.get(name, (None,))[0]
    if fn is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        res = fn(ctx, **(args or {}))
    except TypeError as e:
        res = {"error": f"bad arguments: {e}"}
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(res, default=str)[:30000]


# ------------------------------------------------------------------ API
def _post(key: str, body: dict, session=None) -> dict:
    s = session or requests
    hdr = {"x-api-key": key, "anthropic-version": VERSION, "content-type": "application/json"}
    last = None
    for attempt in range(3):
        try:
            r = s.post(API, headers=hdr, json=body, timeout=180)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        try:
            msg = r.json().get("error", {}).get("message", r.text[:300])
        except Exception:  # noqa: BLE001
            msg = r.text[:300]
        if "workspace" in msg.lower():
            raise StepError(STEP, "This API key isn't tied to a workspace.", "Keys that cover every workspace need extra "
                            "setup the app doesn't use.", "In the Claude Console create a new key and pick the Default "
                            "workspace for it, then paste that key in Settings.", detail=msg)
        if r.status_code in (401, 403):
            raise StepError(STEP, "Anthropic refused the API key.", "The key is wrong, revoked, or the account has no credit.",
                            "Check it at platform.claude.com/settings/keys, paste it in Settings → Anthropic API key, Save.", detail=msg)
        if r.status_code == 400 and ("thinking" in msg or "output_config" in msg or "effort" in msg) and "thinking" in body:
            body.pop("thinking", None)          # a model without these options — drop them for the rest of this answer
            body.pop("output_config", None)
            continue
        if r.status_code == 400 and "credit" in msg.lower():
            raise StepError(STEP, "Your Anthropic account is out of credit.", "", "Add credit at platform.claude.com/settings/billing.", detail=msg)
        if r.status_code in (429, 500, 502, 503, 529):
            last = f"HTTP {r.status_code}: {msg}"
            time.sleep(3 * (attempt + 1))
            continue
        raise StepError(STEP, f"The Anthropic API returned an error (HTTP {r.status_code}).", "", "Try again; if it repeats, "
                        "use Copy diagnostics on Data status.", detail=msg)
    raise StepError(STEP, "Couldn't reach Claude (busy or no connection).", "", "Wait a minute and ask again.", detail=last or "")


def ask(key: str, question: str, ctx: Context, history: list[tuple[str, str]] | None = None,
        model: str = DEFAULT_MODEL, progress=lambda m: None, session=None) -> dict:
    """Answer one question. history = earlier (question, answer) pairs from this session.
    Returns {"answer", "tools_used", "cost", "tokens", "built"}."""
    if not key:
        raise StepError(STEP, "No Anthropic API key set.", "", "Get one at platform.claude.com/settings/keys, paste it in "
                        "Settings → Anthropic API key (sidebar), and click Save settings.")
    msgs = []
    for q, a in (history or [])[-6:]:              # earlier turns as plain text (keeps requests small and valid)
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a or "(no answer)"}]
    msgs.append({"role": "user", "content": question})
    system = SYSTEM.format(season=ctx.season or "?", week=ctx.week or "?")
    body = {"model": model, "max_tokens": 4000, "system": system, "tools": tool_specs(),
            "tool_choice": {"type": "auto"}, "messages": msgs,
            "thinking": {"type": "between_tools"}, "output_config": {"effort": "medium"}}
    used, tin, tout = [], 0, 0
    ctx.built = None
    for _ in range(MAX_ROUNDS):
        resp = _post(key, body, session)
        u = resp.get("usage", {})
        tin += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0) or 0) + int(u.get("cache_creation_input_tokens", 0) or 0)
        tout += int(u.get("output_tokens", 0))
        content = resp.get("content", [])
        calls = [c for c in content if c.get("type") == "tool_use"]
        if resp.get("stop_reason") != "tool_use" or not calls:
            text = "\n\n".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
            break
        body["messages"] = body["messages"] + [{"role": "assistant", "content": content}]
        results = []
        for c in calls:
            progress(f"Looking up: {c['name'].replace('_', ' ')}")
            used.append(c["name"])
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": run_tool(ctx, c["name"], c.get("input"))})
        body["messages"] = body["messages"] + [{"role": "user", "content": results}]
    else:
        text = "I couldn't finish this one within the lookup limit — try a narrower question."
    _, pin, pout = MODELS.get(model, ("", 2.0, 10.0))
    return {"answer": text or "(no answer returned)", "tools_used": used, "tokens": (tin, tout),
            "cost": round(tin / 1e6 * pin + tout / 1e6 * pout, 4), "built": ctx.built}
