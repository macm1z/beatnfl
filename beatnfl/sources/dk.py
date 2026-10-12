"""DraftKings slate retrieval.

Three routes, tried in this order, all ending in the same validated table:
  1. JSON   lobby/getcontests -> draftgroups/{id}/draftables     (automatic)
  2. CSV    lineup/getavailableplayerscsv?draftGroupId=...       (automatic, same file as DK's "Export to CSV")
  3. File   the official DKSalaries.csv dropped into ~/BeatNFL/inbox or uploaded in the app (manual, always works)

Automatic routes are community-documented, not official. They work from a home
connection far more often than from cloud servers (DK blocks many datacenter IPs).
When the JSON route is used, its export IDs are marked "unverified" until an
official CSV for the same slate has been compared (see verify_against_csv).
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..errors import StepError
from ..names import match_key, pos as norm_pos, team as norm_team
from ..net import fetch, fetch_json

STEP = "Load DraftKings slate"
LOBBY = "https://www.draftkings.com/lobby/getcontests"
DRAFTABLES = "https://api.draftkings.com/draftgroups/v1/draftgroups/{dg}/draftables"
CSV_URL = "https://www.draftkings.com/lineup/getavailableplayerscsv"
ET = ZoneInfo("America/New_York")
CLASSIC_CONTEST_TYPE = 21
CLASSIC_POSITIONS = {"QB", "RB", "WR", "TE", "DST"}
SLATE_COLUMNS = ["dk_id", "name", "pos", "team", "opp", "game", "start_utc", "salary", "dk_status", "key"]


@dataclass
class SlateInfo:
    draft_group_id: str
    label: str
    start_utc: str
    game_count: int
    contest_type_id: int
    tag: str = ""
    suffix: str = ""

    @property
    def start_et(self) -> str:
        try:
            return datetime.fromisoformat(self.start_utc.replace("Z", "+00:00")).astimezone(ET).strftime("%a %b %-d %-I:%M %p ET")
        except Exception:
            return self.start_utc


# ---------------------------------------------------------------- discovery
def _parse_dk_date(v) -> str:
    if not v:
        return ""
    s = str(v)
    m = re.search(r"/Date\((\d+)", s)
    if m:
        return datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    s = re.sub(r"\.(\d{6})\d*", r".\1", s)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return s


def parse_lobby(payload: dict) -> list[SlateInfo]:
    if not isinstance(payload, dict) or not ({"DraftGroups", "Contests"} & set(payload)):
        raise StepError(STEP, "DraftKings lobby data was not in the expected format.",
                        "DraftKings changed its lobby feed or returned an error page.",
                        "Use the CSV route: on DraftKings open any NFL Classic contest → 'Export to CSV' → drop DKSalaries.csv into the inbox.")
    out: dict[str, SlateInfo] = {}
    for g in payload.get("DraftGroups") or []:
        ct = int(g.get("ContestTypeId") or 0)
        gt = str(g.get("GameType") or "")
        if ct != CLASSIC_CONTEST_TYPE and "classic" not in gt.lower():
            continue
        dg = str(g.get("DraftGroupId"))
        suffix = (g.get("ContestStartTimeSuffix") or "").strip()
        start = _parse_dk_date(g.get("StartDate") or g.get("StartDateEst"))
        out[dg] = SlateInfo(dg, "", start, int(g.get("GameCount") or 0), ct, str(g.get("DraftGroupTag") or ""), suffix)
    if not out:  # fall back to contests list
        for c in payload.get("Contests") or []:
            if str(c.get("gameType", "")).lower() != "classic":
                continue
            dg = str(c.get("dg"))
            out.setdefault(dg, SlateInfo(dg, "", _parse_dk_date(c.get("sd")), 0, CLASSIC_CONTEST_TYPE))
    for s in out.values():
        s.label = f"{s.start_et} · {s.game_count or '?'} games{(' · ' + s.suffix.strip('() ')) if s.suffix else ''} · DG {s.draft_group_id}"
    return sorted(out.values(), key=lambda s: (s.start_utc, -s.game_count))


def pick_main_slate(slates: list[SlateInfo]) -> SlateInfo:
    """Main slate = featured Classic group with no time suffix and the most games; ties → earliest."""
    if not slates:
        raise StepError(STEP, "No NFL Classic slates are open on DraftKings right now.",
                        "Salaries for next week usually post Monday night or Tuesday.",
                        "Try again later, or load DKSalaries.csv manually.")
    main = [s for s in slates if not s.suffix] or slates
    featured = [s for s in main if s.tag.lower() == "featured"] or main
    return sorted(featured, key=lambda s: (-s.game_count, s.start_utc))[0]


def discover(offline=False) -> list[SlateInfo]:
    data, _ = fetch_json(LOBBY, params={"sport": "NFL"}, step=STEP, max_age_h=0.25, offline=offline)
    return parse_lobby(data)


# ---------------------------------------------------------------- draftables JSON
def parse_draftables(payload: dict) -> tuple[pd.DataFrame, list[str]]:
    warns: list[str] = []
    if not isinstance(payload, dict) or "draftables" not in payload:
        raise StepError(STEP, "DraftKings player data was not in the expected format.",
                        "The draftables feed changed or returned an error.",
                        "Use the CSV route (Export to CSV → inbox).")
    comps = {}
    for c in payload.get("competitions") or []:
        comps[c.get("competitionId")] = c
    rows = []
    for d in payload["draftables"]:
        if d.get("isDisabled"):
            continue
        p = norm_pos(d.get("position"))
        if p not in CLASSIC_POSITIONS:
            continue
        comp = d.get("competition") or comps.get((d.get("competition") or {}).get("competitionId")) or {}
        if not comp and d.get("competitions"):
            comp = d["competitions"][0]
        gname = str(comp.get("name") or "")
        teams = re.findall(r"[A-Z]{2,3}", gname)
        tm = norm_team(d.get("teamAbbreviation"))
        opp = ""
        if len(teams) == 2:
            a, h = norm_team(teams[0]), norm_team(teams[1])
            opp = h if tm == a else a
        name = d.get("displayName") or f"{d.get('firstName', '')} {d.get('lastName', '')}".strip()
        rows.append({
            "dk_id": str(d.get("draftableId")), "player_id": str(d.get("playerId") or d.get("playerDkId") or ""),
            "slot_id": d.get("rosterSlotId"), "name": name, "pos": p, "team": tm, "opp": opp,
            "game": f"{teams[0]}@{teams[1]}" if len(teams) == 2 else gname,
            "start_utc": _parse_dk_date(comp.get("startTime")), "salary": d.get("salary"),
            "dk_status": str(d.get("status") or "None"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise StepError(STEP, "The slate came back with no players.", "The draft group may be closed or empty.",
                        "Pick a different slate or load DKSalaries.csv.")
    # Classic lists a player once per eligible roster slot (e.g. WR and FLEX).
    n_ids = df.groupby("player_id")["dk_id"].nunique()
    if (n_ids > 1).any():
        warns.append(f"{int((n_ids > 1).sum())} players have more than one draftable ID; keeping the lowest-slot ID. "
                     "Export IDs stay 'unverified' until compared with the official CSV.")
    df = df.sort_values(["player_id", "slot_id"]).drop_duplicates("player_id", keep="first")
    df["key"] = [match_key(n, p, t) for n, p, t in zip(df.name, df.pos, df.team)]
    return df.drop(columns=["player_id", "slot_id"]), warns


# ---------------------------------------------------------------- official CSV
CSV_ALIASES = {
    "position": "pos", "name + id": "name_id", "name": "name", "id": "dk_id", "roster position": "roster_pos",
    "salary": "salary", "game info": "game_info", "teamabbrev": "team", "team": "team", "avgpointspergame": "avg",
}


def parse_salary_csv(raw: bytes | str) -> tuple[pd.DataFrame, list[str]]:
    """Parse DK's DKSalaries.csv (Classic). Handles BOM, CRLF, quoted commas, blank lines."""
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw.lstrip("﻿")
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise StepError(STEP, "The salary file is empty.", "", "Re-export from DraftKings and try again.")
    hdr = [CSV_ALIASES.get(h.strip().lower(), h.strip().lower()) for h in rows[0]]
    need = {"dk_id", "name", "salary", "team", "game_info"}
    missing = need - set(hdr) - ({"name"} if "name_id" in hdr else set())
    if missing:
        raise StepError(STEP, f"This does not look like a DraftKings salary file (missing: {', '.join(sorted(missing))}).",
                        "It may be a lineup-entry file or a different site's export.",
                        "On DraftKings open an NFL Classic contest → 'Export to CSV' (the players file).")
    df = pd.DataFrame(rows[1:], columns=hdr[: len(rows[1])] if rows[1:] else hdr)
    warns, rejected = [], []
    if "roster_pos" in df and df["roster_pos"].str.contains("CPT").any():
        raise StepError(STEP, "This is a Showdown (Captain mode) salary file.",
                        "This version builds Classic lineups only.",
                        "Export the CSV from a Classic contest on the same slate.")
    df["dk_id"] = df["dk_id"].astype(str).str.strip()
    df["salary_num"] = pd.to_numeric(df["salary"], errors="coerce")
    bad = df[df.salary_num.isna() | (df.salary_num <= 0) | ~df.dk_id.str.fullmatch(r"\d+")]
    for _, r in bad.iterrows():
        rejected.append(f"{r.get('name')}: invalid salary or ID")
    df = df.drop(bad.index)
    gi = df["game_info"].str.extract(r"(?P<away>[A-Z]{2,3})@(?P<home>[A-Z]{2,3})\s+(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{1,2}:\d{2}[AP]M)")
    df["team"] = df["team"].map(norm_team)
    df["away"], df["home"] = gi["away"].map(norm_team), gi["home"].map(norm_team)
    df["opp"] = df.apply(lambda r: r.home if r.team == r.away else r.away, axis=1)
    df["game"] = gi["away"].fillna("") + "@" + gi["home"].fillna("")

    def to_utc(d, t):
        try:
            return datetime.strptime(f"{d} {t}", "%m/%d/%Y %I:%M%p").replace(tzinfo=ET).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return ""
    df["start_utc"] = [to_utc(d, t) for d, t in zip(gi["date"], gi["time"])]
    df["pos"] = df["pos"].map(norm_pos)
    df = df[df.pos.isin(CLASSIC_POSITIONS)]
    dup = df[df.duplicated("dk_id", keep=False)]
    if not dup.empty:
        conflict = dup.groupby("dk_id").agg({"salary_num": "nunique", "name": "nunique"})
        if (conflict > 1).any(axis=None):
            raise StepError(STEP, "The salary file has conflicting rows for the same player ID.",
                            "The file was edited or two exports were merged.", "Re-export a fresh file from DraftKings.")
        df = df.drop_duplicates("dk_id")
    out = pd.DataFrame({
        "dk_id": df.dk_id, "name": df.name.str.strip(), "pos": df.pos, "team": df.team, "opp": df.opp,
        "game": df.game, "start_utc": df.start_utc, "salary": df.salary_num.astype(int), "dk_status": "None",
    })
    out["key"] = [match_key(n, p, t) for n, p, t in zip(out.name, out.pos, out.team)]
    if rejected:
        warns.append(f"{len(rejected)} rows rejected: " + "; ".join(rejected[:5]))
    return out.reset_index(drop=True), warns


def validate_slate(df: pd.DataFrame) -> list[str]:
    """Hard checks before a slate is usable. Raises on fatal problems, returns warnings."""
    probs = []
    if df["dk_id"].duplicated().any():
        probs.append("duplicate player IDs")
    if (df.salary < 2000).any() or (df.salary > 10500).any():
        probs.append("salaries outside DK's normal $2,000–$10,500 range")
    counts = df.pos.value_counts()
    for p, need in {"QB": 2, "RB": 3, "WR": 4, "TE": 2, "DST": 2}.items():
        if counts.get(p, 0) < need:
            probs.append(f"too few {p} ({counts.get(p, 0)})")
    games = df.game.nunique()
    if games < 2:
        probs.append("fewer than 2 games — Classic needs players from 2+ games")
    if probs:
        raise StepError(STEP, "The slate failed validation: " + "; ".join(probs) + ".",
                        "The source returned incomplete or mixed data.",
                        "Reload the slate; if it repeats, load DKSalaries.csv for the slate you want.")
    warns = []
    if (df.team == "").any():
        warns.append(f"{int((df.team == '').sum())} players missing a team")
    return warns


def content_hash(df: pd.DataFrame) -> str:
    cols = ["dk_id", "name", "pos", "team", "salary", "game"]
    return hashlib.sha256(df[cols].sort_values("dk_id").to_csv(index=False).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- orchestrator
@dataclass
class LoadedSlate:
    players: pd.DataFrame
    slate: SlateInfo | None
    route: str              # "json" | "csv-endpoint" | "file"
    ids_verified: bool
    warnings: list[str]
    fetched_note: str
    content_hash: str


def load_from_file(path: Path | str | bytes, label: str = "") -> LoadedSlate:
    raw = path if isinstance(path, (bytes, str)) and not (isinstance(path, str) and Path(path).exists()) else Path(path).read_bytes()
    df, warns = parse_salary_csv(raw)
    warns += validate_slate(df)
    first = df.start_utc[df.start_utc != ""].min() if (df.start_utc != "").any() else ""
    s = SlateInfo("file", label or f"DKSalaries.csv · {df.game.nunique()} games", first, df.game.nunique(), CLASSIC_CONTEST_TYPE)
    return LoadedSlate(df, s, "file", True, warns, "official CSV", content_hash(df))


def load_auto(draft_group_id: str | None = None, offline=False) -> LoadedSlate:
    """Try JSON, then the CSV endpoint. Raises StepError listing every route tried."""
    tried = []
    try:
        slates = discover(offline=offline)
        slate = next((s for s in slates if s.draft_group_id == str(draft_group_id)), None) if draft_group_id else pick_main_slate(slates)
        if slate is None:
            raise StepError(STEP, f"Draft group {draft_group_id} is not open.", "", "Pick a slate from the list.")
    except StepError as e:
        if not draft_group_id:
            raise StepError(STEP, "Could not find this week's slate automatically.", e.why or e.what,
                            "Load DKSalaries.csv: DraftKings → any NFL Classic contest → Export to CSV → drop it in ~/BeatNFL/inbox.",
                            detail=e.detail)
        slate, tried = SlateInfo(str(draft_group_id), f"DG {draft_group_id}", "", 0, CLASSIC_CONTEST_TYPE), [f"lobby: {e.what}"]

    dg = slate.draft_group_id
    try:
        data, meta = fetch_json(DRAFTABLES.format(dg=dg), params={"format": "json"}, step=STEP, max_age_h=0.5, offline=offline)
        df, warns = parse_draftables(data)
        warns += validate_slate(df)
        if meta.stale:
            warns.insert(0, "STALE: " + meta.note)
        multi = any("more than one draftable ID" in w for w in warns)
        return LoadedSlate(df, slate, "json", not multi, warns,
                           "cached" if meta.from_cache else "live", content_hash(df))
    except StepError as e:
        tried.append(f"draftables JSON: {e.what} ({e.detail[:80]})")
    try:
        body, meta = fetch(CSV_URL, params={"contestTypeId": CLASSIC_CONTEST_TYPE, "draftGroupId": dg}, step=STEP,
                           max_age_h=0.5, offline=offline)
        df, warns = parse_salary_csv(body)
        warns += validate_slate(df)
        if meta.stale:
            warns.insert(0, "STALE: " + meta.note)
        return LoadedSlate(df, slate, "csv-endpoint", True, warns, "cached" if meta.from_cache else "live", content_hash(df))
    except StepError as e:
        tried.append(f"CSV endpoint: {e.what} ({e.detail[:80]})")
    raise StepError(STEP, "Automatic DraftKings download failed on every route.",
                    "Tried → " + " | ".join(tried),
                    "Load DKSalaries.csv manually: DraftKings → any NFL Classic contest on this slate → Export to CSV → "
                    "drop it in ~/BeatNFL/inbox (the app picks up the newest file).")


def verify_against_csv(auto: pd.DataFrame, official: pd.DataFrame) -> tuple[bool, list[str]]:
    """Compare an automatically loaded slate to the official CSV (IDs, salaries, positions, games)."""
    a, o = auto.set_index("dk_id"), official.set_index("dk_id")
    issues = []
    only_a, only_o = set(a.index) - set(o.index), set(o.index) - set(a.index)
    if only_a:
        issues.append(f"{len(only_a)} IDs only in automatic data")
    if only_o:
        issues.append(f"{len(only_o)} IDs only in official CSV")
    common = a.index.intersection(o.index)
    for col in ("salary", "pos", "team"):
        diff = (a.loc[common, col].astype(str) != o.loc[common, col].astype(str)).sum()
        if diff:
            issues.append(f"{diff} {col} mismatches")
    return not issues, issues
