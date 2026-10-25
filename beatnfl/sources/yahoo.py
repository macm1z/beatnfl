"""Yahoo Fantasy Sports — read-only access to your own leagues and rosters.

One-time setup (in the app's My Yahoo team tab):
  1. Create an app at https://developer.yahoo.com/apps/create/
       Redirect URI: https://localhost:8000 · API permissions: Fantasy Sports → Read
  2. Paste the Client ID and Client Secret into Settings.
  3. Click the sign-in link and approve. Yahoo then sends the browser to https://localhost:8000/?code=…
     (the page itself won't load — that's expected); copy the whole address and paste it into the app.
The Redirect URI the app sends must match the one in your Yahoo app EXACTLY, or Yahoo shows
"Something went wrong" — it is set on the My Yahoo team tab (older apps made with "oob" keep working).

Tokens are stored only in ~/BeatNFL/yahoo_token.json (owner-only permissions) and refresh
automatically. Nothing here can change a lineup — the app only asks Yahoo for read access.
"""

from __future__ import annotations

import json
import os
import time
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from ..config import HOME, ensure_dirs
from ..errors import StepError
from ..names import team as norm_team

STEP = "Yahoo Fantasy"
AUTH = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN = "https://api.login.yahoo.com/oauth2/get_token"
API = "https://fantasysports.yahooapis.com/fantasy/v2"
TOKEN_PATH = HOME / "yahoo_token.json"
DEFAULT_REDIRECT = "https://localhost:8000"
YAHOO_TEAM = {"JAC": "JAX", "WSH": "WAS", "LAR": "LA", "KAN": "KC", "SFO": "SF", "GNB": "GB", "NOR": "NO", "TAM": "TB"}


def _need(client_id: str, secret: str):
    if not client_id or not secret:
        raise StepError(STEP, "Yahoo keys are not set.",
                        "The app needs your Yahoo app's Client ID and Client Secret to read your league.",
                        "Create an app at developer.yahoo.com/apps/create (Redirect URI https://localhost:8000, "
                        "Fantasy Sports: Read), then paste both values into Settings.")


def auth_url(client_id: str, redirect_uri: str = DEFAULT_REDIRECT) -> str:
    return AUTH + "?" + urlencode({"client_id": client_id.strip(), "redirect_uri": (redirect_uri or DEFAULT_REDIRECT).strip(),
                                   "response_type": "code", "language": "en-us"})


def extract_code(pasted: str) -> str:
    """The authorization code from whatever was pasted: the code itself, or the whole address Yahoo
    redirected to (https://localhost:8000/?code=abc…)."""
    t = (pasted or "").strip()
    if "code=" in t:
        q = parse_qs(urlparse(t if "://" in t else "https://x/?" + t.split("?", 1)[-1]).query)
        if q.get("code"):
            return q["code"][0].strip()
    return t


def _save(tok: dict):
    ensure_dirs()
    tok["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(tok))
    try:
        os.chmod(TOKEN_PATH, 0o600)
    except OSError:
        pass


def _token_request(client_id: str, secret: str, data: dict) -> dict:
    r = requests.post(TOKEN, data=data, auth=(client_id, secret), timeout=20)
    if r.status_code != 200:
        why = "Yahoo rejected the code or keys."
        try:
            err = r.json()
            why = f"Yahoo said: {err.get('error_description') or err.get('error')}"
        except ValueError:
            pass
        raise StepError(STEP, "Could not sign in to Yahoo.", why,
                        "Check the Client ID/Secret in Settings, then click the sign-in link again and paste the new code "
                        "(codes expire after a few minutes and work once).", detail=r.text[:200])
    return r.json()


def connect(client_id: str, secret: str, code: str, redirect_uri: str = DEFAULT_REDIRECT) -> None:
    _need(client_id, secret)
    ru = (redirect_uri or DEFAULT_REDIRECT).strip()
    code = extract_code(code)
    if not code or code.startswith("http") or not any(ch.isalnum() for ch in code):
        raise StepError(STEP, "No sign-in code found in what was pasted.", "",
                        "After approving on Yahoo, copy the whole address from the browser's address bar "
                        "(it contains ?code=…) and paste it here.")
    tok = _token_request(client_id.strip(), secret.strip(), {"grant_type": "authorization_code", "redirect_uri": ru, "code": code})
    tok["redirect_uri"] = ru
    _save(tok)


def connected() -> bool:
    return TOKEN_PATH.exists()


def disconnect() -> None:
    TOKEN_PATH.unlink(missing_ok=True)


def _access(client_id: str, secret: str) -> str:
    _need(client_id, secret)
    if not TOKEN_PATH.exists():
        raise StepError(STEP, "Yahoo is not connected yet.", "", "Use the sign-in link in the My Yahoo team tab.")
    tok = json.loads(TOKEN_PATH.read_text())
    if time.time() - tok.get("obtained_at", 0) > tok.get("expires_in", 3600) - 120:
        new = _token_request(client_id, secret, {"grant_type": "refresh_token", "redirect_uri": tok.get("redirect_uri", "oob"),
                                                 "refresh_token": tok["refresh_token"]})
        new.setdefault("refresh_token", tok["refresh_token"])
        new.setdefault("redirect_uri", tok.get("redirect_uri", "oob"))
        _save(new)
        tok = new
    return tok["access_token"]


def get(path: str, client_id: str, secret: str) -> dict:
    tok = _access(client_id, secret)
    url = f"{API}/{path}" + ("&" if "?" in path else "?") + "format=json"
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=20)
    if r.status_code == 401:
        raise StepError(STEP, "Yahoo says the sign-in is no longer valid.", "The access was revoked or expired.",
                        "Click Disconnect, then sign in again.")
    if r.status_code != 200:
        raise StepError(STEP, f"Yahoo returned an error (HTTP {r.status_code}).", r.text[:160],
                        "Try again in a minute; check Data status if it repeats.")
    return r.json()


# ---------------------------------------------------------------- Yahoo's nested JSON
def merge(x) -> dict:
    """Yahoo returns objects as lists of one-key dicts (sometimes nested lists). Flatten into one dict."""
    out = {}
    if isinstance(x, list):
        for it in x:
            if isinstance(it, (list, dict)):
                out.update(merge(it) if isinstance(it, list) else it)
    elif isinstance(x, dict):
        out.update(x)
    return out


def find_all(obj, key: str) -> list:
    """Every value stored under `key` anywhere in the document."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(find_all(v, key))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(find_all(v, key))
    return found


def parse_teams(doc: dict) -> list[dict]:
    teams = []
    for t in find_all(doc, "team"):
        m = merge(t)
        if "team_key" in m:
            teams.append({"team_key": m["team_key"], "name": m.get("name", ""),
                          "league_key": m["team_key"].rsplit(".t.", 1)[0]})
    seen, out = set(), []
    for t in teams:
        if t["team_key"] not in seen:
            seen.add(t["team_key"])
            out.append(t)
    return out


def parse_roster(doc: dict) -> list[dict]:
    rows = []
    for p in find_all(doc, "player"):
        m = merge(p)
        if "player_key" not in m:
            continue
        sel = merge(m.get("selected_position", []))
        name = m.get("name", {})
        rows.append({
            "player_key": m["player_key"],
            "name": name.get("full", "") if isinstance(name, dict) else str(name),
            "team": norm_team(YAHOO_TEAM.get(str(m.get("editorial_team_abbr", "")).upper(),
                                             str(m.get("editorial_team_abbr", "")).upper())),
            "pos": m.get("display_position", ""),
            "eligible": [e.get("position") for e in m.get("eligible_positions", []) if isinstance(e, dict)],
            "slot": sel.get("position", ""),
            "status": m.get("status", "") or "",
            "injury_note": m.get("injury_note", "") or "",
            "bye": (m.get("bye_weeks") or {}).get("week", "") if isinstance(m.get("bye_weeks"), dict) else "",
        })
    return rows


def parse_roster_slots(doc: dict) -> list[tuple[str, int]]:
    slots = []
    for rp in find_all(doc, "roster_position"):
        if isinstance(rp, dict) and rp.get("position") and str(rp.get("is_starting_position", 1)) != "0" \
                and rp["position"] not in ("BN", "IR", "IR+", "NA"):
            slots.append((rp["position"], int(rp.get("count", 1))))
    return slots


def my_teams(cid, sec) -> list[dict]:
    return parse_teams(get("users;use_login=1/games;game_keys=nfl/teams", cid, sec))


def roster(team_key: str, week: int, cid, sec) -> list[dict]:
    return parse_roster(get(f"team/{team_key}/roster;week={week}", cid, sec))


def roster_slots(league_key: str, cid, sec) -> list[tuple[str, int]]:
    return parse_roster_slots(get(f"league/{league_key}/settings", cid, sec))


# ---------------------------------------------------------------- start / sit
FLEX = {"W/R/T": {"WR", "RB", "TE"}, "W/R": {"WR", "RB"}, "W/T": {"WR", "TE"}, "R/T": {"RB", "TE"},
        "Q/W/R/T": {"QB", "WR", "RB", "TE"}, "OP": {"QB", "WR", "RB", "TE"}}


def suggest(rows: list[dict], slots: list[tuple[str, int]]) -> list[dict]:
    """Fill starting slots by projection: fixed positions first, flex slots last.
    Players ruled out (O, IR, PUP, SUSP) or on bye are never suggested."""
    avail = [r for r in rows if r.get("proj") is not None and r["status"] not in ("O", "IR", "PUP-R", "SUSP", "NA")
             and not r.get("on_bye")]
    avail.sort(key=lambda r: -r["proj"])
    used, out = set(), []
    fixed = [(s, n) for s, n in slots if s not in FLEX]
    flex = [(s, n) for s, n in slots if s in FLEX]
    for s, n in fixed + flex:
        ok = FLEX.get(s, {s})
        for _ in range(n):
            pick = next((r for r in avail if r["player_key"] not in used and
                         (set(r.get("eligible") or [r["pos"]]) & ok or r["pos"] in ok)), None)
            if pick:
                used.add(pick["player_key"])
            out.append({"slot": s, "player": pick["name"] if pick else "— nobody projected —",
                        "proj": pick["proj"] if pick else None, "player_key": pick["player_key"] if pick else None})
    return out


# ---------------------------------------------------------------- league-wide player info (injuries, % rostered)
STATUS_MAP = {"Q": "Q", "D": "D", "O": "O", "IR": "IR", "IR-R": "IR", "IR-NR": "IR", "PUP-R": "O", "PUP-P": "O",
              "SUSP": "O", "NFI-R": "O", "NA": "O", "COVID-19": "O", "P": "P"}


def parse_players(doc: dict) -> list[dict]:
    rows = []
    for p in find_all(doc, "player"):
        m = merge(p)
        if "player_key" not in m:
            continue
        name = m.get("name", {})
        po = merge(m.get("percent_owned", []))
        pos = m.get("display_position", "") or m.get("primary_position", "")
        rows.append({
            "name": name.get("full", "") if isinstance(name, dict) else str(name),
            "team": norm_team(YAHOO_TEAM.get(str(m.get("editorial_team_abbr", "")).upper(),
                                             str(m.get("editorial_team_abbr", "")).upper())),
            "pos": "DST" if pos == "DEF" else pos.split(",")[0],
            "status": STATUS_MAP.get(str(m.get("status", "") or "").upper(), ""),
            "status_raw": m.get("status", "") or "", "injury_note": m.get("injury_note", "") or "",
            "pct_rostered": float(po["value"]) if po.get("value") not in (None, "") else None,
            "pct_change": float(po["delta"]) if po.get("delta") not in (None, "") else None,
        })
    return rows


def league_players(league_key: str, cid, sec, pages: int = 16) -> list[dict]:
    """Top ~400 NFL players in Yahoo's overall rank with injury status and % rostered (16 small requests)."""
    out = []
    for i in range(pages):
        doc = get(f"league/{league_key}/players;sort=OR;start={i * 25};count=25/percent_owned", cid, sec)
        rows = parse_players(doc)
        out.extend(rows)
        if len(rows) < 25:
            break
    return out


def first_league(cid, sec) -> str | None:
    teams = my_teams(cid, sec)
    return teams[0]["league_key"] if teams else None
