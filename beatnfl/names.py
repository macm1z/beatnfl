"""Identity helpers: team codes and player-name matching keys.

Names are never the primary key — they are only used to *match* rows from sources
that do not share an ID, and every match carries the team and position too.
"""

from __future__ import annotations

import re
import unicodedata

TEAM_ALIASES = {
    "LAR": "LA", "STL": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "OAK": "LV", "ARZ": "ARI",
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SD": "LAC", "KAN": "KC", "GNB": "GB", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "SL": "LA",
}
TEAMS = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND",
         "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",
         "TB", "TEN", "WAS"}

# DK-facing codes (used only when writing human-readable output)
TO_DK = {"LA": "LAR"}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
NICK = {"hollywood brown": "marquise brown", "gabe davis": "gabriel davis", "chig okonkwo": "chigoziem okonkwo",
        "josh palmer": "joshua palmer", "kenneth walker": "kenneth walker", "deebo samuel": "deebo samuel",
        "mitch trubisky": "mitchell trubisky", "ken walker": "kenneth walker", "bam knight": "zonovan knight",
        "dj moore": "dj moore", "aj brown": "aj brown", "tank dell": "nathaniel dell", "scotty miller": "scott miller"}


def team(code) -> str:
    if code is None:
        return ""
    c = str(code).strip().upper()
    return TEAM_ALIASES.get(c, c)


def norm_name(name) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("’", "'")
    s = _SUFFIX.sub("", s)
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return NICK.get(s, s)


def pos(p) -> str:
    p = str(p or "").upper().strip()
    return {"DEF": "DST", "D/ST": "DST", "D": "DST", "DST": "DST", "FB": "RB", "HB": "RB"}.get(p, p)


def match_key(name, position, team_code) -> str:
    p = pos(position)
    if p == "DST":
        return f"DST|{team(team_code)}"
    return f"{p}|{team(team_code)}|{norm_name(name)}"


def loose_key(name, position) -> str:
    """Fallback when a source has a stale team (recent trade/signing)."""
    return f"{pos(position)}|{norm_name(name)}"
