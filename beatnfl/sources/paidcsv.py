"""Import projections exported from a paid site (FantasyPros, ETR, 4for4, RotoGrinders,
Stokastic, etc.). Columns are detected automatically; the detected mapping is shown
to the user so nothing is guessed silently.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from ..errors import StepError
from ..names import loose_key, match_key, pos as norm_pos, team as norm_team

STEP = "Paid projections import"
NAME = ["player", "name", "player name", "playername", "full name"]
POS = ["pos", "position", "dk position", "roster position"]
TEAM = ["team", "tm", "teamabbrev", "team abbrev"]
ID = ["dk id", "dkid", "draftkings id", "dfs id", "id", "draftableid", "player id (dk)"]
PROJ_PREF = [r"^dk.*(proj|fpts|fp|points|pts)", r"draftkings.*(proj|fpts|points|pts)", r"^(dk )?proj(ection)?(s)?$",
             r"^fpts$", r"^ffpts$", r"^proj", r"^fp$", r"^projected points$", r"^points$", r"^pts$", r"^median$", r"^fpts proj"]


def _find(cols, opts):
    low = {c.lower().strip(): c for c in cols}
    for o in opts:
        if o in low:
            return low[o]
    return None


def parse(raw: bytes | str, filename: str = "") -> tuple[pd.DataFrame, dict]:
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    text = text.strip("\ufeff\n\r ")
    first = text.splitlines()[0] if text else ""
    sep = "\t" if "\t" in first else ("," if "," in first else r"\s{2,}")  # pasted tables arrive tab-separated
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        raise StepError(STEP, f"Could not read {filename or 'the projection file'} as CSV.", str(e)[:120],
                        "Export the projections as CSV (not Excel) and try again.")
    cols = list(df.columns)
    name_c, pos_c, team_c, id_c = _find(cols, NAME), _find(cols, POS), _find(cols, TEAM), _find(cols, ID)
    proj_c = None
    for pat in PROJ_PREF:
        hit = [c for c in cols if re.search(pat, c.lower().strip()) and "own" not in c.lower() and "salary" not in c.lower()]
        if hit:
            proj_c = hit[0]
            break
    if not name_c or not proj_c:
        raise StepError(STEP, "Could not find the player-name or DraftKings-projection column.",
                        f"Columns in the file: {', '.join(cols[:15])}",
                        "Make sure the export is DraftKings scoring; rename the projection column to 'DK Proj' if needed.")
    out = pd.DataFrame({
        "name": df[name_c].astype(str).str.replace(r"\s*\(.*\)$", "", regex=True).str.strip(),
        "pos": df[pos_c].map(norm_pos) if pos_c else "",
        "team": df[team_c].map(norm_team) if team_c else "",
        "proj_paid": pd.to_numeric(df[proj_c], errors="coerce"),
        "dk_id": df[id_c].astype(str).str.replace(r"\.0$", "", regex=True) if id_c else "",
    })
    out = out[out.proj_paid.notna()]
    out["key"] = [match_key(n, p, t) for n, p, t in zip(out.name, out.pos, out.team)]
    out["lkey"] = [loose_key(n, p) for n, p in zip(out.name, out.pos)]
    mapping = {"name": name_c, "position": pos_c, "team": team_c, "projection": proj_c, "dk_id": id_c, "rows": len(out)}
    return out, mapping


def newest_in(folder: Path) -> Path | None:
    files = [p for p in folder.glob("*.csv") if not p.name.lower().startswith("dksalaries")]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None
