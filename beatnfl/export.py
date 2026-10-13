"""Exports: DraftKings bulk-upload CSV (IDs only, DK's header with repeated slot names)
and a readable CSV with names, salaries and projections. Every DK file is re-parsed
and re-validated after writing; a file that fails is deleted, never handed over.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import EXPORTS, ensure_dirs
from .optimizer import SLOTS, validate


def _stamp(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")[:40]
    return f"{datetime.now():%Y%m%d_%H%M}_{safe}"


def dk_csv_text(lineups: list[list[str]], pool: pd.DataFrame) -> str:
    """Validated DK upload CSV as text (raises ValueError if any lineup breaks a rule)."""
    bad = {i + 1: validate(lu, pool) for i, lu in enumerate(lineups)}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise ValueError("Not exported — invalid lineups: " + "; ".join(f"#{k}: {', '.join(v)}" for k, v in bad.items()))
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    w.writerow(SLOTS)
    for lu in lineups:
        w.writerow(lu)
    text = buf.getvalue()
    rows = list(csv.reader(io.StringIO(text)))  # independent re-parse
    if not (rows[0] == SLOTS and len(rows) - 1 == len(lineups) and all(r == lu for r, lu in zip(rows[1:], lineups))):
        raise ValueError("Export failed its read-back check; nothing was saved.")
    return text


def dk_upload(lineups: list[list[str]], pool: pd.DataFrame, label: str, folder: Path = EXPORTS) -> Path:
    ensure_dirs()
    text = dk_csv_text(lineups, pool)
    path = Path(folder) / f"DK_upload_{_stamp(label)}.csv"
    path.write_text(text, newline="")
    with open(path, newline="") as fh:
        if list(csv.reader(fh))[1:] != lineups:
            path.unlink(missing_ok=True)
            raise ValueError("Export failed its read-back check; nothing was saved.")
    return path


def readable_text(lineups: list[list[str]], pool: pd.DataFrame) -> str:
    return _readable_df(lineups, pool).to_csv(index=False)


def readable(lineups: list[list[str]], pool: pd.DataFrame, label: str, folder: Path = EXPORTS) -> Path:
    ensure_dirs()
    df = _readable_df(lineups, pool)
    path = Path(folder) / f"Lineups_{_stamp(label)}.csv"
    df.to_csv(path, index=False)
    return path


def _readable_df(lineups, pool) -> pd.DataFrame:
    idx = pool.set_index("dk_id")
    rows = []
    for i, lu in enumerate(lineups, 1):
        for slot, d in zip(SLOTS, lu):
            r = idx.loc[d]
            rows.append({"lineup": i, "slot": slot, "dk_id": d, "name": r["name"], "pos": r.pos, "team": r.team,
                         "opp": r.opp, "salary": int(r.salary), "proj": round(float(r.proj), 2)})
    return pd.DataFrame(rows)


def lineup_table(lineups: list[list[str]], pool: pd.DataFrame) -> pd.DataFrame:
    idx = pool.set_index("dk_id")
    out = []
    for i, lu in enumerate(lineups, 1):
        r = idx.loc[lu]
        row = {"Rank": i}
        for slot, (_, p) in zip(SLOTS, r.iterrows()):
            k = slot if slot not in row else f"{slot}{sum(1 for c in row if c.startswith(slot)) + 1}"
            row[k] = f"{p['name']} ({p.team})"
        row["Salary"] = int(r.salary.sum())
        row["Proj"] = round(float(r.proj.sum()), 1)
        out.append(row)
    df = pd.DataFrame(out)
    return df[["Rank", "Proj", "Salary"] + [c for c in df.columns if c not in ("Rank", "Proj", "Salary")]]


def exposure(lineups: list[list[str]], pool: pd.DataFrame) -> pd.DataFrame:
    from collections import Counter
    c = Counter(d for lu in lineups for d in lu)
    idx = pool.set_index("dk_id")
    return pd.DataFrame([{"name": idx.at[d, "name"], "pos": idx.at[d, "pos"], "team": idx.at[d, "team"],
                          "salary": int(idx.at[d, "salary"]), "lineups": n, "exposure": f"{n / len(lineups):.0%}"}
                         for d, n in c.most_common()])
