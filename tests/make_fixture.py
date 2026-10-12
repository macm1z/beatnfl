"""Builds a FIXTURE DraftKings salary file for 2026 week 6 from real nflverse rosters.

Salaries are SYNTHETIC (derived from recent DK points) and IDs are fake. It exists only
to exercise the pipeline offline; nothing built from it is ever labeled live.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beatnfl.sources import nflverse as nv  # noqa: E402

GAMES = [("CHI", "ATL"), ("BAL", "CLE"), ("TEN", "IND"), ("NYJ", "NE"), ("NO", "NYG"), ("CAR", "PHI"),
         ("PIT", "TB"), ("ARI", "LA"), ("LAC", "KC"), ("BUF", "LV")]
TIMES = {("ARI", "LA"): "04:05PM", ("LAC", "KC"): "04:25PM", ("BUF", "LV"): "04:25PM"}
DK_TEAM = {"LA": "LAR"}
NICK = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
        "CLE": "Browns", "IND": "Colts", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders",
        "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers",
        "TB": "Buccaneers", "TEN": "Titans"}
RANGE = {"QB": (4800, 8200), "RB": (4000, 9200), "WR": (3000, 9200), "TE": (2500, 7500), "DST": (2200, 3800)}


def main(out: Path):
    st = nv.weekly_stats([2026])
    st = st[(st.week <= 5) & st.position.isin(["QB", "RB", "WR", "TE"])]
    ppg = st.groupby(["player_id", "player_display_name", "position", "team"]).agg(ppg=("dk_pts", "mean"), g=("dk_pts", "size")).reset_index()
    last_team = st.sort_values("week").drop_duplicates("player_id", keep="last")[["player_id", "team"]]
    ppg = ppg.merge(last_team, on=["player_id", "team"])
    rows, pid = [], 40100000
    for a, h in GAMES:
        for t in (a, h):
            d = ppg[ppg.team == t]
            for p, n in {"QB": 2, "RB": 5, "WR": 7, "TE": 4}.items():
                for r in d[d.position == p].sort_values("ppg", ascending=False).head(n).itertuples():
                    lo, hi = RANGE[p]
                    sal = int(np.clip(round((lo + r.ppg * (hi - lo) / 26) / 100) * 100, lo, hi))
                    pid += 1
                    rows.append((p, r.player_display_name, pid, f"{p}/FLEX" if p != "QB" else "QB", sal, a, h, t, round(r.ppg, 2)))
            pid += 1
            rows.append(("DST", f"{NICK[t]} ", pid, "DST", int(np.random.default_rng(pid).integers(22, 38)) * 100, a, h, t, 7.0))
    lines = ["Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for p, name, i, rp, sal, a, h, t, avg in rows:
        da, dh, dt = DK_TEAM.get(a, a), DK_TEAM.get(h, h), DK_TEAM.get(t, t)
        gi = f"{da}@{dh} 10/18/2026 {TIMES.get((a, h), '01:00PM')} ET"
        nm = f'"{name}"' if "," in name else name
        lines.append(f'{p},{nm} ({i}),{nm},{i},{rp},{sal},{gi},{dt},{avg}')
    out.write_text("﻿" + "\r\n".join(lines) + "\r\n\r\n")  # BOM + CRLF + trailing blank line on purpose
    print(f"wrote {out} ({len(rows)} players)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures" / "DKSalaries_FIXTURE_2026wk6.csv")
