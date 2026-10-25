"""DraftKings NFL Classic lineup optimizer (mixed-integer program via SciPy/HiGHS).

Objective: maximise projected DK points (with optional per-lineup noise for GPP
variety). It never pads results: if fewer valid distinct lineups exist under the
rules, it returns fewer and says which rule bound.

Exposure rounding: max lineups per player = max(1, floor(max_exposure × requested)).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

CAP = 50000
SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
FLEX_OK = {"RB", "WR", "TE"}


@dataclass
class OptSettings:
    n_lineups: int = 20
    min_salary: int = 49000
    max_salary: int = CAP
    max_exposure: float = 0.35
    min_unique: int = 3
    stack_min: int = 1
    bring_back: int = 0
    stack_positions: tuple = ("WR", "TE")
    randomness: float = 0.12
    no_offense_vs_dst: bool = True
    locks: list = field(default_factory=list)
    excludes: list = field(default_factory=list)
    seed: int = 2026
    time_limit: float = 10.0


# Contest presets. Single-entry contests are won by your one best lineup, so there is no noise
# and no exposure limit: the optimizer returns the highest-projected lineup first, plus the next
# best distinct ones as backups for late news (change ≥ 2 players).
CONTESTS = {
    "cash": "Cash — 50/50, double-up, head-to-head",
    "se": "Single-entry tournament",
    "multi": "Multi-entry tournament (many lineups)",
}
PRESETS = {
    "cash": dict(n_lineups=3, max_exposure=1.0, min_unique=2, stack_min=0, bring_back=0, randomness=0.0),
    "se": dict(n_lineups=3, max_exposure=1.0, min_unique=2, stack_min=1, bring_back=0, randomness=0.0),
    "multi": dict(n_lineups=20, max_exposure=0.35, min_unique=3, stack_min=1, bring_back=0, randomness=0.12),
}


def preset(contest: str, **over) -> dict:
    """Optimizer settings for a contest type (unknown type → single-entry tournament)."""
    return {**PRESETS.get(contest, PRESETS["se"]), **over}


@dataclass
class OptResult:
    lineups: list            # list[list[dk_id]] in SLOT order
    requested: int
    status: str              # "complete" | "partial" | "infeasible"
    diagnostics: list
    elapsed: float
    seed: int


def _eligible(pool: pd.DataFrame, s: OptSettings) -> pd.DataFrame:
    p = pool[pool.proj.notna() & ~pool.out & ~pool.dk_id.isin(s.excludes)].copy()
    return p.reset_index(drop=True)


def build(pool: pd.DataFrame, s: OptSettings) -> OptResult:
    t0 = time.time()
    p = _eligible(pool, s)
    diags = []
    missing_locks = [l for l in s.locks if l not in set(p.dk_id)]
    if missing_locks:
        names = pool.set_index("dk_id").reindex(missing_locks)["name"].fillna("?").tolist()
        return OptResult([], s.n_lineups, "infeasible",
                         [f"Locked player(s) can't be used (ruled out, excluded or no projection): {', '.join(names)}"],
                         0, s.seed)
    n = len(p)
    pos = p.pos.to_numpy()
    sal = p.salary.to_numpy(dtype=float)
    base = p.proj.to_numpy(dtype=float)
    games = sorted(p.game.unique())
    gi = {g: i for i, g in enumerate(games)}
    G = len(games)
    nv = n + G  # x vars then y (game used) vars

    A, lo, hi = [], [], []

    def add(row, l, h):
        A.append(row); lo.append(l); hi.append(h)

    def onehot(mask):
        r = np.zeros(nv); r[:n] = mask.astype(float); return r

    add(onehot(np.ones(n, bool)), 9, 9)
    for ps, (l, h) in {"QB": (1, 1), "DST": (1, 1), "RB": (2, 3), "WR": (3, 4), "TE": (1, 2)}.items():
        add(onehot(pos == ps), l, h)
    r = np.zeros(nv); r[:n] = sal; add(r, s.min_salary, s.max_salary)
    # at least two games
    for g, j in gi.items():
        r = np.zeros(nv); r[:n] = -(p.game.to_numpy() == g).astype(float); r[n + j] = 1; add(r, -np.inf, 0)
    r = np.zeros(nv); r[n:] = 1; add(r, 2, np.inf)
    # stacks
    team, opp = p.team.to_numpy(), p.opp.to_numpy()
    for qi in np.where(pos == "QB")[0]:
        if s.stack_min > 0:
            r = onehot((team == team[qi]) & np.isin(pos, s.stack_positions)); r[qi] = -s.stack_min; add(r, 0, np.inf)
        if s.bring_back > 0:
            r = onehot((team == opp[qi]) & np.isin(pos, ["RB", "WR", "TE"])); r[qi] = -s.bring_back; add(r, 0, np.inf)
    if s.no_offense_vs_dst:
        for di in np.where(pos == "DST")[0]:
            r = onehot((team == opp[di]) & (pos != "DST")); r[di] = 8; add(r, -np.inf, 8)
    base_A = np.array(A); base_lo, base_hi = np.array(lo), np.array(hi)

    ub = np.ones(nv)
    lb = np.zeros(nv)
    lock_idx = [int(np.where(p.dk_id == l)[0][0]) for l in s.locks]
    lb[lock_idx] = 1
    max_uses = max(1, math.floor(s.max_exposure * s.n_lineups))
    uses = np.zeros(n, int)
    rng = np.random.default_rng(s.seed)
    integrality = np.ones(nv)
    lineups, sets = [], []
    uniq_rows = []
    for k in range(s.n_lineups):
        noise = rng.normal(1.0, s.randomness, n) if s.randomness > 0 else np.ones(n)
        c = np.zeros(nv); c[:n] = -(base * np.clip(noise, 0.5, 1.5))
        ub_k = ub.copy()
        capped = (uses >= max_uses)
        capped[lock_idx] = False
        ub_k[:n][capped] = 0
        Ak, lok, hik = base_A, base_lo, base_hi
        if uniq_rows:
            Ak = np.vstack([base_A, np.array(uniq_rows)])
            lok = np.concatenate([base_lo, np.full(len(uniq_rows), -np.inf)])
            hik = np.concatenate([base_hi, np.full(len(uniq_rows), 9 - s.min_unique)])
        res = milp(c, constraints=LinearConstraint(Ak, lok, hik), integrality=integrality,
                   bounds=Bounds(lb, ub_k), options={"time_limit": s.time_limit, "disp": False})
        if res.x is None or res.status not in (0, 1):
            diags.append(_explain(p, s, k, uses, max_uses))
            break
        if res.status == 1:
            diags.append(f"Lineup {k + 1}: solver hit the {s.time_limit}s limit — best found lineup kept (feasible, may not be optimal).")
        chosen = np.where(res.x[:n] > 0.5)[0]
        key = frozenset(p.dk_id.iloc[chosen])
        if key in sets:  # never pad with duplicates
            diags.append("Solver returned a duplicate lineup; stopping.")
            break
        sets.append(key)
        uses[chosen] += 1
        r = np.zeros(nv); r[chosen] = 1; uniq_rows.append(r)
        lineups.append(assign_slots(p.iloc[chosen]))
    status = "complete" if len(lineups) == s.n_lineups else ("partial" if lineups else "infeasible")
    return OptResult(lineups, s.n_lineups, status, diags, round(time.time() - t0, 2), s.seed)


def _explain(p, s, k, uses, max_uses) -> str:
    if k == 0:
        cheapest = p.groupby("pos").salary.nsmallest(3).groupby(level=0).apply(list).to_dict()
        hints = []
        if s.min_salary > 49800:
            hints.append("minimum salary is very high")
        if s.stack_min >= 3 or s.bring_back >= 2:
            hints.append("stack/bring-back rules are strict")
        if s.locks:
            hints.append("locked players may conflict (salary, position counts, or DST-vs-offense rule)")
        return ("No valid lineup exists under these rules" + (": " + "; ".join(hints) if hints else ".") +
                " Loosen the rules and try again.")
    return (f"Only {k} distinct lineups satisfy the rules. Binding limits: exposure cap = {max_uses} lineups per "
            f"player, min {s.min_unique} changed players between lineups. Raise max exposure or lower min-unique for more.")


def assign_slots(lu: pd.DataFrame) -> list[str]:
    """Order players into DK's slot order. FLEX = the RB/WR/TE with the latest kickoff (late-swap friendly)."""
    lu = lu.sort_values(["start_utc", "proj"], ascending=[True, False])
    flex_cand = []
    for i, r in lu.iterrows():
        if r.pos not in FLEX_OK:
            continue
        rest = lu.drop(index=i)
        c = rest.pos.value_counts()
        if c.get("RB", 0) == 2 and c.get("WR", 0) == 3 and c.get("TE", 0) == 1:
            flex_cand.append(i)
    flex = max(flex_cand, key=lambda i: (str(lu.at[i, "start_utc"]), -lu.at[i, "proj"])) if flex_cand else None
    out = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
    for i, r in lu.iterrows():
        if i != flex:
            out[r.pos].append(r.dk_id)
    return out["QB"] + out["RB"] + out["WR"] + out["TE"] + ([lu.at[flex, "dk_id"]] if flex is not None else []) + out["DST"]


def validate(lineup: list[str], pool: pd.DataFrame) -> list[str]:
    """Independent DK Classic rules check (does not trust the solver)."""
    errs = []
    if len(lineup) != 9:
        return [f"lineup has {len(lineup)} players, needs 9"]
    if len(set(lineup)) != 9:
        errs.append("same player used twice")
    idx = pool.set_index("dk_id")
    missing = [d for d in lineup if d not in idx.index]
    if missing:
        return errs + [f"player ID(s) not on this slate: {', '.join(missing)}"]
    rows = idx.loc[lineup]
    for slot, (_, r) in zip(SLOTS, rows.iterrows()):
        ok = r.pos in FLEX_OK if slot == "FLEX" else r.pos == slot
        if not ok:
            errs.append(f"{r['name']} ({r.pos}) can't fill {slot}")
    total = int(rows.salary.sum())
    if total > CAP:
        errs.append(f"salary ${total:,} is over the ${CAP:,} cap")
    if rows.game.nunique() < 2:
        errs.append("players must come from at least 2 games")
    if "out" in rows and rows.out.any():
        errs.append("includes a player ruled out: " + ", ".join(rows[rows.out]["name"]))
    return errs
