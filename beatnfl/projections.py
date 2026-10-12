"""Merge slate + injuries + every projection source into one player pool.

Rules:
- A missing source is NaN, never 0. The blend renormalises over sources present.
- Ruled-out players (O/IR/D, from DK or Sleeper — the worse status wins) are kept in
  the table but flagged and never optimised.
- Questionable players get a configurable haircut and a visible flag.
- Every row records which sources contributed (provenance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .names import loose_key

STATUS_RANK = {"": 0, "None": 0, "P": 1, "Q": 2, "D": 3, "O": 4, "IR": 4, "OUT": 4}
DK_STATUS = {"NONE": "", "": "", "Q": "Q", "D": "D", "O": "O", "OUT": "O", "IR": "IR", "PUP": "O", "SUS": "O", "NA": "O"}


def attach_status(slate: pd.DataFrame, sleeper: pd.DataFrame | None) -> pd.DataFrame:
    s = slate.reset_index(drop=True).copy()
    s["status_dk"] = s.dk_status.map(lambda v: DK_STATUS.get(str(v).strip().upper(), ""))
    s["status_sleeper"], s["injury_note"], s["depth"] = "", "", np.nan
    if sleeper is not None and not sleeper.empty:
        sp = sleeper.drop_duplicates("key")
        m = s[["key"]].merge(sp[["key", "injury", "injury_raw", "injury_body", "depth_order", "depth_pos", "news_utc"]],
                             on="key", how="left")
        # second pass on loose key (team changed)
        miss = m.injury.isna() & m.depth_order.isna()
        if miss.any():
            lk = [loose_key(n, p) for n, p in zip(s.loc[miss, "name"], s.loc[miss, "pos"])]
            sp2 = sleeper.drop_duplicates("lkey", keep=False).set_index("lkey")
            for i, k in zip(m.index[miss], lk):
                if k in sp2.index:
                    for c in ("injury", "injury_raw", "injury_body", "depth_order", "depth_pos", "news_utc"):
                        m.at[i, c] = sp2.at[k, c]
        s["status_sleeper"] = m.injury.fillna("").values
        s["injury_note"] = (m.injury_raw.fillna("") + " " + m.injury_body.fillna("")).str.strip().values
        s["depth"] = pd.to_numeric(m.depth_order, errors="coerce").values
    worst = [a if STATUS_RANK.get(a, 0) >= STATUS_RANK.get(b, 0) else b for a, b in zip(s.status_dk, s.status_sleeper)]
    s["status"] = worst
    s["out"] = s.status.isin(["O", "IR", "D"])
    return s


def blend(pool: pd.DataFrame, settings, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """sources: name -> df with dk_id + proj_<name> (already matched to the slate)."""
    p = pool.reset_index(drop=True).copy()
    for name, df in sources.items():
        col = f"proj_{name}"
        if df is None or df.empty or col not in df:
            p[col] = np.nan
            continue
        p = p.merge(df[["dk_id", col]].drop_duplicates("dk_id"), on="dk_id", how="left")
    weights = {"paid": settings.w_paid, "sleeper": settings.w_sleeper, "model": settings.w_model}
    num = pd.Series(0.0, index=p.index)
    den = pd.Series(0.0, index=p.index)
    used = [[] for _ in range(len(p))]
    for name, w in weights.items():
        col = f"proj_{name}"
        if col not in p:
            p[col] = np.nan
        have = p[col].notna()
        num[have] += w * p.loc[have, col]
        den[have] += w
        for i in np.where(have)[0]:
            used[i].append(name)
    p["proj_raw"] = (num / den.replace(0, np.nan)).round(2)
    p["sources"] = [",".join(u) for u in used]
    p["n_sources"] = [len(u) for u in used]
    # spread between sources = disagreement flag
    cols = [f"proj_{n}" for n in weights]
    p["proj_spread"] = p[cols].max(axis=1) - p[cols].min(axis=1)
    hair = np.where(p.status == "Q", settings.questionable_haircut, 1.0)
    p["proj"] = (p.proj_raw * hair).round(2)
    p.loc[p.out, "proj"] = np.nan
    p["value"] = (p.proj / (p.salary / 1000)).round(2)
    return p


def match_external(pool: pd.DataFrame, ext: pd.DataFrame, col: str) -> tuple[pd.DataFrame, int]:
    """Attach an external projection table (with key/lkey or dk_id) to slate dk_ids."""
    if ext is None or ext.empty:
        return pd.DataFrame(columns=["dk_id", col]), 0
    if "dk_id" in ext and ext.dk_id.astype(str).str.fullmatch(r"\d+").mean() > 0.8:
        m = pool[["dk_id"]].merge(ext[["dk_id", col]].astype({"dk_id": str}).drop_duplicates("dk_id"), on="dk_id", how="inner")
        return m, len(m)
    a = pool[["dk_id", "key", "name", "pos"]].merge(ext[["key", col]].drop_duplicates("key"), on="key", how="left")
    miss = a[col].isna()
    if miss.any() and "lkey" in ext:
        e2 = ext.drop_duplicates("lkey", keep=False).set_index("lkey")[col]
        lk = pd.Series([loose_key(n, p) for n, p in zip(a.loc[miss, "name"], a.loc[miss, "pos"])], index=a.index[miss])
        a.loc[miss, col] = pd.to_numeric(lk.map(e2), errors="coerce")
    a = a[a[col].notna()]
    return a[["dk_id", col]], len(a)


def attach_external_status(pool: pd.DataFrame, ext: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    """Merge another source's injury tags (name/pos/team, then name/pos). Worst status wins."""
    from .names import match_key
    p = pool.reset_index(drop=True).copy()
    e = ext.reset_index(drop=True).copy()
    e["key"] = [match_key(n, ps, t) for n, ps, t in zip(e.name, e.pos, e.team)]
    e["lkey"] = [loose_key(n, ps) for n, ps in zip(e.name, e.pos)]
    by_key = e.drop_duplicates("key").set_index("key")
    by_lk = e.drop_duplicates("lkey", keep=False).set_index("lkey")
    col = f"status_{label}"
    vals, notes, pct = [], [], []
    for r in p.itertuples():
        hit = by_key.loc[r.key] if r.key in by_key.index else (by_lk.loc[loose_key(r.name, r.pos)]
                                                                if loose_key(r.name, r.pos) in by_lk.index else None)
        vals.append(hit.status if hit is not None else "")
        notes.append(hit.injury_note if hit is not None else "")
        pct.append(hit.pct_rostered if hit is not None and "pct_rostered" in hit else None)
    p[col] = vals
    p[f"pct_rostered_{label}"] = pct
    before = p.status.copy()
    p["status"] = [a if STATUS_RANK.get(a, 0) >= STATUS_RANK.get(b, 0) else b for a, b in zip(p.status, p[col])]
    upgraded = (p.status != before).to_numpy()
    changed = int(upgraded.sum())
    p["injury_note"] = [n if (u and not old) else old for old, n, u in zip(p.injury_note.fillna(""), notes, upgraded)]
    p["out"] = p.status.isin(["O", "IR", "D"])
    return p, changed
