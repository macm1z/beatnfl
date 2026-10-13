"""Local SQLite store: slate snapshots, frozen projection runs, saved lineup sets,
research quotes, and a data-status log. One file: ~/BeatNFL/beatnfl.db.

Snapshots are immutable; saving the same slate content twice reuses the first row.
"""

from __future__ import annotations

import io
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS slates(
  id INTEGER PRIMARY KEY, created_utc TEXT, label TEXT, route TEXT, draft_group_id TEXT,
  content_hash TEXT UNIQUE, season INT, week INT, ids_verified INT, first_start_utc TEXT, players BLOB);
CREATE TABLE IF NOT EXISTS projection_runs(
  id INTEGER PRIMARY KEY, slate_id INT, created_utc TEXT, model_version TEXT, sources TEXT, pool BLOB);
CREATE TABLE IF NOT EXISTS lineup_sets(
  id INTEGER PRIMARY KEY, name TEXT, slate_id INT, projection_run_id INT, created_utc TEXT, updated_utc TEXT,
  settings TEXT, revision INT DEFAULT 1, status TEXT, diagnostics TEXT);
CREATE TABLE IF NOT EXISTS lineups(
  set_id INT, idx INT, ids TEXT, salary INT, proj REAL, PRIMARY KEY(set_id, idx));
CREATE TABLE IF NOT EXISTS quotes(
  id INTEGER PRIMARY KEY, kind TEXT, event_id TEXT, season INT, week INT, book TEXT, market TEXT, stat TEXT,
  player TEXT, side TEXT, line REAL, price REAL, source_ts TEXT, retrieved_utc TEXT,
  model_pred REAL, model_p_over REAL, model_version TEXT,
  UNIQUE(event_id, book, market, player, side, line, source_ts));
CREATE TABLE IF NOT EXISTS data_status(
  source TEXT PRIMARY KEY, last_attempt_utc TEXT, last_ok_utc TEXT, status TEXT, message TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    ensure_dirs()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _blob(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_parquet(b, index=False)
    return b.getvalue()


def _df(blob: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob))


# ------------------------------------------------------------- slates
def save_slate(ls, season=None, week=None) -> int:
    with db() as c:
        row = c.execute("SELECT id FROM slates WHERE content_hash=?", (ls.content_hash,)).fetchone()
        if row:
            if season is not None:
                c.execute("UPDATE slates SET season=?, week=? WHERE id=?", (season, week, row["id"]))
            return row["id"]
        first = ls.players.start_utc[ls.players.start_utc != ""].min() if (ls.players.start_utc != "").any() else ""
        cur = c.execute("INSERT INTO slates(created_utc,label,route,draft_group_id,content_hash,season,week,ids_verified,"
                        "first_start_utc,players) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (now(), ls.slate.label if ls.slate else "", ls.route, ls.slate.draft_group_id if ls.slate else "",
                         ls.content_hash, season, week, int(ls.ids_verified), first, _blob(ls.players)))
        return cur.lastrowid


def get_slate(slate_id: int) -> tuple[dict, pd.DataFrame]:
    with db() as c:
        r = c.execute("SELECT * FROM slates WHERE id=?", (slate_id,)).fetchone()
    if not r:
        raise KeyError(slate_id)
    meta = {k: r[k] for k in r.keys() if k != "players"}
    return meta, _df(r["players"])


def mark_verified(slate_id: int, ok: bool):
    with db() as c:
        c.execute("UPDATE slates SET ids_verified=? WHERE id=?", (int(ok), slate_id))


# ------------------------------------------------------------- projections (frozen)
def save_projection_run(slate_id: int, pool: pd.DataFrame, model_version: str, sources: dict) -> int:
    with db() as c:
        cur = c.execute("INSERT INTO projection_runs(slate_id,created_utc,model_version,sources,pool) VALUES(?,?,?,?,?)",
                        (slate_id, now(), model_version, json.dumps(sources, default=str), _blob(pool)))
        return cur.lastrowid


def get_projection_run(run_id: int) -> tuple[dict, pd.DataFrame]:
    with db() as c:
        r = c.execute("SELECT * FROM projection_runs WHERE id=?", (run_id,)).fetchone()
    meta = {k: r[k] for k in r.keys() if k != "pool"}
    meta["sources"] = json.loads(meta["sources"])
    return meta, _df(r["pool"])


def projection_runs_for_grading() -> pd.DataFrame:
    """Last frozen run per slate created *before* the slate's first kickoff (true pregame forecasts)."""
    with db() as c:
        rows = c.execute("""SELECT p.id, p.slate_id, p.created_utc, p.model_version, s.season, s.week, s.first_start_utc
                            FROM projection_runs p JOIN slates s ON s.id=p.slate_id
                            WHERE s.first_start_utc != '' AND p.created_utc < s.first_start_utc""").fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    return df.sort_values("created_utc").drop_duplicates("slate_id", keep="last")


# ------------------------------------------------------------- lineups
def save_lineup_set(name: str, slate_id: int, run_id: int, settings: dict, lineups: list[list[str]],
                    pool: pd.DataFrame, status: str, diagnostics: list[str]) -> int:
    idx = pool.set_index("dk_id")
    dup = find_same_set(slate_id, lineups)
    if dup is not None:
        return dup
    with db() as c:
        cur = c.execute("INSERT INTO lineup_sets(name,slate_id,projection_run_id,created_utc,updated_utc,settings,status,"
                        "diagnostics) VALUES(?,?,?,?,?,?,?,?)",
                        (name, slate_id, run_id, now(), now(), json.dumps(settings, default=str), status, json.dumps(diagnostics)))
        sid = cur.lastrowid
        for i, lu in enumerate(lineups):
            c.execute("INSERT INTO lineups VALUES(?,?,?,?,?)",
                      (sid, i, json.dumps(lu), int(idx.loc[lu].salary.sum()), float(idx.loc[lu].proj.sum())))
    # confirm by reading back
    got = load_lineup_set(sid)
    if len(got["lineups"]) != len(lineups):
        raise RuntimeError("Save could not be confirmed (read-back mismatch).")
    return sid


def find_same_set(slate_id: int, lineups: list[list[str]]) -> int | None:
    """An already-saved set on this slate with exactly the same lineups (order ignored), if any."""
    want = sorted(json.dumps(sorted(lu)) for lu in lineups)
    with db() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM lineup_sets WHERE slate_id=? AND "
                                          "(SELECT COUNT(*) FROM lineups WHERE set_id=lineup_sets.id)=?",
                                          (slate_id, len(lineups))).fetchall()]
        for i in ids:
            got = sorted(json.dumps(sorted(json.loads(r["ids"]))) for r in
                         c.execute("SELECT ids FROM lineups WHERE set_id=?", (i,)).fetchall())
            if got == want:
                return i
    return None


def duplicate_sets() -> dict[int, int]:
    """{set_id: older set_id it duplicates} — same slate, same lineups (order ignored)."""
    with db() as c:
        rows = c.execute("SELECT l.id, l.slate_id, x.ids FROM lineup_sets l JOIN lineups x ON x.set_id=l.id ORDER BY l.id").fetchall()
    sig: dict[int, list] = {}
    slate: dict[int, int] = {}
    for r in rows:
        sig.setdefault(r["id"], []).append(json.dumps(sorted(json.loads(r["ids"]))))
        slate[r["id"]] = r["slate_id"]
    seen, out = {}, {}
    for i in sorted(sig):
        k = (slate[i], tuple(sorted(sig[i])))
        if k in seen:
            out[i] = seen[k]
        else:
            seen[k] = i
    return out


def list_lineup_sets() -> pd.DataFrame:
    with db() as c:
        rows = c.execute("""SELECT l.id, l.name, l.created_utc, l.updated_utc, l.revision, l.status, s.label AS slate,
                            s.season, s.week, (SELECT COUNT(*) FROM lineups WHERE set_id=l.id) AS n
                            FROM lineup_sets l JOIN slates s ON s.id=l.slate_id ORDER BY l.id DESC""").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def load_lineup_set(set_id: int) -> dict:
    with db() as c:
        s = c.execute("SELECT * FROM lineup_sets WHERE id=?", (set_id,)).fetchone()
        if not s:
            raise KeyError(set_id)
        lus = c.execute("SELECT ids FROM lineups WHERE set_id=? ORDER BY idx", (set_id,)).fetchall()
    d = dict(s)
    d["settings"] = json.loads(d["settings"])
    d["diagnostics"] = json.loads(d["diagnostics"] or "[]")
    d["lineups"] = [json.loads(r["ids"]) for r in lus]
    return d


def rename_lineup_set(set_id: int, name: str, expected_revision: int):
    with db() as c:
        n = c.execute("UPDATE lineup_sets SET name=?, updated_utc=?, revision=revision+1 WHERE id=? AND revision=?",
                      (name, now(), set_id, expected_revision)).rowcount
    if n == 0:
        raise RuntimeError("This lineup set changed since you opened it. Reopen it and try again.")


def delete_lineup_set(set_id: int):
    with db() as c:
        c.execute("DELETE FROM lineups WHERE set_id=?", (set_id,))
        c.execute("DELETE FROM lineup_sets WHERE id=?", (set_id,))


# ------------------------------------------------------------- research quotes
def save_quotes(df: pd.DataFrame, kind: str, season: int, week: int, model_version: str = "") -> int:
    if df is None or df.empty:
        return 0
    n = 0
    with db() as c:
        for r in df.itertuples():
            cur = c.execute("""INSERT OR IGNORE INTO quotes(kind,event_id,season,week,book,market,stat,player,side,line,price,
                               source_ts,retrieved_utc,model_pred,model_p_over,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (kind, getattr(r, "event_id", ""), season, week, r.book, r.market, getattr(r, "stat", ""),
                             getattr(r, "player", getattr(r, "side", "")), r.side, r.line, r.price, r.source_ts, now(),
                             getattr(r, "model_pred", None), getattr(r, "model_p_over", None), model_version))
            n += cur.rowcount
    return n


def quotes(kind: str | None = None) -> pd.DataFrame:
    with db() as c:
        q = "SELECT * FROM quotes" + (" WHERE kind=?" if kind else "")
        rows = c.execute(q, (kind,) if kind else ()).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ------------------------------------------------------------- data status
def log_status(source: str, ok, message: str):
    """ok: True/False, or a step status string ("ok", "warn", "skip", "fail")."""
    status = ok if isinstance(ok, str) else ("ok" if ok else "failed")
    ok = status in ("ok", "warn", "skip")
    t = now()
    with db() as c:
        c.execute("""INSERT INTO data_status(source,last_attempt_utc,last_ok_utc,status,message) VALUES(?,?,?,?,?)
                     ON CONFLICT(source) DO UPDATE SET last_attempt_utc=excluded.last_attempt_utc,
                     last_ok_utc=COALESCE(excluded.last_ok_utc, data_status.last_ok_utc),
                     status=excluded.status, message=excluded.message""",
                  (source, t, t if ok else None, status, message))


def data_status() -> pd.DataFrame:
    with db() as c:
        rows = c.execute("SELECT * FROM data_status ORDER BY source").fetchall()
    return pd.DataFrame([dict(r) for r in rows])
