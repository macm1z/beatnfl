"""HTTP with retries, a disk cache, and honest staleness.

fetch() returns (payload_bytes, Meta). If the network call fails but a cached copy
exists, the cached copy is returned with meta.stale=True so the pipeline can warn
instead of stopping. If neither works, a StepError explains what to do.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import requests

from .config import CACHE, ensure_dirs
from .errors import StepError

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


@dataclass
class Meta:
    url: str
    fetched_at: float
    from_cache: bool
    stale: bool
    note: str = ""

    @property
    def age_hours(self) -> float:
        return (time.time() - self.fetched_at) / 3600


def _key(url: str, params: dict | None) -> str:
    raw = url + json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _redact(params: dict | None) -> dict:
    return {k: ("***" if "key" in k.lower() else v) for k, v in (params or {}).items()}


def fetch(url: str, *, step: str, params: dict | None = None, max_age_h: float = 1.0,
          timeout: float = 20, retries: int = 3, headers: dict | None = None,
          offline: bool = False) -> tuple[bytes, Meta]:
    ensure_dirs()
    k = _key(url, params)
    body_p, meta_p = CACHE / f"{k}.bin", CACHE / f"{k}.json"
    cached_at = None
    if body_p.exists() and meta_p.exists():
        try:
            cached_at = json.loads(meta_p.read_text())["fetched_at"]
        except Exception:
            cached_at = None
    if cached_at and (offline or (time.time() - cached_at) / 3600 < max_age_h):
        return body_p.read_bytes(), Meta(url, cached_at, True, False)
    if offline:
        raise StepError(step, "Offline mode and no cached copy of this data.",
                        "This source has never been downloaded on this computer.",
                        "Turn off offline mode and run again while connected.")

    last_err, code = "", None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA, "Accept": "application/json, */*", **(headers or {})})
            if r.status_code == 200:
                body_p.write_bytes(r.content)
                meta_p.write_text(json.dumps({"url": url, "params": _redact(params), "fetched_at": time.time()}))
                return r.content, Meta(url, time.time(), False, False)
            last_err, code = f"HTTP {r.status_code}: {r.text[:200]}", r.status_code
            if r.status_code in (401, 403, 404, 422):
                break  # retrying will not help
            if r.status_code == 429:
                ra = r.headers.get("Retry-After", "")
                time.sleep(min(30, float(ra)) if ra.replace(".", "").isdigit() else 3 * (attempt + 1))
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            m = __import__("re").search(r"\b(40[134]|429)\b", str(e))
            code = int(m.group(1)) if m else None
            if code in (401, 403, 404):
                break
        time.sleep(1.5 * (attempt + 1))

    if cached_at:
        return body_p.read_bytes(), Meta(url, cached_at, True, True,
                                         note=f"live fetch failed ({last_err[:80]}); using copy from "
                                              f"{(time.time() - cached_at) / 3600:.1f}h ago")
    why = "The site refused or did not answer."
    if code in (401, 403):
        why = (f"The site refused the request (HTTP {code}): the API key is wrong or out of credits."
               if "the-odds-api" in url else
               f"The site refused the request (HTTP {code}) — it is blocking this network or automated access "
               "(VPNs and cloud servers are often blocked).")
    elif code == 404:
        why = "The address no longer exists — the provider may have changed its feed."
    elif code == 429:
        why = "Rate limited — too many requests in a short time, or the API quota is used up."
    elif "Connection" in last_err or "Timeout" in last_err:
        why = "No internet connection, or the site is down."
    raise StepError(step, f"Could not download data from {url.split('?')[0]}.", why,
                    "Check your connection and run again. If it keeps failing, use the manual fallback shown for this step.",
                    detail=last_err)


def fetch_json(url: str, **kw):
    body, meta = fetch(url, **kw)
    try:
        return json.loads(body), meta
    except ValueError as e:
        raise StepError(kw.get("step", "download"), f"{url.split('?')[0]} returned something that is not data.",
                        "The site probably returned an error or login page instead of JSON.",
                        "Run again later; if it persists use the manual fallback for this step.", detail=str(e))
