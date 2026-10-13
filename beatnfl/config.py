"""Settings and paths. Everything lives in one folder (default ~/BeatNFL)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

HOME = Path(os.environ.get("BEATNFL_HOME", Path.home() / "BeatNFL"))
CACHE = HOME / "cache"
INBOX = HOME / "inbox"          # drop paid projection CSVs or DKSalaries.csv here
EXPORTS = HOME / "exports"
DB_PATH = HOME / "beatnfl.db"
SETTINGS_PATH = HOME / "settings.json"
LOG_PATH = HOME / "beatnfl.log"


def ensure_dirs() -> None:
    for p in (HOME, CACHE, INBOX, EXPORTS):
        p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    odds_api_key: str = ""
    fantasypros_api_key: str = ""   # production key (MVP/HOF) → weekly projections fill "Yours" automatically
    yahoo_client_id: str = ""
    yahoo_client_secret: str = ""
    update_repo: str = "macm1z/beatnfl"  # GitHub "owner/name" the Update button pulls from
    github_token: str = ""       # only for a private repository (read-only token)
    # Projection blend weights; sources that are missing are dropped and the rest renormalised.
    w_paid: float = 0.50
    w_sleeper: float = 0.25
    w_model: float = 0.25
    # Optimizer defaults (Classic GPP)
    n_lineups: int = 20
    min_salary: int = 49000
    max_exposure: float = 0.60
    min_unique: int = 2
    stack_min: int = 1          # pass catchers stacked with QB
    bring_back: int = 0         # opponents of QB's team
    randomness: float = 0.12    # per-lineup projection noise (fraction of projection)
    no_offense_vs_dst: bool = True
    questionable_haircut: float = 0.90
    paid_source_label: str = "Paid"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        ensure_dirs()
        s = cls()
        if SETTINGS_PATH.exists():
            try:
                raw = json.loads(SETTINGS_PATH.read_text())
                for k, v in raw.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except Exception:
                pass  # a corrupt settings file never blocks the app; defaults are used
        env_key = os.environ.get("ODDS_API_KEY")
        if env_key:
            s.odds_api_key = env_key
        return s

    def save(self) -> None:
        ensure_dirs()
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2))
        try:
            os.chmod(SETTINGS_PATH, 0o600)  # holds API keys: readable by your user account only
        except OSError:
            pass
