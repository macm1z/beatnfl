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
    # Tested per-position weights {"QB": {"paid": .4, "sleeper": .3, "model": .3}, …} — override the three above
    use_book_lines: bool = True     # fold sportsbook prop lines into the model's projection (needs the Odds API key)
    pos_weights: dict = field(default_factory=dict)
    # Tested share of FantasyPros in each stat forecast for Props {"rec_yds": .5, …}
    stat_weights: dict = field(default_factory=dict)
    bettingpros_api_key: str = ""   # BettingPros Consumer API (personal use; follows your membership)
    anthropic_api_key: str = ""     # Ask tab (platform.claude.com/settings/keys); billed per use by Anthropic
    ask_model: str = "claude-sonnet-5-5"
    # Contest type: "cash" | "se" (single-entry tournament) | "multi" — picks the lineup-builder preset
    contest: str = "se"
    # Optimizer defaults (multi-entry tournament preset; cash / single entry use optimizer.PRESETS)
    n_lineups: int = 20
    min_salary: int = 49000
    max_exposure: float = 0.35    # ≤ 7 of 20 lineups per player (was 0.60 — one bust sank most lineups)
    min_unique: int = 3           # lineups differ by ≥ 3 players (was 2)
    settings_version: int = 2
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
                if int(raw.get("settings_version", 1)) < 2:   # move people still on the old defaults to the new ones
                    if float(raw.get("max_exposure", 0.6)) == 0.6:
                        s.max_exposure = 0.35
                    if int(raw.get("min_unique", 2)) == 2:
                        s.min_unique = 3
                    s.settings_version = 2
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


def local_tz() -> str:
    """This computer's time zone name (e.g. America/Denver), for showing kickoff and update times."""
    import os
    try:
        real = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in real:
            return real.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return os.environ.get("TZ") or "America/New_York"
