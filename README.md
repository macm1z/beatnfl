# BeatNFL

DraftKings NFL Classic lineups plus game and player-prop research, with historical tests that show whether the predictions help. It runs on your own computer. It is read-only: no bets, no sportsbook logins, no contest entry.

## Updates

In the app: **App updates** (sidebar) → **Check for updates** → **Install update**, then restart the app.
- Code comes from github.com/macm1z/beatnfl.
- Your data, keys and Python setup are kept.
- If the new version fails its self-tests, the previous version is restored.

## Setup (once)

**Mac:** unzip, then double-click **Install BeatNFL.command**. If macOS blocks it, right-click → Open → Open.
Or in Terminal: `cd ~/Downloads/BeatNFL && bash "Install BeatNFL.command"`

**Windows:** double-click **Install BeatNFL.bat**.

The installer does four things:
1. Fetches a private Python 3.13 through `uv`, so it never touches your system Python.
2. Installs pinned package versions.
3. Runs the self-tests.
4. Runs the **doctor**, which checks every data source live from your network and explains any failure.

**What setup installs:**
- **uv 0.11.32:** pinned version, installed in your user folder.
- **Python 3.13:** a private copy for this app only.
- **47 packages from PyPI:** each locked to an exact version and SHA-256 checksum. Setup stops if any file doesn't match.

Setup needs no admin rights, changes no system settings and adds nothing to startup. Prefer Homebrew? Run `brew install uv` first and setup skips the uv download.

Setup also creates a **BeatNFL** app in your Applications folder. Open it from Launchpad or drag it to the Dock. It starts BeatNFL in the background (no Terminal window) and opens your browser; if it's already running, it just opens the page. To stop it, run `bash launch.sh stop` in the BeatNFL folder, or restart the Mac.

You can also open the app with **Start BeatNFL.command** (or `.bat`). It opens in your browser.

## Each week

Tabs: **This week** (pool and lineups) · **Saved lineups** (rename, delete, remove duplicates, filter by player) · **Games** (spread, moneyline and total chances: market vs model) · **Props** · **My Yahoo team** · **Evidence** · **Load data** · **Data status**.

1. **⟳ Refresh** (top right). This loads and checks the slate, then injuries, lines and projections, then blends them into the **Combined** projection and freezes the forecast. Anything that needs attention shows in a fold-out on *This week*; the full step log is on *Load data*, which is also where you pick a different slate, upload a salary file or paste your own projections.
2. **Review the pool.** Filter by position or team, lock or exclude players. Hover a column name to see what it means.
3. **⚙ Build lineups**, then **💾 Save** if you want to keep them. Lineups are ranked best projection first. Saving writes the DK upload CSV and a readable CSV to `~/BeatNFL/exports`; an identical set is never saved twice. Download buttons work without saving.
4. Upload `DK_upload_….csv` on DraftKings yourself (Upload Lineups).

**Optional inputs, all dropped in `~/BeatNFL/inbox`:**
- **Paid projections:** drop your site's DK projection CSV in the inbox. Columns are detected and shown.
- **Fallback slate:** if DraftKings blocks the automatic download, drop `DKSalaries.csv` in the inbox. The app uses it automatically and says so. To get it: DraftKings → any NFL Classic contest → Export to CSV.
- **Odds API key** (free at the-odds-api.com): add it in Settings for live lines and props. Without it, lineups still work.

**Yahoo (optional, read-only):** the **My Yahoo team** tab walks through a one-time setup. Create a Yahoo app with Redirect URI `oob` and Fantasy Sports Read access, paste the keys in Settings, sign in, then paste the code Yahoo shows you. After that:
- My Yahoo team shows your roster with projections and start/sit suggestions.
- Yahoo's injury tags join DraftKings' and Sleeper's in every run; the most serious status wins.
- Research adds % rostered on Yahoo.

Nothing is ever changed on Yahoo.

Headless option: `python -m beatnfl run` does the whole week and writes the files.

## Data sources

| What | Source | If it fails |
|---|---|---|
| Slate, salaries, IDs | DraftKings lobby + draftables JSON → DK CSV endpoint | Inbox `DKSalaries.csv` (always works) |
| Injuries, depth charts | Sleeper (free) + DraftKings' own status tags | DK tags only, flagged |
| Projections | Paid CSV, Sleeper, in-house model (blend weights in Settings) | Remaining sources renormalized |
| History, lines, results | nflverse (GitHub releases) | Cached copy, shown as STALE |
| Live lines, props | The Odds API | nflverse lines; props hidden |

Rules:
- Missing data is never treated as 0.
- Old salary files are refused rather than swapped in silently.
- Cached data is always labeled stale.
- DK export IDs from the JSON route stay "unverified" until checked against an official CSV in the inbox.

## In-house model (pm-1.2)

Gradient-boosted forecasts of DK points and of each prop stat:
- **Player inputs:** recent form (exponentially weighted), target and rush share, efficiency, and opportunity vacated by inactive teammates.
- **Game inputs:** team pace, game line (implied total and spread), and opponent defense.

DK points come from DraftKings scoring code, including yardage bonuses. Prop probabilities come from out-of-sample error distributions, recalibrated so they are not overconfident.

## Evidence (bundled in `evidence/backtest_baseline.json`, re-run from the Evidence tab)

Retrospective test: each season 2022–2026 is forecast by a model trained only on earlier seasons, and every input uses only earlier games.

**DK points, mean absolute error** (lower is better):

| | Model | Recent-form average | Season average |
|---|---|---|---|
| QB | **6.26** | 6.63 | 6.73 |
| RB | **4.51** | 4.57 | 4.59 |
| WR | **4.71** | 4.76 | 4.86 |
| TE | **3.62** | 3.68 | 3.75 |

**Decision test:** each week, take the top-N players at each position and measure the actual points they scored.

| | Model's top picks | Season-average picks | Model better in |
|---|---|---|---|
| QB top 12 | 20.2 | 19.4 | 70% of weeks |
| WR top 36 | 14.3 | 13.9 | 74% of weeks |
| TE top 12 | 11.4 | 10.7 | 66% of weeks |
| RB top 24 | 15.3 | 15.0 | 58% of weeks |

Other results:
- **DST:** the model ranks defenses with about twice the rank correlation of recent form (0.18–0.34 vs 0.10–0.17).
- **Prop stats:** the model beats the recent-form baseline on every stat, and its probabilities are calibrated. This test used baseline-derived lines, not real sportsbook lines, so it is not evidence of beating a book. The app saves every real line it pulls and grades those prospectively (`python -m beatnfl grade`).
- **Matchup stats:** team efficiency per play, success rate, pass rate over expected, explosive plays and red zone, all from free play-by-play. They're shown in Research. As model inputs they made forecasts slightly worse (4 of 5 seasons), so the model doesn't use them; see `evidence/matchup_ab_test.json`.
- **Games:** the model does **not** beat closing lines. Margin error is 10.18 for the model vs 9.81 for the market. Against the spread it wins 49.8–51.9%, below the 52.4% break-even. Treat game-model numbers as context only.

Two checks only run on your computer, because their sources were blocked from the build machine:
- `python -m beatnfl backtest --sleeper` compares Sleeper's published projections with the model and picks blend weights from that evidence.
- After each slate, `python -m beatnfl grade` grades the frozen pregame forecasts per source and suggests weights.

## Not yet verified (needs your network)

- DraftKings automatic import: the community-documented endpoints were blocked from the build machine. The doctor tests them on yours. If they fail, the inbox CSV fallback takes over.
- Exact export-ID mapping from the JSON route. Settled the first week you also drop the official CSV in the inbox.
- Sleeper response formats (schema-checked at runtime).

## Folders

- `~/BeatNFL/inbox` — files you drop in
- `~/BeatNFL/exports` — files to upload
- `~/BeatNFL/beatnfl.db` — slates, frozen forecasts, saved lineups, prop quotes
- `~/BeatNFL/evidence` — test results
- `~/BeatNFL/beatnfl.log` — technical detail for any failure

Tests: `.venv/bin/python -m pytest -q tests` (20 offline tests, including the acceptance cases from your spec).
