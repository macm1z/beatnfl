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

Tabs: **This week** (pool and lineups) · **Ask** (questions in plain English, answered from the app's data) · **Saved lineups** (rename, delete, remove duplicates, filter by player) · **Games** (spread, moneyline and total chances: market vs model) · **Props** · **My Yahoo team** · **Evidence** · **Load data** · **Data status**.

1. **⟳ Refresh** (top right). This loads and checks the slate, then injuries, lines and projections, then blends them into the **Combined** projection and freezes the forecast. Anything that needs attention shows in a fold-out on *This week*; the full step log is on *Load data*, which is also where you pick a different slate, upload a salary file or paste your own projections.
2. **Review the pool.** Filter by position or team, lock or exclude players. Hover a column name to see what it means.
3. Pick the **Contest** type above the build controls. **Cash** (50/50, double-up, head-to-head) gives the highest-projected lineup with no stack required; **Single-entry tournament** (the default) is the same with a QB stack. Both return your best lineup as #1 plus two backups that change 2+ players, for late news. **Multi-entry tournament** builds a varied set (20 lineups, 35% max exposure, 3 changes, randomness). Your choice is remembered.
   **⚙ Build lineups**, then **💾 Save** if you want to keep them. Lineups are ranked best projection first. Saving writes the DK upload CSV and a readable CSV to `~/BeatNFL/exports`; an identical set is never saved twice. Download buttons work without saving.
4. Upload `DK_upload_….csv` on DraftKings yourself (Upload Lineups).

**FantasyPros API (recommended if you're an MVP/HOF member):** paste your production key from secure.fantasypros.com/api-keys into **Settings → FantasyPros API key**. Every Refresh then pulls FantasyPros' weekly projections and converts their projected stats to DraftKings points for the **Yours** column — no pasting. A table you paste on *Load data* replaces them for that refresh. A free (sample-data) key is detected and reported, never used as if complete.

**Tested weights (Evidence → Run the source test):** scores FantasyPros', Sleeper's and the model's past weekly projections against what actually happened and picks the best blend for each position, plus how much of FantasyPros' stat lines to mix into the props forecasts. **Use these weights** applies them; **Back to simple weights** undoes it. If your key can't fetch past weeks, the test runs on Sleeper and the model, and FantasyPros is graded live instead: every Refresh freezes the numbers, and after 3 finished weeks *Live check → Grade live weights* suggests weights from those real pregame projections.

**Optional inputs, all dropped in `~/BeatNFL/inbox`:**
- **Paid projections:** drop your site's DK projection CSV in the inbox. Columns are detected and shown.
- **Fallback slate:** if DraftKings blocks the automatic download, drop `DKSalaries.csv` in the inbox. The app uses it automatically and says so. To get it: DraftKings → any NFL Classic contest → Export to CSV.
- **Odds API key** (free at the-odds-api.com): add it in Settings for live lines and props. Without it, lineups still work.

**Sportsbook lines in the model (step 4d):** with the Odds API key set, each Refresh reads the books' player prop lines (yards, receptions, anytime TD) for the slate and moves the model's DraftKings projection toward them — the *Book lines* column shows by how much. On unseen 2025 weeks the lines out-predicted the model on every yardage stat, and folding them in lowered the DK-points miss. About 5 credits per game, reused for 3 hours. Turn off in Settings.

**Ask:** type a question on the *Ask* tab — "best value RBs under $6K?", "why is Bijan projected so high?", "build my cash lineup with Bo Nix locked", "how did my lineups do last week?". Claude answers from what the app has loaded (pool, projection sources, injuries, lines, props, saved lineups and grades, Evidence) and says when something isn't loaded. Lineups it builds can be sent to *This week* to save or download. Needs an Anthropic API key (console.anthropic.com → API keys) in Settings; billed per question by Anthropic, and each answer shows its cost. Choose the model in Settings (Sonnet 5.5 default; Haiku 5.5 is cheapest).

**Grade your lineups:** *Saved lineups → open a set → 📊 Grade against actual results* (after the games are final, usually Tuesday) shows each lineup's real DraftKings points vs projection, every player's real score and how many lineups he was in, and the best lineup that was possible on that slate in hindsight.

**Game-day check:** after you save lineups, *This week* flags any player in them who is later ruled out (or Questionable). With Automatic checks on (sidebar), the Mac re-checks after each inactive list posts (Thu 7 PM, Sun 11:50 AM / 3:10 PM / 7 PM, Mon 7 PM) and notifies you. CLI: `python -m beatnfl gamecheck`. Practice reports were tested as model inputs and did not help (Evidence), so they aren't used.

**Picks (research only, experimental):** the 2025 backtest on real lines found no proven edge for the model on props. The one hint — Unders where the model is 10+ points below the market — is what *Picks → Find this week's picks* now logs (FantasyPros must agree when your key is set), at the best price across books, so the rule is tested on games it has never seen. Each pick is logged at that price and graded after the games: result, units at 1 per pick, and **closing-line value** (your price vs the market's last price before kickoff — the quick, honest test of whether an edge is real). On a Mac, turn on *Automatic checks* to run the list Wednesday 7 PM (with a notification) and record closing prices before Thursday, Sunday and Monday games. Nothing is ever bet. CLI: `python -m beatnfl midweek`, `python -m beatnfl close`.

**Props backtest on real lines (paid Odds API plan):** *Evidence → Props on real sportsbook lines* replays a past season with the model as it stood before that season, using each prop's real Wednesday-evening and closing prices from every US book. It reports the Picks rule's record, ROI and closing-line value at 4/6/8/10-point thresholds. 2025 with pass/rush/receiving yards costs about 16,000 credits; a credit floor (default 4,000) pauses it before weekly use is starved, and a paused run resumes later without paying twice. Credits left show under Settings. CLI: `python -m beatnfl backtest-props --season 2025`.

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
| Projections | FantasyPros API (or pasted/CSV table), Sleeper, in-house model (blend weights in Settings) | Remaining sources renormalized |
| History, lines, results | nflverse (GitHub releases) | Cached copy, shown as STALE |
| Live lines, props | The Odds API | nflverse lines; props hidden |

Rules:
- Missing data is never treated as 0.
- Old salary files are refused rather than swapped in silently.
- Cached data is always labeled stale.
- DK export IDs from the JSON route stay "unverified" until checked against an official CSV in the inbox.

## In-house model (pm-1.6)

Gradient-boosted forecasts of DK points and of each prop stat:
- **Player inputs:** recent form (exponentially weighted), target and rush share, snap share (pm-1.3), the starting QB — who's listed to start, whether it's a change, and his record (pm-1.4; expected starter from Sleeper's depth chart before games) — efficiency, and opportunity vacated by inactive teammates.
- **Game inputs:** team pace, game line (implied total and spread), and opponent defense.

Settings (pm-1.6) were picked by scoring 19 variants on 2022–2025: history back to 2012, smaller trees and a 20% blend with a linear model won on every target, every season.

Each player also gets his own **Floor** and **Ceiling** (15th/85th percentile, pm-1.5) — boom-or-bust players get wider ranges.

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

Tests: `.venv/bin/python -m pytest -q tests` (49 offline tests; `BEATNFL_FULL=1` adds 6 end-to-end tests that click every button).
