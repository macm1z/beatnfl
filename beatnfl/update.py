"""Self-update from the GitHub repository set in Settings (update_repo = "owner/name").

  1. Ask GitHub for the newest commit on main. Nothing to do if it matches the installed one.
  2. Download that commit's source zip and back up the current code.
  3. Copy in the new code. Your data in ~/BeatNFL and the app's Python (.venv) are never touched.
  4. Reinstall packages only if requirements.txt changed (checksum-locked, as at setup).
  5. Run the self-tests. If anything fails, restore the backup and report why.
"""

from __future__ import annotations

import filecmp
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

from .errors import StepError

APP_DIR = Path(__file__).resolve().parent.parent
STAMP = APP_DIR / ".installed_commit"
CODE = ["beatnfl", "app.py", "requirements.txt", "requirements.in", "README.md", "tests", "evidence", ".streamlit",
        "Install BeatNFL.command", "Start BeatNFL.command", "Install BeatNFL.bat", "Start BeatNFL.bat", "launch.sh",
        "make_mac_app.sh", ".gitignore"]
STEP = "Update"


def _headers(token: str) -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "BeatNFL-updater"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def installed() -> str:
    return STAMP.read_text().strip() if STAMP.exists() else ""


def latest(repo: str, token: str = "") -> dict:
    if not repo or "/" not in repo:
        raise StepError(STEP, "No update source set.", "", "Enter your GitHub repository (owner/name) in Settings.")
    r = requests.get(f"https://api.github.com/repos/{repo}/commits/main", headers=_headers(token), timeout=20)
    if r.status_code == 404:
        raise StepError(STEP, f"GitHub can't find {repo}.",
                        "The name is wrong, or the repository is private and no token is set.",
                        "Check the repository name in Settings; for a private repository add a read-only token.")
    if r.status_code in (401, 403):
        raise StepError(STEP, "GitHub refused the request.", "The token is wrong or expired, or the rate limit was hit.",
                        "Create a new read-only token for this repository and paste it into Settings.")
    r.raise_for_status()
    c = r.json()
    return {"sha": c["sha"], "date": c["commit"]["committer"]["date"], "message": c["commit"]["message"].splitlines()[0]}


def check(repo: str, token: str = "") -> dict:
    new = latest(repo, token)
    new["update_available"] = new["sha"] != installed()
    return new


def _uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for p in (Path.home() / ".local/bin/uv", Path.home() / ".local/bin/uv.exe", Path("/opt/homebrew/bin/uv")):
        if p.exists():
            return str(p)
    return None


def _venv_python() -> str:
    win = APP_DIR / ".venv" / "Scripts" / "python.exe"
    return str(win if win.exists() else APP_DIR / ".venv" / "bin" / "python")


def apply(repo: str, token: str = "", progress=lambda m: None) -> dict:
    info = latest(repo, token)
    if info["sha"] == installed():
        return {**info, "changed": False, "note": "Already up to date."}
    progress("Downloading the update…")
    r = requests.get(f"https://api.github.com/repos/{repo}/zipball/{info['sha']}", headers=_headers(token), timeout=60)
    r.raise_for_status()
    tmp = Path(tempfile.mkdtemp(prefix="beatnfl_update_"))
    src_root = tmp / "src"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for m in z.namelist():  # refuse paths that try to escape the folder
            if m.startswith("/") or ".." in Path(m).parts:
                raise StepError(STEP, "The update download looks malformed.", "", "Nothing was changed. Try again later.")
        z.extractall(src_root)
    roots = [p for p in src_root.iterdir() if p.is_dir()]
    src = roots[0] if len(roots) == 1 else src_root
    if not (src / "beatnfl").is_dir() or not (src / "app.py").exists():
        raise StepError(STEP, "The repository doesn't contain the BeatNFL app.", f"Checked {repo}.",
                        "Check the repository name in Settings.")

    progress("Backing up the current version…")
    backup = tmp / "backup"
    backup.mkdir()
    for name in CODE:
        p = APP_DIR / name
        if p.is_dir():
            shutil.copytree(p, backup / name)
        elif p.exists():
            shutil.copy2(p, backup / name)
    req_changed = not (APP_DIR / "requirements.txt").exists() or not filecmp.cmp(
        APP_DIR / "requirements.txt", src / "requirements.txt", shallow=False)

    def restore():
        for name in CODE:
            p = APP_DIR / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            b = backup / name
            if b.is_dir():
                shutil.copytree(b, p)
            elif b.exists():
                shutil.copy2(b, p)

    try:
        progress("Installing the new code…")
        for name in CODE:
            s, d = src / name, APP_DIR / name
            if not s.exists():
                continue
            if d.is_dir():
                shutil.rmtree(d)
            elif d.exists():
                d.unlink()
            shutil.copytree(s, d) if s.is_dir() else shutil.copy2(s, d)
        for f in ("Install BeatNFL.command", "Start BeatNFL.command", "launch.sh", "make_mac_app.sh"):
            if (APP_DIR / f).exists():
                (APP_DIR / f).chmod(0o755)
        if req_changed:
            progress("Updating packages (checksum-locked)…")
            uv = _uv()
            if not uv:
                raise StepError(STEP, "Packages changed but uv wasn't found.", "", "Run the installer once more.")
            out = subprocess.run([uv, "pip", "install", "--python", str(APP_DIR / ".venv"), "--require-hashes",
                                  "-r", str(APP_DIR / "requirements.txt")], capture_output=True, text=True)
            if out.returncode != 0:
                raise StepError(STEP, "Package install failed.", out.stderr[-300:], "The previous version was restored.")
        progress("Running self-tests…")
        t = subprocess.run([_venv_python(), "-m", "pytest", "-q", "tests"], cwd=APP_DIR, capture_output=True, text=True)
        if t.returncode != 0:
            raise StepError(STEP, "The new version failed its self-tests.", t.stdout[-300:],
                            "The previous version was restored. Send me the message above.")
    except Exception:
        restore()
        raise
    STAMP.write_text(info["sha"])
    if sys.platform == "darwin" and (APP_DIR / "make_mac_app.sh").exists():
        subprocess.run(["/bin/bash", str(APP_DIR / "make_mac_app.sh")], capture_output=True)  # refresh the Mac app icon
    shutil.rmtree(tmp, ignore_errors=True)
    return {**info, "changed": True, "packages_updated": req_changed,
            "note": "Updated. Click Restart below (or quit and reopen BeatNFL) to load the new version."}


if __name__ == "__main__":  # python -m beatnfl.update owner/name [token]
    print(json.dumps(apply(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "", print), indent=1))


def restart() -> bool:
    """Restart the background app so new code loads (macOS/Linux). Returns False where unsupported."""
    if sys.platform.startswith("win") or not (APP_DIR / "launch.sh").exists():
        return False
    # "[s]treamlit" matches the app but not this restart command's own text (a plain pattern made pkill
    # stop the restart itself, leaving the old version running)
    cmd = (f"sleep 1; pkill -f '[s]treamlit run {APP_DIR}/app.py'; sleep 3; /bin/bash '{APP_DIR / 'launch.sh'}'")
    subprocess.Popen(["/bin/bash", "-c", cmd], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True
