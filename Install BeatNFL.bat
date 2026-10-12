@echo off
cd /d "%~dp0"
echo == BeatNFL setup ==
where uv >nul 2>nul
if %errorlevel% neq 0 (
  if not exist "%USERPROFILE%\.local\bin\uv.exe" (
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.32/install.ps1 | iex"
  )
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
uv venv --python 3.13 .venv --allow-existing || goto :fail
uv pip install --python .venv --require-hashes -r requirements.txt || goto :fail
.venv\Scripts\python -m pytest -q tests
.venv\Scripts\python -m beatnfl doctor
echo Setup complete. Double-click "Start BeatNFL.bat" to open the app.
pause
exit /b 0
:fail
echo Setup failed - see the message above.
pause
exit /b 1
