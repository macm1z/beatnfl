@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run "Install BeatNFL.bat" first. & pause & exit /b 1)
.venv\Scripts\python -m beatnfl app
