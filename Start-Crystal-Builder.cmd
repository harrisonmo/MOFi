@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
rem Requires the gui extra: uv sync --extra gui
".venv\Scripts\crystal-builder.exe" %*
