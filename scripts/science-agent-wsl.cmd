@echo off
rem Run the science agent on Windows (existing Codex CLI login) with the science hub inside WSL,
rem where GPAW, CP2K, RASPA3 and Zeo++ are installed. Prefer --task @file for task text with
rem characters that cmd treats specially. Example:
rem   scripts\science-agent-wsl.cmd --agent-config configs\science-agent-sol-smoke.json --run runs\my-run --task @task.txt
setlocal
set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%;%ROOT%\Crystal-Builder"
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
"%ROOT%\.venv\Scripts\python.exe" -m rsi.science.agent --hub-runner wsl --hub-config "%ROOT%\configs\science-hub-wsl.json" %*
exit /b %ERRORLEVEL%
