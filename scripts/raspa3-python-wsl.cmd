@echo off
rem Run Python with the isolated, project-compatible RASPA3 bindings in WSL.
if "%RASPA3_WSL_DISTRO%"=="" set RASPA3_WSL_DISTRO=Ubuntu
if "%RASPA3_WSL_PYTHON%"=="" (
  echo Set RASPA3_WSL_PYTHON to your Linux installation path 1>&2
  exit /b 2
)
wsl.exe -d %RASPA3_WSL_DISTRO% --exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "%RASPA3_WSL_PYTHON%" %*
exit /b %ERRORLEVEL%
