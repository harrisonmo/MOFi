@echo off
rem Run the isolated RASPA3 CLI in WSL, preserving the caller's working directory.
if "%RASPA3_WSL_DISTRO%"=="" set RASPA3_WSL_DISTRO=Ubuntu
if "%RASPA3_WSL_HOME%"=="" (
  echo Set RASPA3_WSL_HOME to your Linux installation path 1>&2
  exit /b 2
)
wsl.exe -d %RASPA3_WSL_DISTRO% --exec env "RASPA_DIR=%RASPA3_WSL_HOME%/usr/share/raspa3" "LD_LIBRARY_PATH=%RASPA3_WSL_HOME%/usr/lib/x86_64-linux-gnu/blas64:%RASPA3_WSL_HOME%/usr/lib/x86_64-linux-gnu/lapack64:%RASPA3_WSL_HOME%/usr/lib/llvm-14/lib" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "%RASPA3_WSL_HOME%/usr/bin/raspa3" %*
exit /b %ERRORLEVEL%
