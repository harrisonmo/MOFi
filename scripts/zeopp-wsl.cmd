@echo off
rem Run a Zeo++ "network" binary built inside WSL from Windows.
rem Point XTAL_ZEOPP at this file and ZEOPP_WSL_BINARY at the Linux path.
rem Relative file arguments resolve in the caller's directory, which WSL maps.
if "%ZEOPP_WSL_DISTRO%"=="" set ZEOPP_WSL_DISTRO=Ubuntu
if "%ZEOPP_WSL_BINARY%"=="" (
  echo ZEOPP_WSL_BINARY is not set 1>&2
  exit /b 2
)
wsl.exe -d %ZEOPP_WSL_DISTRO% --exec "%ZEOPP_WSL_BINARY%" %*
exit /b %ERRORLEVEL%
