@echo off
setlocal
cd /d "%~dp0\.."
set PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe
if exist "%PS_EXE%" (
  "%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "desktop_app\build_windows_exe.ps1"
) else (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "desktop_app\build_windows_exe.ps1"
)
pause
