@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "%~dp0wii_xbox_bridge_v2.py" %*
if errorlevel 1 pause
endlocal
