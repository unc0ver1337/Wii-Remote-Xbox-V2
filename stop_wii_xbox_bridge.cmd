@echo off
setlocal
set "WII_BRIDGE_SCRIPT=%~dp0wii_xbox_bridge_v2.py"
powershell -NoProfile -Command "$target=$env:WII_BRIDGE_SCRIPT; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -and $_.CommandLine.Contains($target) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
endlocal
