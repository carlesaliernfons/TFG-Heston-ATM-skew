@echo off
rem TFG - double-click this file to schedule the daily download of option data (Monday-Friday, 23:00).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure_task.ps1"
echo.
pause
