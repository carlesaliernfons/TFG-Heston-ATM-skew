# TFG - registers a Windows scheduled task that runs download_daily.py
# every weekday (Monday to Friday) at 23:00, after the US market has closed.
# If the computer is off or asleep at 23:00, the task runs as soon as it is available again.
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $dir "download_daily.py"

# pythonw.exe = Python without a console window (the script writes everything to log.txt)
$candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\pythonw.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\pythonw.exe"
)
$py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source }
}
if (-not $py) { Write-Host "ERROR: pythonw.exe not found. Is Python installed?"; exit 1 }

$action   = New-ScheduledTaskAction -Execute $py -Argument ('"' + $script + '"') -WorkingDirectory $dir
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "23:00"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName "TFG - download options" -Action $action -Trigger $trigger -Settings $settings `
    -Description "TFG: daily snapshot of S&P 500 option chains from Yahoo Finance (download_daily.py)" -Force | Out-Null

Write-Host ""
Write-Host "OK: scheduled task 'TFG - download options' registered (Monday-Friday at 23:00)."
Write-Host "Python: $py"
Write-Host "Script: $script"
Write-Host "Log:    $(Join-Path $dir 'log.txt')"
