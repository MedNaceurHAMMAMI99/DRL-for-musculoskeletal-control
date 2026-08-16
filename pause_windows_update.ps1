<#
    Stop Windows Update from restarting the machine mid-run.

    Why this exists: on 2026-08-14 a forced update restart destroyed a ~900k-step
    SAC run, 5 minutes short of finishing. Active hours were 10:00-03:00, so the
    restart fired at 03:59 — the first moment Windows was allowed to take the
    machine. Long training runs live precisely in that window.

    Active hours cannot fix this: Windows caps the span at 18 hours, so some part
    of the day is always exposed. Pausing updates outright is the only reliable
    cover for a multi-day grid.

    RUN ELEVATED. Reverting is not optional housekeeping — a machine with updates
    paused indefinitely is a machine missing security patches. Revert as soon as
    the grid finishes (same note applies to the powercfg sleep-disable).

    Usage:
        powershell -ExecutionPolicy Bypass -File pause_windows_update.ps1 -Days 7
        powershell -ExecutionPolicy Bypass -File pause_windows_update.ps1 -Revert
#>
param(
    [int]$Days = 7,
    [switch]$Revert
)

$ErrorActionPreference = 'Stop'

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$pr = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Must run elevated (Run as administrator)."
    exit 1
}

$UX = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
$AU = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'

if ($Revert) {
    foreach ($n in 'PauseUpdatesExpiryTime','PauseFeatureUpdatesStartTime',
                   'PauseFeatureUpdatesEndTime','PauseQualityUpdatesStartTime',
                   'PauseQualityUpdatesEndTime') {
        try { Remove-ItemProperty -Path $UX -Name $n -ErrorAction Stop } catch {}
    }
    try { Remove-ItemProperty -Path $AU -Name 'NoAutoRebootWithLoggedOnUsers' -ErrorAction Stop } catch {}
    Write-Host "Reverted: updates un-paused, auto-restart policy removed."
    Write-Host "Also revert the sleep setting if the grid is done:"
    Write-Host "    powercfg /change standby-timeout-ac 30"
    exit 0
}

$start = (Get-Date).ToUniversalTime()
$end   = $start.AddDays($Days)
$fmt   = 'yyyy-MM-ddTHH:mm:ssZ'

if (-not (Test-Path $UX)) { New-Item -Path $UX -Force | Out-Null }
if (-not (Test-Path $AU)) { New-Item -Path $AU -Force | Out-Null }

Set-ItemProperty -Path $UX -Name 'PauseUpdatesExpiryTime'        -Value $end.ToString($fmt)   -Type String
Set-ItemProperty -Path $UX -Name 'PauseFeatureUpdatesStartTime'  -Value $start.ToString($fmt) -Type String
Set-ItemProperty -Path $UX -Name 'PauseFeatureUpdatesEndTime'    -Value $end.ToString($fmt)   -Type String
Set-ItemProperty -Path $UX -Name 'PauseQualityUpdatesStartTime'  -Value $start.ToString($fmt) -Type String
Set-ItemProperty -Path $UX -Name 'PauseQualityUpdatesEndTime'    -Value $end.ToString($fmt)   -Type String

# Belt and braces: even when an update does install, do not auto-restart while a
# user is logged on. Honoured by the Windows Update agent independently of the
# pause window above.
Set-ItemProperty -Path $AU -Name 'NoAutoRebootWithLoggedOnUsers' -Value 1 -Type DWord

Write-Host "Windows Update paused until $($end.ToString($fmt)) (UTC), $Days days."
Write-Host "Auto-restart with a logged-on user disabled."
Write-Host ""
Write-Host "Verify in Settings > Windows Update (should read 'Resume updates')."
Write-Host "When the grid is done, revert with:  -Revert"
