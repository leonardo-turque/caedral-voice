# LAUNCH_E0013.ps1 — one-shot transition: stop aligner -> write durations -> launch regressor
# Preconditions: ALIGN-01 epoch>=1 complete or manually approved for early-stop.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..

Write-Host "[1/4] stopping aligner if running..."
Get-Process -Id 41464 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

Write-Host "[2/4] writing durations for all kept utterances..."
& .venv\Scripts\python.exe -X utf8 experiments\CV-G1-E0011-ctc-aligner\write_durations.py 2>&1 | Tee-Object -FilePath experiments\CV-G1-E0011-ctc-aligner\write_durations.log | Select-Object -Last 5
if (-not (Test-Path data\manifests\librittsr-clean100.durations.jsonl)) {
    throw "durations file missing - ABORT"
}
$n = (Get-Content data\manifests\librittsr-clean100.durations.jsonl | Measure-Object -Line).Lines
Write-Host "durations rows: $n"

Write-Host "[3/4] arming regressor watchdog..."
# watchdog launched separately by coordinator Monitor

Write-Host "[4/4] launching CV-G1-E0013 regressor (detached)..."
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$script = Join-Path (Get-Location) 'experiments\CV-G1-E0013-regressor\train_regressor.py'
$out = Join-Path (Get-Location) 'experiments\CV-G1-E0013-regressor\logs'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$p = Start-Process -FilePath $py -ArgumentList @(
    '-X','utf8', $script,
    '--manifests','librittsr-clean100,teacher-kokoro',
    '--durations','data/manifests/librittsr-clean100.durations.jsonl',
    '--steps','15000','--threads','6',
    '--exp-id','CV-G1-E0013-regressor-r1'
) -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$out\r1.out.log" -RedirectStandardError "$out\r1.err.log"
Write-Host "REGRESSOR LAUNCHED PID $($p.Id)"
$p.Id | Set-Content experiments\CV-G1-E0013-regressor\r1.pid
