# start-rescue-orchestrator.ps1 — launch detached rescue orchestrator
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$script = Join-Path (Get-Location) 'scripts\rescue_orchestrator.py'
$out = Join-Path (Get-Location) 'artifacts\rescue-state\orchestrator.stdout.log'
$err = Join-Path (Get-Location) 'artifacts\rescue-state\orchestrator.stderr.log'
$p = Start-Process -FilePath $py -ArgumentList @('-X','utf8',$script) `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
"ORCHESTRATOR LAUNCHED PID $($p.Id)"
$p.Id | Set-Content 'artifacts\rescue-state\orchestrator.pid'
