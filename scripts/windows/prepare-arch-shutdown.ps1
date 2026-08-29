<#
########################################################################
#################  REQUIRES FOUNDER GO ARCH NOW  #######################
########################################################################
FINAL Windows shutdown preparation for the Arch Linux wipe
(TASK BACKUP-SHUTDOWN-01).

*** DO NOT RUN THIS SCRIPT unless the FOUNDER has explicitly given the ***
*** GO for the Arch migration. Running it STOPS the live training     ***
*** worker listed in $AllowList below.                                ***

Order of operations:
  1. Verify all 5 backup zips exist in -TargetRoot and each zip's
     SHA256 matches its line in <GROUP>.SHA256SUMS.txt.
  2. Request the FINAL CHECKPOINT. The trainer
     (experiments/CV-G1-E0013-regressor/train_regressor.py) has NO
     signal handler: "checkpoint on exit" is the MANUAL act of letting
     it pass a periodic --ckpt-every save so
     experiments/r3/checkpoints/last.pt is current (optionally copying
     last.pt aside first). This script then verifies last.pt either
     CHANGED vs the recorded baseline (new final state captured) or
     MATCHES it (no progress since the audit/backup). Missing file or
     unverifiable hash = hard abort before any process is touched.
  3. Stop ONLY the PIDs hard-listed in $AllowList below (populated from
     scripts/windows/process-inventory.md, audited 2026-08-23). Every
     PID is identity-checked (process name AND command line) BEFORE
     anything is stopped; a single mismatch aborts with nothing killed.
  4. Confirm no train_regressor / kokoro / vocos python remains. Any
     UNLISTED match is REPORTED and fails the run -- this script NEVER
     kills generic python.exe.
  5. Write <TargetRoot>\FINAL-WINDOWS-STATE.json and print
     SAFE FOR ARCH WIPE only if every gate passed.

This script NEVER reboots, shuts down, formats, or touches boot
configuration; it contains no Stop-Computer / Restart-Computer /
disk-partitioning calls and never will.

Claude-session watchdogs note: watchdog Monitors live inside Claude
Code sessions (see LAUNCH_E0013.ps1 step [3/4]) and DIE WITH THE
SESSION. Close/stop all Claude sessions running monitors BEFORE running
this script; this script cannot see or kill them and does not try.

Usage (founder, interactive):
  pwsh -File scripts\windows\prepare-arch-shutdown.ps1 `
      -TargetRoot <backup-staging-dir> -Confirm

Exit codes: 0 safe · 2 refused/aborted by operator · 3 backup or
checkpoint verification failed · 4 allowlist identity mismatch ·
5 residual training process found · 10 checkpoint not confirmed.
########################################################################
#>
[CmdletBinding()]
param(
    [string]$TargetRoot = '.\backup-staging',
    # Mandatory go-flag. Nothing executes without it.
    [switch]$Confirm
)

$ErrorActionPreference = 'Stop'

function Resolve-Abs([string]$P) {
    if ([IO.Path]::IsPathRooted($P)) { return [IO.Path]::GetFullPath($P) }
    return [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $P))
}

$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot   = [IO.Path]::GetFullPath((Join-Path $scriptDir '..\..'))
$TargetRoot = Resolve-Abs $TargetRoot

# ======================================================================
# ALLOWLIST -- populated from scripts/windows/process-inventory.md
# (live audit 2026-08-23). DO NOT ADD ENTRIES WITHOUT A FRESH AUDIT.
# Identity checks are mandatory: name AND command line must match.
# ======================================================================
$AllowList = @(
    @{ Key = 'R3_WORKER';        Pid = 21728; ExpectName = 'python.exe'
       CmdMustMatch = 'train_regressor\.py.*--exp-id\s+r3'
       Desc = 'CV-G1-E0013 r3 training worker (child python)' },
    @{ Key = 'R3_PARENT_PYTHON'; Pid = 11616; ExpectName = 'python.exe'
       CmdMustMatch = 'train_regressor\.py.*--exp-id\s+r3'
       Desc = 'r3 parent python (owner/spawner of R3_WORKER)' },
    @{ Key = 'R3_CMD_WRAPPER';   Pid = 19968; ExpectName = 'cmd.exe'
       CmdMustMatch = 'train_regressor\.py'
       Desc = 'r3 cmd.exe launcher wrapper' }
)

# Baseline sha256 of experiments/r3/checkpoints/last.pt recorded during
# the 2026-08-23 inventory audit. Preferred over the value below when
# VOICE-CHECKPOINTS.manifest.json exists in TargetRoot.
$BaselineLastPtSha256 = 'cb0af758437f69a72e7c043bb94836a8a0ef88fa199d09e4aec10c2c581023b4'

$ExpectedGroups = @('VOICE-SOURCE','VOICE-STATE','VOICE-CHECKPOINTS','VOICE-UNIQUE-DATA','VOICE-PREVIEWS')

$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
function Add-Fail([string]$Msg) {
    $failures.Add($Msg)
    Write-Host ("  [FAIL] {0}" -f $Msg) -ForegroundColor Red
}
function Add-Warn([string]$Msg) {
    $warnings.Add($Msg)
    Write-Host ("  [warn] {0}" -f $Msg) -ForegroundColor Yellow
}

# ------------------------------------------------------------------ gate
if (-not $Confirm) {
    Write-Host ''
    Write-Host 'REQUIRES FOUNDER GO ARCH NOW.' -ForegroundColor Red
    Write-Host 'Re-run with -Confirm (and an interactive console) to proceed.'
    Write-Host ('Would verify backups in: {0}' -f $TargetRoot)
    Write-Host ('Would stop allowlisted PIDs: {0}' -f (($AllowList | ForEach-Object { "$($_.Key)=$($_.Pid)" }) -join ', '))
    exit 2
}
Write-Host ''
Write-Host '=== prepare-arch-shutdown ===' -ForegroundColor Cyan
Write-Host 'REQUIRES FOUNDER GO ARCH NOW -- explicit -Confirm received.'
$ans = Read-Host 'Type ARCH-GO to proceed (stops the allowlisted training PIDs after verification)'
if ($ans -cne 'ARCH-GO') {
    Write-Host 'Aborted by operator.' -ForegroundColor Yellow
    exit 2
}

$state = [ordered]@{
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    host          = $env:COMPUTERNAME
    repo_root     = $RepoRoot
    target_root   = $TargetRoot
}

# ---------------------------------------------------- 1. backup integrity
Write-Host ''
Write-Host '[1/6] verifying backup zips + SHA256SUMS sidecars...' -ForegroundColor Cyan
$backups = @()
foreach ($name in $ExpectedGroups) {
    $zipPath  = Join-Path $TargetRoot "$name.zip"
    $sidePath = Join-Path $TargetRoot "$name.SHA256SUMS.txt"
    $rec = [ordered]@{ group = $name; present = $false; verified = $false }
    if (-not (Test-Path -LiteralPath $zipPath)) {
        Add-Fail "missing backup archive: $zipPath"
        $backups += $rec; continue
    }
    $rec.present = $true
    if (-not (Test-Path -LiteralPath $sidePath)) {
        Add-Fail "missing sidecar: $sidePath"
        $backups += $rec; continue
    }
    $zipLine = Select-String -LiteralPath $sidePath -Pattern ('\s' + [regex]::Escape("$name.zip") + '\s*$') |
               Select-Object -First 1
    if (-not $zipLine) {
        Add-Fail "sidecar has no line for ${name}.zip"
        $backups += $rec; continue
    }
    $expected = ($zipLine.Line.Trim() -split '\s+')[0].ToLower()
    $actual   = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLower()
    $rec.zip_sha256 = $actual
    if ($actual -ne $expected) {
        Add-Fail ("hash mismatch {0}: sidecar={1} actual={2}" -f $name, $expected, $actual)
    } else {
        $rec.verified = $true
        Write-Host ("  [ok] {0} ({1} bytes)" -f $name, (Get-Item -LiteralPath $zipPath).Length)
    }
    $backups += $rec
}
$state.backups = $backups
if ($failures.Count -gt 0) {
    Write-Host ''
    Write-Host 'NOT SAFE FOR ARCH WIPE -- backup verification failed. Fix backups first.' -ForegroundColor Red
    exit 3
}

# ------------------------------------------------ 2. final checkpoint gate
Write-Host ''
Write-Host '[2/6] final checkpoint request...' -ForegroundColor Cyan
Write-Host @'
  The trainer has NO signal handler. "Checkpoint on exit" is MANUAL:
  let the trainer pass one more periodic --ckpt-every save so that
  experiments/r3/checkpoints/last.pt reflects the final state
  (optionally copy last.pt aside yourself first). On any later machine
  training resumes exactly from last.pt (--fresh absent => bitwise
  resume per E0003).
'@
$ck = Read-Host 'Has the final checkpoint save point passed? [y/N]'
if ($ck -notmatch '^[Yy]') {
    Write-Host 'Aborted: finalize the checkpoint first, then re-run.' -ForegroundColor Yellow
    exit 10
}
$lastPt = Join-Path $RepoRoot 'experiments\r3\checkpoints\last.pt'
if (-not (Test-Path -LiteralPath $lastPt)) {
    Add-Fail "final checkpoint missing: $lastPt"
    Write-Host 'NOT SAFE FOR ARCH WIPE.' -ForegroundColor Red
    exit 3
}
# prefer the backup-time baseline from the CHECKPOINTS manifest
$baseSha = $BaselineLastPtSha256
$baseSrc = 'inventory-audit-constant'
$mjPath = Join-Path $TargetRoot 'VOICE-CHECKPOINTS.manifest.json'
if (Test-Path -LiteralPath $mjPath) {
    try {
        $mj = ConvertFrom-Json (Get-Content -LiteralPath $mjPath -Raw)
        foreach ($f in @($mj.files)) {
            if ($f.path -eq 'experiments/r3/checkpoints/last.pt') {
                $baseSha = $f.sha256.ToLower(); $baseSrc = 'VOICE-CHECKPOINTS.manifest.json'
            }
        }
    } catch { Add-Warn "could not parse $mjPath; using audit constant baseline" }
}
$curSha   = (Get-FileHash -LiteralPath $lastPt -Algorithm SHA256).Hash.ToLower()
$curItem  = Get-Item -LiteralPath $lastPt
$changed  = ($curSha -ne $baseSha)
Write-Host ("  baseline ({0}): {1}" -f $baseSrc, $baseSha)
Write-Host ("  current           : {0}" -f $curSha)
if (-not $changed) {
    Write-Host '  [ok] last.pt MATCHES baseline (no training progress since baseline).' -ForegroundColor Green
} else {
    Write-Host '  [ok] last.pt CHANGED vs baseline (newer final state captured).' -ForegroundColor Green
}
$state.final_checkpoint = [ordered]@{
    path              = $lastPt
    sha256            = $curSha
    bytes             = $curItem.Length
    last_write_utc    = $curItem.LastWriteTimeUtc.ToString('o')
    baseline_sha256   = $baseSha
    baseline_source   = $baseSrc
    changed_vs_base   = $changed
    policy            = 'changed-or-matching accepted; missing/unreadable aborts'
}

# ------------------------------------------- 3. stop ONLY allowlisted PIDs
Write-Host ''
Write-Host '[3/6] stopping allowlisted PIDs (identity-checked)...' -ForegroundColor Cyan
Write-Host '  NOTE: Claude-session watchdog Monitors die with their session.'
Write-Host '  Close all Claude sessions running monitors BEFORE relying on a'
Write-Host '  quiet box; this script never touches session processes.'

# validate EVERY entry first; stop nothing until all identities check out
$targets  = @()
$preState = @()
foreach ($a in $AllowList) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($a.Pid)" -ErrorAction SilentlyContinue
    $pre = [ordered]@{ key = $a.Key; pid = $a.Pid; found = [bool]$p }
    if (-not $p) {
        Write-Host ("  [ok] {0} pid {1}: already gone" -f $a.Key, $a.Pid)
        $pre.status = 'already-gone'
        $preState += $pre
        continue
    }
    $pre.name = $p.Name
    if ($p.Name -ne $a.ExpectName) {
        Add-Fail ("identity mismatch {0} pid {1}: expected name '{2}', found '{3}'. NOTHING was stopped." -f $a.Key, $a.Pid, $a.ExpectName, $p.Name)
        $pre.status = 'mismatch'
        $preState += $pre
        continue
    }
    if ($p.CommandLine -notmatch $a.CmdMustMatch) {
        Add-Fail ("identity mismatch {0} pid {1}: command line does not match required pattern. NOTHING was stopped." -f $a.Key, $a.Pid)
        $pre.status = 'mismatch'
        $preState += $pre
        continue
    }
    $pre.status = 'verified'
    $pre.cmdline = ($p.CommandLine -replace '\s+', ' ')
    $preState += $pre
    $targets += @{ Key = $a.Key; Pid = $a.Pid; Desc = $a.Desc }
}
$state.allowlist_precheck = $preState
if ($failures.Count -gt 0) {
    Write-Host 'NOT SAFE FOR ARCH WIPE -- allowlist identity check failed; no process was stopped.' -ForegroundColor Red
    exit 4
}
# stop child-first so the parent cannot respawn work underneath us
foreach ($t in $targets) {
    Write-Host ("  stopping {0} pid {1} ({2})" -f $t.Key, $t.Pid, $t.Desc)
    try { Stop-Process -Id $t.Pid -Force -ErrorAction Stop } catch {
        Add-Warn ("Stop-Process {0}: {1}" -f $t.Pid, $_.Exception.Message)
    }
    $goneBy = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $goneBy) {
        if (-not (Get-Process -Id $t.Pid -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }
    if (Get-Process -Id $t.Pid -ErrorAction SilentlyContinue) {
        Add-Fail ("pid {0} ({1}) still alive 30s after Stop-Process" -f $t.Pid, $t.Key)
    } else {
        Write-Host ("  [ok] pid {0} exited" -f $t.Pid)
    }
}

# --------------------------------------------- 4. residual trainer scan
Write-Host ''
Write-Host '[4/6] scanning for residual training processes...' -ForegroundColor Cyan
$residual = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'py_%' OR Name LIKE 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'train_regressor|kokoro|vocos' })
$state.residual_matches = @($residual | ForEach-Object {
    [ordered]@{ pid = $_.ProcessId; name = $_.Name; cmdline = ($_.CommandLine -replace '\s+', ' ') } })
if ($residual.Count -gt 0) {
    foreach ($r in $residual) {
        Add-Fail ("residual training process pid {0} ({1}): {2}" -f $r.ProcessId, $r.Name, ($r.CommandLine -replace '\s+', ' '))
    }
    Write-Host '  This script NEVER kills generic python.exe. Investigate the PIDs above,' -ForegroundColor Yellow
    Write-Host '  add vetted entries to the allowlist with fresh identity data, re-run.' -ForegroundColor Yellow
    Write-Host 'NOT SAFE FOR ARCH WIPE.' -ForegroundColor Red
    exit 5
}
Write-Host '  [ok] no train_regressor / kokoro / vocos python remains'

# ----------------------------------------------- 5. FINAL-WINDOWS-STATE.json
Write-Host ''
Write-Host '[5/6] writing FINAL-WINDOWS-STATE.json...' -ForegroundColor Cyan
$gitCommit = $null; $gitDirty = $null; $gitBranch = $null
try {
    $gitCommit = (& git -C $RepoRoot rev-parse HEAD | Select-Object -First 1)
    $gitBranch = (& git -C $RepoRoot rev-parse --abbrev-ref HEAD | Select-Object -First 1)
    $gitDirty  = (@(& git -C $RepoRoot status --porcelain)).Count
} catch { Add-Warn "git queries failed: $($_.Exception.Message)" }

$ckptInfo = @()
foreach ($cp in @('experiments\r3\checkpoints\best.pt',
                  'experiments\r3\checkpoints\last.pt',
                  'experiments\r2\checkpoints\best.pt')) {
    $full = Join-Path $RepoRoot $cp
    if (Test-Path -LiteralPath $full) {
        $i = Get-Item -LiteralPath $full
        $ckptInfo += [ordered]@{
            path           = ($cp -replace '\\', '/')
            bytes          = $i.Length
            sha256         = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLower()
            last_write_utc = $i.LastWriteTimeUtc.ToString('o')
        }
    } else {
        $ckptInfo += [ordered]@{ path = ($cp -replace '\\', '/'); present = $false }
    }
}

$teacher = $null
$metaPath = Join-Path $RepoRoot 'data\manifests\teacher-kokoro.meta.json'
if (Test-Path -LiteralPath $metaPath) {
    try {
        $meta = ConvertFrom-Json (Get-Content -LiteralPath $metaPath -Raw)
        $shards = @()
        for ($s = 0; $s -lt 2; $s++) {
            $sp = Join-Path $RepoRoot ("experiments\CV-G1-E0012-teacher-gen\shard{0}.progress.json" -f $s)
            $row = [ordered]@{ shard = $s; status = 'unknown' }
            if (Test-Path -LiteralPath $sp) {
                $pj = ConvertFrom-Json (Get-Content -LiteralPath $sp -Raw)
                $alive = [bool](Get-Process -Id $pj.pid -ErrorAction SilentlyContinue)
                $row.status = $(if ($alive) { 'RUNNING' } else { 'DONE' })
                $row.pid = $pj.pid; $row.pid_alive = $alive
                $row.kept = $pj.kept; $row.audio_hours = $pj.audio_hours
                $row.updated = $pj.updated
            }
            $shards += $row
        }
        $teacher = [ordered]@{
            dataset         = $meta.dataset
            meta_updated    = $meta.last_update
            kept            = $meta.totals.kept
            rows_in_manifest= $meta.totals.rows_in_manifest
            audio_hours     = $meta.totals.audio_hours
            seen            = $meta.totals.seen
            processed       = $meta.totals.processed
            rejected_total  = (($meta.totals.rejected_by_reason.PSObject.Properties | Measure-Object Value -Sum).Sum)
            shards          = $shards
        }
    } catch { Add-Warn "teacher meta parse failed: $($_.Exception.Message)" }
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Add-Warn 'nvidia-smi not on PATH (Windows Home): GPU state unverified; process table used as source of truth.'
}

$state.git_commit = $gitCommit
$state.git_branch = $gitBranch
$state.git_dirty_files = $gitDirty
$state.checkpoints = $ckptInfo
$state.teacher_corpus = $teacher
$state.warnings = $warnings

$json = ConvertTo-Json -InputObject $state -Depth 6
[IO.File]::WriteAllText((Join-Path $TargetRoot 'FINAL-WINDOWS-STATE.json'), $json,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host ("  written: {0}" -f (Join-Path $TargetRoot 'FINAL-WINDOWS-STATE.json'))

# ------------------------------------------------------------ 6. verdict
Write-Host ''
if ($failures.Count -eq 0) {
    Write-Host '==================== SAFE FOR ARCH WIPE ====================' -ForegroundColor Green
    Write-Host 'Backups verified, final checkpoint captured, allowlisted'
    Write-Host 'training processes stopped, no residual trainer found.'
    Write-Host 'Founder may now move backups off-machine and wipe C:.'
    exit 0
}
Write-Host 'NOT SAFE FOR ARCH WIPE -- unresolved failures above:' -ForegroundColor Red
$failures | ForEach-Object { Write-Host ("  - {0}" -f $_) -ForegroundColor Red }
exit 3
