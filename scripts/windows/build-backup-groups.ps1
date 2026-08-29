<#
========================================================================
 build-backup-groups.ps1 -- caedral-voice Windows evacuation backups
========================================================================
BUILDER script (TASK BACKUP-SHUTDOWN-01). READ-ONLY against the repo:
reads repository files and writes ONLY into -TargetRoot. Never stops
processes, never modifies the repo tree. Safe to run while the E0013
training worker (see scripts/windows/process-inventory.md) is live --
except files being actively written (metrics, checkpoints) may be
captured mid-update; authoritative final builds should be produced
after prepare-arch-shutdown.ps1 has quiesced the box, or accepted as
near-final.

Groups (5):
  VOICE-SOURCE        git archive HEAD (tracked files only, so
                      .gitignore effectively applies)
  VOICE-STATE         docs/**, experiments/**/metrics.jsonl,
                      experiments/**/conclusion.md, EXPERIMENT-INDEX.md,
                      artifacts/caedral-voice-v1/provenance/**
  VOICE-CHECKPOINTS   experiments/r3/checkpoints/{best.pt,last.pt},
                      experiments/r2/checkpoints/best.pt
  VOICE-UNIQUE-DATA   data/processed/teacher-kokoro/**,
                      data/manifests/** (.jsonl/.json only)
  VOICE-PREVIEWS      artifacts/caedral-voice-v1/voices/**,
                      artifacts/latest-preview/**

Per group written into TargetRoot:
  <GROUP>.zip             zip with forward-slash entry paths preserved
                          relative to the repo root
  <GROUP>.SHA256SUMS.txt  line 1 = "<zip-sha256>  <GROUP>.zip";
                          following lines = per-member sha256 keyed by
                          in-zip path (sha256sum-style "<hash>  <path>")
  <GROUP>.manifest.json   generated_utc, git commit, per-file size +
                          sha256 (SOURCE group: tracked-file list with
                          git blob hashes instead)

Usage:
  pwsh -File scripts\windows\build-backup-groups.ps1 `
      [-SourceRoot <repo-root>] [-TargetRoot .\backup-staging] `
      [-Groups SOURCE,STATE,CHECKPOINTS,UNIQUE-DATA,PREVIEWS] `
      [-ManifestOnly]

D:\ placeholder: the SSD to be wiped is C:. Once the external target
is mounted, pass it explicitly, e.g.:
  -TargetRoot D:\caedral-evac-20260823

Global exclusions (all groups): __pycache__ directories, *.tmp files.
NOTE: EXPERIMENT-INDEX.md lives at experiments\EXPERIMENT-INDEX.md in
this repo (audited 2026-08-23); both spellings are included so the
group stays correct wherever it lands.
========================================================================
#>
[CmdletBinding()]
param(
    [string]$SourceRoot = '',
    [string]$TargetRoot = '.\backup-staging',
    [ValidateSet('SOURCE','STATE','CHECKPOINTS','UNIQUE-DATA','PREVIEWS')]
    [string[]]$Groups = @('SOURCE','STATE','CHECKPOINTS','UNIQUE-DATA','PREVIEWS'),
    [switch]$ManifestOnly
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------- paths
function Resolve-Abs([string]$P) {
    if ([IO.Path]::IsPathRooted($P)) { return [IO.Path]::GetFullPath($P) }
    return [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $P))
}

$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $SourceRoot) { $SourceRoot = [IO.Path]::GetFullPath((Join-Path $scriptDir '..\..')) }
$SourceRoot = Resolve-Abs $SourceRoot
$TargetRoot = Resolve-Abs $TargetRoot
$rootPrefix = $SourceRoot.TrimEnd('\') + '\'

if (-not (Test-Path -LiteralPath $SourceRoot)) { throw "SourceRoot not found: $SourceRoot" }
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

try { Add-Type -AssemblyName System.IO.Compression } catch {}
try { Add-Type -AssemblyName System.IO.Compression.FileSystem } catch {}

# NB: no stderr redirection here -- under $ErrorActionPreference='Stop'
# a redirected native stderr stream raises and would silently lose HEAD.
$gitHead = $null
try {
    $headOut = @(& git -C $SourceRoot rev-parse HEAD)
    if ($LASTEXITCODE -eq 0 -and $headOut.Count -gt 0) { $gitHead = [string]$headOut[0] }
} catch {
    Write-Verbose "git rev-parse unavailable: $($_.Exception.Message)"
}

# ------------------------------------------------------------ group defs
$GroupDefs = [ordered]@{
    'SOURCE' = @{
        Zip   = 'VOICE-SOURCE'
        Kind  = 'GitArchive'
    }
    'STATE' = @{
        Zip      = 'VOICE-STATE'
        Kind     = 'Files'
        Includes = @(
            'docs/**',
            'experiments/**/metrics.jsonl',
            'experiments/**/conclusion.md',
            'EXPERIMENT-INDEX.md',
            'experiments/EXPERIMENT-INDEX.md',
            'artifacts/caedral-voice-v1/provenance/**'
        )
    }
    'CHECKPOINTS' = @{
        Zip      = 'VOICE-CHECKPOINTS'
        Kind     = 'Files'
        Includes = @(
            'experiments/r3/checkpoints/best.pt',
            'experiments/r3/checkpoints/last.pt',
            'experiments/r2/checkpoints/best.pt'
        )
    }
    'UNIQUE-DATA' = @{
        Zip      = 'VOICE-UNIQUE-DATA'
        Kind     = 'Files'
        Includes = @(
            'data/processed/teacher-kokoro/**',
            'data/manifests/**/*.jsonl',
            'data/manifests/**/*.json'
        )
    }
    'PREVIEWS' = @{
        Zip      = 'VOICE-PREVIEWS'
        Kind     = 'Files'
        Includes = @(
            'artifacts/caedral-voice-v1/voices/**',
            'artifacts/latest-preview/**'
        )
    }
}

# ----------------------------------------------------------- glob -> regex
# Supports: literal paths, trailing '/**' (subtree), '**/' (any depth),
# '*' (within one segment), '?' (one char). Returns $null for literals.
function Convert-GlobToRegex([string]$Glob) {
    $g = $Glob -replace '\\', '/'
    if ($g -notmatch '[*?]') { return $null }
    if ($g.EndsWith('/**')) {
        return '^' + [regex]::Escape($g.Substring(0, $g.Length - 3)) + '/.+$'
    }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('^')
    $i = 0
    while ($i -lt $g.Length) {
        $c = $g[$i]
        if ($c -eq '*') {
            if (($i + 1) -lt $g.Length -and $g[$i + 1] -eq '*') {
                if (($i + 2) -lt $g.Length -and $g[$i + 2] -eq '/') {
                    [void]$sb.Append('(?:[^/]+/)*'); $i += 3; continue
                }
                [void]$sb.Append('.*'); $i += 2; continue
            }
            [void]$sb.Append('[^/]*'); $i += 1; continue
        }
        if ($c -eq '?') { [void]$sb.Append('[^/]'); $i += 1; continue }
        [void]$sb.Append([regex]::Escape([string]$c)); $i += 1
    }
    [void]$sb.Append('$')
    return $sb.ToString()
}

function Test-ExcludedRel([string]$Rel) {
    if ($Rel -like '*.tmp') { return $true }
    foreach ($seg in $Rel.Split('/')) {
        if ($seg -eq '__pycache__') { return $true }
    }
    return $false
}

# ------------------------------------------------------------ file census
# Only walk the top-level trees referenced by the selected file groups
# (skips .git, data/cache and other unreferenced heavy subtrees).
$wantedKeys = $Groups | Where-Object { $GroupDefs[$_].Kind -eq 'Files' }
$scanTops   = New-Object System.Collections.Generic.HashSet[string]
$literals   = New-Object System.Collections.Generic.HashSet[string]
foreach ($k in $wantedKeys) {
    foreach ($inc in $GroupDefs[$k].Includes) {
        if ($inc -match '[*?]') {
            $top = ($inc -replace '\\', '/').Split('/')[0]
            if ($top -and $top -notmatch '[*?]') { [void]$scanTops.Add($top) }
        } else {
            [void]$literals.Add(($inc -replace '\\', '/'))
        }
    }
}

$byRel = @{}
foreach ($top in $scanTops) {
    $full = Join-Path $SourceRoot $top
    if (-not (Test-Path -LiteralPath $full)) { continue }
    Write-Host ("[scan] enumerating {0}\" -f $top)
    Get-ChildItem -LiteralPath $full -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($rootPrefix.Length) -replace '\\', '/'
        if (Test-ExcludedRel $rel) { return }
        $byRel[$rel] = $_
    }
}
foreach ($lit in $literals) {
    $full = Join-Path $SourceRoot ($lit -replace '/', '\')
    if ((Test-Path -LiteralPath $full) -and -not $byRel.ContainsKey($lit)) {
        $item = Get-Item -LiteralPath $full
        if (-not (Test-ExcludedRel $lit)) { $byRel[$lit] = $item }
    }
}
Write-Host ("[scan] candidate files after exclusions: {0}" -f $byRel.Count)

# lazy sha256 cache shared across groups
$shaCache = @{}
$script:skippedLocked = New-Object System.Collections.Generic.List[string]

# Reader tolerant of files the live trainer is writing (metrics.jsonl,
# checkpoints): request ReadWrite|Delete sharing; retry briefly; if the
# holder keeps an exclusive lock, give up WITHOUT disturbing it.
function Open-SharedReader([string]$FullPath) {
    $share = [System.IO.FileShare]([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
    try {
        return [IO.File]::Open($FullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    } catch {
        foreach ($ms in 250, 500, 1000, 2000) {
            Start-Sleep -Milliseconds $ms
            try {
                return [IO.File]::Open($FullPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
            } catch {}
        }
        throw
    }
}

function Get-Sha256([string]$FullPath) {
    if (-not $shaCache.ContainsKey($FullPath)) {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $stream = Open-SharedReader $FullPath
            try {
                $bytes = $sha.ComputeHash($stream)
                $shaCache[$FullPath] = (($bytes | ForEach-Object { $_.ToString('x2') }) -join '')
            } finally { $stream.Dispose() }
        } finally { $sha.Dispose() }
    }
    return $shaCache[$FullPath]
}

function Write-Utf8NoBom([string]$Path, [string[]]$Lines) {
    [IO.File]::WriteAllLines($Path, $Lines, (New-Object System.Text.UTF8Encoding($false)))
}

# ------------------------------------------------------------- build loop
$results = @()
foreach ($key in $Groups) {
    $def       = $GroupDefs[$key]
    $groupName = $def.Zip
    $zipName   = "$groupName.zip"
    $zipPath   = Join-Path $TargetRoot $zipName
    Write-Host ("[group] {0}" -f $groupName)
    $members = @()      # objects: Rel, Full, Len, Sha (optional)
    $tracked = @()      # SOURCE only: git ls-tree rows

    if ($def.Kind -eq 'GitArchive') {
        $tracked = @(& git -C $SourceRoot ls-tree -r HEAD | ForEach-Object { $_ })
        if ($LASTEXITCODE -ne 0) { throw "git ls-tree failed for $SourceRoot" }
        if (-not $ManifestOnly) {
            Write-Host ("[git] git archive HEAD -> {0}" -f $zipName)
            & git -C $SourceRoot archive --format=zip -o $zipPath HEAD
            if ($LASTEXITCODE -ne 0) { throw "git archive failed ($LASTEXITCODE)" }
        }
    }
    else {
        foreach ($inc in $def.Includes) {
            $rx = Convert-GlobToRegex $inc
            if ($rx) {
                foreach ($kv in $byRel.GetEnumerator()) {
                    if ($kv.Key -match $rx) {
                        $members += [pscustomobject]@{ Rel = $kv.Key; Item = $kv.Value }
                    }
                }
            } else {
                if ($byRel.ContainsKey($inc)) {
                    $members += [pscustomobject]@{ Rel = $inc; Item = $byRel[$inc] }
                } else {
                    Write-Verbose ("include not present on disk: {0}" -f $inc)
                }
            }
        }
        $members = @($members | Sort-Object Rel)
        # resolve all member hashes up front; anything exclusively locked
        # by the live trainer is SKIPPED (recorded), never allowed to abort
        $skipBefore = $script:skippedLocked.Count
        $resolved = New-Object System.Collections.Generic.List[object]
        foreach ($m in $members) {
            try {
                $h = Get-Sha256 $m.Item.FullName
                $resolved.Add([pscustomobject]@{ Rel = $m.Rel; Item = $m.Item; Sha = $h })
            } catch {
                Write-Host ("  [warn] locked by another process, SKIPPED: {0}" -f $m.Rel) -ForegroundColor Yellow
                $script:skippedLocked.Add("$groupName/$($m.Rel)")
            }
        }
        $members = $resolved
        $groupSkipped = @()
        if ($script:skippedLocked.Count -gt $skipBefore) {
            $groupSkipped = @($script:skippedLocked.ToArray())[$skipBefore..($script:skippedLocked.Count - 1)]
            Add-Content -Path (Join-Path $TargetRoot 'SKIPPED-LOCKED.txt') -Value $groupSkipped
        }
        if (-not $ManifestOnly) {
            if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
            $fs = [IO.File]::Open($zipPath, [IO.FileMode]::Create, [IO.FileAccess]::Write)
            try {
                $za = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
                try {
                    foreach ($m in $members) {
                        # source opened before entry creation: a locked file
                        # cannot leave an empty ghost entry (already skipped
                        # during hash resolution, but stay defensive)
                        try { $src = Open-SharedReader $m.Item.FullName } catch {
                            Write-Host ("  [warn] locked at copy time, skipping: {0}" -f $m.Rel) -ForegroundColor Yellow
                            continue
                        }
                        try {
                            $entry = $za.CreateEntry($m.Rel, [System.IO.Compression.CompressionLevel]::Optimal)
                            $es = $entry.Open()
                            try { $src.CopyTo($es) } finally { $es.Dispose() }
                        } finally { $src.Dispose() }
                    }
                } finally { $za.Dispose() }
            } finally { $fs.Dispose() }
        }
    }

    # sidecar + zip hash (only when a real zip exists)
    $zipSha  = $null
    $zipSize = 0
    if (-not $ManifestOnly) {
        $zipSha  = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLower()
        $zipSize = (Get-Item -LiteralPath $zipPath).Length
        $lines = New-Object System.Collections.Generic.List[string]
        $lines.Add(('{0}  {1}' -f $zipSha, $zipName))
        if ($def.Kind -eq 'Files') {
            foreach ($m in $members) {
                $lines.Add(('{0}  {1}' -f $m.Sha, $m.Rel))
            }
        }
        Write-Utf8NoBom (Join-Path $TargetRoot "$groupName.SHA256SUMS.txt") $lines.ToArray()
    }

    # manifest json (always: this is the inspectable manifest-generation part)
    $totalBytes = 0
    $fileEntries = @()
    if ($def.Kind -eq 'Files') {
        foreach ($m in $members) {
            $totalBytes += $m.Item.Length
            $fileEntries += [ordered]@{
                path   = $m.Rel
                bytes  = $m.Item.Length
                sha256 = $m.Sha
            }
        }
    } else {
        foreach ($row in $tracked) {
            $meta = $null
            if ($row -match '^(\S+)\s+(\S+)\s+(\S+)\t(.*)$') {
                $meta = [ordered]@{ mode = $Matches[1]; type = $Matches[2]; git_blob = $Matches[3]; path = $Matches[4] }
            } else {
                $meta = [ordered]@{ path = $row }
            }
            $fileEntries += $meta
        }
    }
    $manifest = [ordered]@{
        group         = $groupName
        kind          = $def.Kind
        generated_utc = (Get-Date).ToUniversalTime().ToString('o')
        source_root   = $SourceRoot
        git_commit    = $gitHead
        manifest_only = [bool]$ManifestOnly
        zip           = $zipName
        zip_bytes     = $zipSize
        zip_sha256    = $zipSha
        file_count    = $fileEntries.Count
        total_bytes   = $totalBytes
        skipped_locked= @($script:skippedLocked)
        files         = $fileEntries
    }
    $json = (ConvertTo-Json -InputObject $manifest -Depth 4)
    [IO.File]::WriteAllText((Join-Path $TargetRoot "$groupName.manifest.json"), $json,
        (New-Object System.Text.UTF8Encoding($false)))

    $results += [pscustomobject]@{
        Group     = $groupName
        Files     = $fileEntries.Count
        SrcMB     = [math]::Round($totalBytes / 1MB, 2)
        ZipMB     = [math]::Round($zipSize / 1MB, 2)
        ZipSha256 = $(if ($zipSha) { $zipSha.Substring(0, 12) } else { 'MANIFEST-ONLY' })
    }
}

$results | Format-Table -AutoSize
Write-Host ("done. target: {0}" -f $TargetRoot)
