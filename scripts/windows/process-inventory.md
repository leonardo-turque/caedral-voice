# Windows Voice-process inventory (BACKUP-SHUTDOWN-01)

Audit: 2026-08-23, live read-only inspection via `Get-CimInstance Win32_Process`
on this machine (repo `C:\Users\leona\Downloads\caedral-voice`, git HEAD
`fe30bca584aa4d594b14b1c1875b236fdafe33d2`, clean tree). This file is the
source of truth for the `$AllowList` in `prepare-arch-shutdown.ps1`. Re-audit
before every shutdown run: PIDs die and get reused.

## Live training tree (CV-G1-E0013 regressor, run "r3")

| Allowlist key | PID | Process | Parent | Purpose |
|---|---|---|---|---|
| `R3_WORKER` | **21728** | python.exe (.venv) | 11616 | Main r3 training worker |
| `R3_PARENT_PYTHON` | 11616 | python.exe (.venv) | 19968 | Parent python, same cmdline; spawner/owner of 21728 |
| `R3_CMD_WRAPPER` | 19968 | cmd.exe | 23240 | `cmd /c .venv\Scripts\python.exe ... > logs\r3.out.log 2> logs\r3.err.log` |

Command line (identical for both pythons, wrapped by the cmd):

```
.venv\Scripts\python.exe -X utf8 experiments\CV-G1-E0013-regressor\train_regressor.py
  --manifests librittsr-clean100,teacher-kokoro
  --durations data/manifests/all-durations.jsonl
  --mel-cache-dir data/cache/mel-v1
  --steps 20000 --threads 6 --exp-id r3
```

- Started: 2026-08-23 ~08:25 local (third restart of E0013; earlier runs r1/r1b/r1c/r2 exited).
- Logs: `experiments/CV-G1-E0013-regressor/logs/r3.out.log` (active, melcache progress lines)
  and `logs/r3.err.log` (0 bytes). Historical run pids in `r1.pid`, `r1c.pid`.
- Checkpoint dependency: writes `experiments/r3/checkpoints/last.pt` and `best.pt`
  every `--ckpt-every` steps (both last seen 185,324,952 bytes at 2026-08-23T15:18Z).
  The trainer has NO signal/atexit handler: killing it between saves loses progress
  since the last save; on relaunch (no `--fresh`) it resumes bitwise from `last.pt`
  (E0003 guarantee). "Checkpoint on exit" = MANUAL: let it pass one more periodic
  save, optionally copy `last.pt` aside, then stop.
- Baseline sha256 at audit time:
  - `experiments/r3/checkpoints/last.pt` = `cb0af758437f69a72e7c043bb94836a8a0ef88fa199d09e4aec10c2c581023b4`
  - `experiments/r3/checkpoints/best.pt` = `c1c8478b0e649a321dfed10c6d5acb0b26b98dbc521077fd46342694bc192bf1`
    (identical to last.pt at audit moment — best==last checkpoint written together)
  - `experiments/r2/checkpoints/best.pt` = `c55ac1e9090742720d19d8dce93bc7af1d4af737db87b28500589bfbcc4eddee`

Stopping order used by the shutdown script: 21728 (child) -> 11616 (parent) -> 19968 (wrapper).

## Claude-session monitors (NOT OS processes to allowlist)

- Watchdog v5 / coordinator Monitors live INSIDE Claude Code sessions
  (`LAUNCH_E0013.ps1` step [3/4]: "watchdog launched separately by coordinator Monitor").
  They die when their session ends. The shutdown script cannot see or kill them;
  close all Claude sessions running monitors BEFORE treating the box as quiet.
- No standalone watchdog .py/.ps1 process exists (process scan found none).

## Teacher shards (CV-G1-E0012-teacher-gen)

Both DONE, no live generation:
- shard0 pid 24532: kept 5964, 8.9705 audio-h, updated 2026-08-23T12:48:21Z — PID verified gone.
- shard1 pid 13032: kept 6169, 8.9029 audio-h, updated 2026-08-23T12:48:12Z — PID verified gone.
- Merged meta: `data/manifests/teacher-kokoro.meta.json`, rows_in_manifest 15498,
  kept 12133, 17.8733 audio-h (matches 15498 wavs on disk in `data/processed/teacher-kokoro`).

## Other Voice processes

- kokoro-onnx / vocos inference: NONE running (process scan 2026-08-23).
- aligner (E0011, old pid 41464): long gone. No other python/pwsh Voice processes
  besides the r3 tree above.

## Environment notes

- `nvidia-smi` is NOT on PATH (Windows 11 Home): GPU occupancy cannot be confirmed;
  treat this process table as the source of truth.
- Only C: exists; backup target default `.\backup-staging` (D:\ placeholder passed via `-TargetRoot`).
