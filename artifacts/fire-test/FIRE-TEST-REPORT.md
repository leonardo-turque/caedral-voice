# CAEDRAL VOICE V1 — FIRE TEST

Date: 2026-08-24 · Isolation: Docker linux/amd64, hard cgroup v2 limits,
values read from `/sys/fs/cgroup` INSIDE each container (never from CLI
flags). No swap: `memory.swap.max=0`, `--memory-swap == --memory`.
Engine: FROZEN V1 (ORT CPU, ENABLE_ALL, intra_op=6, cpu_arena=false) in every
profile; the CPU QUOTA enforces 1.0 / 0.5 vCPU — thread count was not touched.
Canonical request: `canonical-input.txt` (sha256 `9fa32ab5…`), caedral-f1,
speed 1.0 → **29.82 s** audio (accepted window 29.0–31.0 s), identical text
for both profiles. Production engine only — no ASR/DNSMOS/acceptance tooling.

---

## PROFILE A — 1 GB / 1 vCPU

| Field | Result |
|---|---|
| Model load | **PASS** (ctor 5.39 s, warmup 6.52 s, RSS after warmup 511 MB, HWM peak 623 MB) |
| Cold load | ~12.7 s script→warm inside container (imports 1.18 s + ctor 5.39 s + warmup 6.52 s) |
| 30-s output actual duration | n/a — generation did not survive |
| Generation p50 / p95 | NOT POSSIBLE — OOM kill during FIRST generation |
| RTF p50 / p95 | — |
| Realtime multiplier | — |
| Warm RAM | 511 MB (load+warmup fit comfortably) |
| Peak RAM | kernel OOM-killed the process mid-generation |
| CPU throttling | n/a (died before steady state) |
| Theoretical serial RPM | — |
| Measured max sustainable RPM | **0** |
| Recommended production RPM | **0** |
| Recommended concurrency | n/a |
| Failure/overload point | first ~30 s generation; `OOMKilled=true`, exit 137, zero swap spill |
| Production viable | **NO** |

**Limiting resource: MEMORY during inference — not during load.**
Host-side decomposition explains it exactly: engine-only long-form render
peaks at **1136 MB** (`artifacts/benchmarks/v1-rtf-ram-recheck.json`); the
in-container Linux measurement of the same canonical generation peaks at
**1023 MB VmHWM** (supplementary run below) — i.e. Profile A misses by only
~2.5 %. The frozen engine cannot complete a canonical 30-s generation inside
1,000,000,000 bytes, and per directive nothing was retuned to force a pass.

## PROFILE B — 500 MB / 0.5 vCPU

| Field | Result |
|---|---|
| Model load | **FAIL — OOM kill DURING model load** (`OOMKilled=true`, exit 137) |
| Cold load | never completed (310.5 MB FP32 graph + Python/runtime > 500 MB) |
| 30-s output actual duration | NOT POSSIBLE |
| Generation p50 / p95 | — |
| RTF p50 / p95 | — |
| Realtime multiplier | — |
| Warm RAM | never reached |
| Peak RAM | killed below limit during session construction |
| CPU throttling | visible already at startup (4 throttled periods) but irrelevant — memory bound |
| Theoretical serial RPM | — |
| Measured max sustainable RPM | **0** |
| Recommended production RPM | **0** |
| Recommended concurrency | n/a |
| Failure/overload point | model load itself |
| Production viable | **NO** |

No estimates were invented for this profile.

---

## SUPPLEMENTARY BOUNDARY MEASUREMENT — 1.4 GB / 1 vCPU

*(outside the two specified profiles; recorded because the founder asked for
reality, and reality here is that the memory boundary sits between 1.0 GB and
1.4 GB)*

cgroup verified inside: `memory.max=1399996416`, `memory.swap.max=0`,
`cpu.max=100000 100000`.

- Cold load PASS: ctor 3.95 s, warmup 4.37 s, RSS after warmup 519 MB, HWM 622 MB.
- **10/10 sequential canonical generations succeeded; zero OOM/high events;
  RSS returns to ~547 MB after each request.**
- Generation wall: **p50 66.09 s · p95 92.88 s · min 55.39 s · max 92.88 s**
- RTF: **p50 2.216 · p95 3.114** → realtime multiplier ≈ **0.45× (p50)**,
  0.32× (p95). Every request is fully quota-bound (99.7 % of periods
  throttled; cpu-time ≈ wall).
- Peak RSS: deterministic **1023 MB** on every generation (fused-graph batch
  workspace), warm idle 519–547 MB.

### Capacity ladder (service container, same limits)

`load-summary-aplus-1.4gb-1vcpu.json` + `load-test.csv`. Theoretical serial
capacity = 60 / gen_p50 = **0.908 RPM**.

| Level | Target RPM | Submitted→200 | Achieved RPM | E2E p50 / max | Queue |
|---|---|---|---|---|---|
| 0.25× | 0.227 | 2/2 | — | 64.6 / 76.9 s | 0 |
| 0.50× | 0.454 | 3/3 | 0.574 | 54.4 / 59.3 s | 0 |
| 0.75× | 0.681 | 4/4 | 0.756 | 52.7 / 53.9 s | 0 |
| **1.00×** | **0.908** | **5/5** | **0.928** | **54.8 / 57.8 s** | **0** |
| 1.25× | 1.135 | 6/6 | 0.912 (pinned) | 173 / 233 s | grows |

Every accepted request returned HTTP 200 with valid RIFF audio at every
level; no 429 was ever needed below queue saturation; zero OOM/crash.

- **MAX SUSTAINABLE (measured edge): ~0.91 RPM** — queue-stable through the
  100 % level; beyond it throughput pins at the serial limit and latency
  grows linearly (demonstrated at the 125 % level).
- **ABSOLUTE OVERLOAD POINT: any sustained arrival rate > ~0.91 RPM**
  for canonical 30-s requests.
- **SAFE RECOMMENDED PRODUCTION RPM: 0.65 RPM** (~28 % headroom below the
  edge) for 30-s requests; higher rates are fine for shorter text.
- **Concurrency:** bursts of 2 and 4 all completed cleanly but serialized
  perfectly (makespans 114 s ≈ 2×57 s; 237 s ≈ 4×57 s) — extra concurrency
  adds only queue wait. **Recommended concurrency = 1.**

### Packaging validation (production image, unconstrained)

Fresh `caedral-voice:v1` container, no limits (`packaging-validation.log`):
cold load 511→514 MB (HWM 607), one canonical request **PASS**, wall 22.5 s
(RTF 0.756, realtime ×1.32 on idle host), **peak 1029 MB**, post-request
515 MB, zero throttle/OOM events. Confirms the packaged image behaves exactly
like the fire-tested runtime.

## Conclusions for deployment

1. Frozen V1 needs **~1023–1029 MB peak** for one canonical 30-s request:
   minimum viable single-instance envelope ≈ **1.25–1.4 GB RAM**.
2. At exactly **1.0 vCPU**, one 30-s request takes **55–93 s** (RTF 1.9–3.1);
   capacity is strictly serial (**0.91 RPM** measured sustainable).
3. Both specified profiles FAILED honestly under frozen-engine constraints;
   no engine, model, TN or quality file was modified to change any outcome.
4. Shorter inputs scale roughly with phoneme count; sub-second control
   sentences stay well under both time and memory peaks (host grid:
   RTF 0.45–0.86 on idle full machine).

V1 ENGINE STATUS: FROZEN · FIRE TESTS: EXECUTED AND PERSISTED · ARCH: DO NOT START
