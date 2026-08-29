# CAEDRAL VOICE V1 — REAL VPS 4 GB / 4 vCPU TEST (canonical ~10 s audio)

Host: **144.91.116.147** (`mail.caedral.com`) · Date: 2026-08-25
Engine: FROZEN V1 (`caedral-voice:v1`, unchanged hashes). Only cgroup caps
changed. Canonical input FROZEN: `canonical-10s-input.txt` (179 chars,
caedral-f1, speed 1.0 → deterministic **10.07 s** audio every run;
SHA-256 `7a7c4af723372920bb8556f75ad3563a02aad64c874b7e21edeac2ee18125adb`).
Limits verified inside the container every phase:
`memory.max=3999997952 (~4 GB)` · `memory.swap.max=0` · `cpu.max=400000 100000 (4.0 vCPU)`.

Pre-change safety snapshot: load 1.37/1.63/1.83 on 6 cores, 6204 MB RAM
available, steal 0 %, Stalwart healthy (0 restarts), only pre-existing
discord-bot unhealthy. Post-everything sweep identical (§ Safety below).

## 1. BASELINE WARM MEMORY

| State | Value |
|---|---|
| Container idle before engine ready | cgroup current ≈ 300–450 MB (imports/model mapping) |
| Engine ready / warm | `/metrics rss_mb` 560 MB |
| Warm idle before benchmark | cgroup `memory.current` **600.8 MB** |
| CPU idle | quota unused (throttle counters flat) |

## 2. SINGLE-REQUEST BENCHMARK — 20 sequential × 10.07 s audio @ 4 GB / 4 vCPU

20/20 HTTP 200, valid RIFF/WAVE 24 kHz mono PCM16, zero OOM, zero failures.

| Metric | generation wall | RTF | realtime multiplier |
|---|---|---|---|
| best | **5.47 s** | 0.55 | ×1.84 |
| **p50** | **6.32 s** | **0.63** | **×1.59** |
| p95 | 7.25 s | 0.72 | ×1.39 |
| worst | 7.56 s | 0.75 | ×1.33 |

### FIRST IMPORTANT RESULT

**HOW LONG DOES IT TAKE TO GENERATE 10 SECONDS OF AUDIO?**
Measured (not estimated): **p50 = 6.3 s · p95 = 7.3 s · best 5.5 s · worst 7.6 s**
→ 10.07 s audio renders **faster than real-time** (≈ 1.6× realtime at p50).

## 3. MEMORY CONSUMPTION (4 GB / 4 vCPU)

| Measure | Value |
|---|---|
| Warm RAM (idle) | 561–691 MB (RSS) / ~601 MB cgroup current |
| Single 10 s request peak | cgroup peak 1198 MB (deterministic across run window) |
| 20-run maximum | **1198.5 MB** |
| Post-request settled | ~600–796 MB (working set returns to warm band) |
| % of 4 GB limit used at peak | **~30 %** |
| Memory leak across repeated requests | **NO** — flat working set over 20 recorded + 266 further requests (final settled RSS 690 MB after 286 lifetime completions) |

## 4. CPU CONSUMPTION (honest utilization)

Sampled every 10 s from inside-container cgroup during the 20-run window:

| Measure | Value |
|---|---|
| Average synthesis CPU | ≈ 2.5 cores |
| Peak instantaneous | 3.69 cores (of 4.0 allowed) |
| Effective core utilization | engine uses **~2.5–3.7 of 4 vCPU** — it does NOT fully saturate 4 cores (frozen intra_op=6 threads on shared EPYC, memory-bandwidth bound) |
| cgroup throttling | 57 % of periods throttled (median), 78 % max |
| Host CPU pressure (PSI some avg10) | up to 23.6 % during full-cap ladder; no service degradation |

Frozen runtime settings were NOT modified to force higher utilization.

## 5. THEORETICAL SERIAL RPM (10 s audio)

`60 / gen_p50(6.29 s)` = **9.55 RPM** (theory only — see measured edge below).
Equivalent ≈ 573 requests/hour if perfectly serialized.

## 6. MEASURED SUSTAINABLE RPM — LADDER (300 s levels, same canonical input)

| Level | Target RPM | Submitted→200 | Achieved RPM | E2E p50 / max | Queue depth |
|---|---|---|---|---|---|
| 25 % | 2.39 | 12/12 | 2.40 | 6.4 / 7.6 s | 0 |
| 50 % | 4.77 | 24/24 | 4.79 | 5.3 / 6.9 s | 0 |
| 75 % | 7.16 | 36/36 | 7.15 | 5.6 / 6.9 s | 0 |
| **100 %** | **9.55** | **46/46** | **9.10** | **5.7 / 13.8 s** | **0** |
| 125 % | 11.93 | 49/49 | pinned **9.60** | 5.7 / 14.8 s | 0–1 |

Every accepted request: HTTP 200 + valid RIFF. Zero 429/5xx/timeouts/OOM.

**MEASURED SUSTAINABLE EDGE: ≈ 9.6 RPM** (throughput pins there when arrival
rate exceeds it; latency stays bounded because the backlog stays shallow).
**RECOMMENDED PRODUCTION RPM (≈27 % headroom): ≤ 7 RPM** for 10-second-class
requests ⇒ ≈ 420 requests/hour sustainable per instance.

## 7. CONCURRENCY TEST (one shared ORT session)

| Concurrency | Throughput observed | Latency behavior | Errors |
|---|---|---|---|
| 1 (sequential) | 9.5 RPM serial-equivalent | gen p50 6.3 s | none |
| 2 | burst makespan 10.6 s (= 2 × single cycle) | each request waits full serialization | none |
| 4 | burst makespan 23.0 s (serial would be 25.2 s — ≤8 % incidental overlap) | pairs serialize; no real speedup | none |

Extra concurrency does NOT increase throughput (single synth worker by
design). **Best/recommended concurrency: 1** — bursts are absorbed safely but
latency grows linearly.

## 8. SHORT-BURST TEST

10 simultaneous client requests (short text): **all 10 served HTTP 200 with
valid WAV in a 16.8 s makespan**; healthz/readyz PASS after; normal request
succeeds after; zero OOM/crash. As documented in the V1 server report, the
frozen endpoint admits requests serially, so its internal 429 path is not
exercised via HTTP — bursts are absorbed by bounded serialized waiting rather
than rejected. Gateway-side limits remain mandatory.

## 9. CPU SCALING COMPARISON (same 10 s input, RAM fixed at 4 GB, ≥10 warm runs each)

| Profile | gen p50 | gen p95 | RTF p50 | Peak MB |
|---|---|---|---|---|
| 4 GB / 1 vCPU | 28.73 s | 30.95 s | 2.85 | 919 |
| 4 GB / 2 vCPU | 10.60 s | 16.60 s | 1.05 | 1015 |
| **4 GB / 4 vCPU** | **6.32 s** | **7.25 s** | **0.63** | 1198 |

Speedups: **2 vs 1 vCPU = 2.71× · 4 vs 2 = 1.68× · 4 vs 1 = 4.55×**

## 10. EXISTING SERVICES SAFETY DURING HEAVIEST TEST

Host load reached 5.6–5.8 of 6 cores while Voice ran at its full 4-vCPU cap;
memory available stayed ≥ 6.1 GB; steal 0 %. Stalwart/mail: healthy, 0
restarts, throughout. embedding-api / api-gateway / site: HTTP 200 checked
mid-ladder and post-test. Only unhealthy container remains the PRE-EXISTING
discord-bot fault (baseline before any Voice work). **No service starved;
the 4 GB / 4 vCPU cap is SAFE on this VPS.**

## 11. FINAL DEPLOYED ENVELOPE

Left configured exactly as authorized: **memory.max ≈ 4 GB (3999997952),
swap-free, cpu.max = 4.0 vCPU**. Idle consumption stays ~0.6 GB / 0 vCPU —
the cap is a ceiling, not a reservation.

Gateway capacity guidance: provision Voice at **≤ 7 RPM sustained for
10-second-class requests, concurrency 1**, abandonment timeout ≥ 30 s
(p95 7.3 s) — keep the previous 240 s abandonment only for long-form inputs.

V1 remained FROZEN for every measurement. No ASR/DNSMOS/Whisper tooling was
run against the production runtime.
