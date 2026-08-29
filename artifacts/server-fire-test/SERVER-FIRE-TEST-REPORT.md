# CAEDRAL VOICE V1 — REAL SERVER FIRE TEST REPORT

Host: **144.91.116.147** (`mail.caedral.com`) · Date: 2026-08-25
Deployment: `caedral-voice:v1` at `/opt/caedral/services/voice-api/releases/v1`,
published **127.0.0.1:8600 only**, internal secret configured (never printed).
Canonical request: frozen `canonical-input.txt` (548 chars, caedral-f1,
speed 1.0 → 29.82 s audio) — the SAME input as the local fire test.
Limits verified INSIDE the container from `/sys/fs/cgroup` every profile.
No engine/model/TN file was modified; V1 remained FROZEN throughout.

---

## HOST / ENVIRONMENT

| Field | Value |
|---|---|
| OS / kernel | Ubuntu 26.04 LTS / 7.0.0-22-generic, up 80 days |
| CPU | AMD EPYC, 6 logical cores (1 thread/core) |
| RAM | 11,955 MB total · ~6.5 GB available pre-deploy |
| Swap | 4 GB (3.08 GB already used before testing — Voice runs swap-free) |
| Disk | 96 G root, 73 % used pre-deploy (~27 GB free) |
| Docker | 29.5.2 + Compose v5.1.4 |
| Existing load BEFORE | load avg 2.09/2.41/2.08; PSI cpu some 5.35 %, mem 0 %; 29 containers running (Caedral platform incl. api-gateway, model-inference, embedding-api, site; Stalwart mail on 25/143/465/587/993/995; zennitex, whatsapp-api, clicars, n8n, DBs) |
| Pre-existing faults | `caedral-discord-bot` container unhealthy BEFORE any Voice change (unchanged through all tests) |
| CPU steal during tests | 0 % (vmstat + /proc/stat sampled every 10 s) |

Existing services safety checks were run mid-test and after every heavy
phase: no degradation of any other container, mail healthy with zero
restarts, embedding-api and api-gateway HTTP 200 throughout.

## SEQUENTIAL FIRE TESTS — canonical ~30 s input × 10 each

All three profiles: **10/10 PASS**, valid RIFF/WAVE 24 kHz mono PCM16 every
time, deterministic per-request audio duration 29.82 s, zero OOM events.

| Metric | P1: 1400 MB / 1 vCPU | P2: 2048 MB / 1 vCPU | P3: 2048 MB / 2 vCPU |
|---|---|---|---|
| Wall min / p50 / p95 / max | 81.3 / **86.9** / 139.1* / 139.1 s | 77.5 / **88.4** / 128.9 / 128.9 s | 33.1 / **37.3** / 58.6 / 58.6 s |
| RTF p50 / p95 | 2.91 / 4.67 | 2.96 / 4.32 | 1.25 / 1.96 |
| Peak RAM | 1160.9 MB | **1411.9 MB** | 1085.5 MB |
| CPU periods throttled | 87 % | 96 % | 91 % |
| Theoretical serial RPM | 0.691 | 0.679 | 1.611 |
| OOM / oom_kill | 0 / 0 | 0 / 0 | 0 / 0 |

\* P1 p95/max is the one-time cold-path first request (139 s); steady-state
max was 105.7 s.

### Comparison vs previous LOCAL constrained fire test (same input)

| Metric | Local 1400 MB / 1 vCPU | THIS VPS 1400 MB / 1 vCPU |
|---|---|---|
| p50 | 66.1 s | **86.9 s (+31 %)** |
| p95 | 92.9 s | 139.1 s (cold) / 105.7 steady (+14–50 %) |
| RTF p50 | 2.22 | 2.91 |
| Peak RSS | 1023 MB | 1161 MB |
| Sustainable edge | 0.91 RPM | 0.72 RPM |

The shared EPYC VPS is materially slower per-core than the local benchmark
host — the reference numbers must NOT be used for capacity planning here.

### Jitter / contention (P1 sequential)

min 81.3 · median 86.9 · steady max 105.7 · cold outlier 139.1 s;
steady-state coefficient of variation ≈ **8.7 %**; host load 2.1–2.9 of 6
cores; steal 0 %. Verdict: the VPS is reasonably consistent, with occasional
+20 % excursions and a one-time cold-path penalty.

## RPM LADDER + SUSTAINED WINDOWS (300 s per level)

### PROFILE — 2048 MB / 2 vCPU

| Level | Target RPM | Submitted→200 | Achieved RPM | E2E p50 / max | Queue |
|---|---|---|---|---|---|
| 0.25× | 0.403 | 3/3 | 0.54 | 36.7 / 36.8 s | 0 |
| 0.50× | 0.806 | 5/5 | 0.89 | 39.0 / 43.3 s | 0 |
| 0.75× | 1.209 | 7/7 | 1.29 | 33.6 / 39.0 s | 0 |
| **1.00×** | **1.612** | **9/9** | **1.62** | **30.5 / 36.6 s** | **0** |
| 1.25× | 2.015 | 10/10 | pinned 1.81 | 65.2 / 88.6 s | grows via HTTP backlog |

Every accepted request returned HTTP 200 + valid RIFF at every level.

- **Measured sustainable edge: ~1.8 RPM** (throughput still rising through
  the 100 % level and pinning at 125 %).
- **Recommended safe RPM (≈30 % headroom): ≤ 1.25 RPM** for canonical 30 s
  requests.
- Verdict: PASS.

### PROFILE — 1400 MB / 1 vCPU

| Level | Target RPM | Submitted→200 | Achieved RPM | E2E p50 / max | Queue |
|---|---|---|---|---|---|
| 0.25× | 0.173 | 1/1 | 0.20 | 130.4 s (cold) | 0 |
| 0.50× | 0.346 | 2/2 | 0.40 | 80.0 / 84.0 s | 0 |
| 0.75× | 0.519 | 3/3 | 0.57 | 81.2 / 82.2 s | 0 |
| **1.00×** | **0.691** | **4/4** | **0.72** | **78.5 / 78.6 s** | **0** |
| 1.25× | 0.864 | 5/5 | pinned 0.67 | 169.9 / 239.7 s | grows |

- **Measured sustainable edge: ~0.72 RPM.**
- **Recommended safe RPM (≈30 % headroom): ≤ 0.48 RPM** for canonical 30 s
  requests.
- Verdict: PASS (with the latency cost noted above).

## CONCURRENCY FIRE TEST

Bursts against a single shared ORT session (frozen design):

| Profile | Burst 2 makespan | Burst 4 makespan | Speedup vs serial |
|---|---|---|---|
| 2 vCPU | 60.6 s (= 2 × single) | 126.4 s (= 4 × single) | **none** |
| 1 vCPU | 197.0 s (= 2 × single) | 363.1 s (= 4 × single) | **none** |

Concurrency adds ONLY queue wait — throughput never improves.
**Recommended concurrency: 1.**

## OVERLOAD SURVIVAL

- Burst of 12 concurrent short requests: all 12 served HTTP 200 with valid
  WAV; health/ready PASS after; normal request succeeds after.
- Deterministic overflow probe (20 concurrent canonical requests):
  revealed the frozen endpoint admits requests **serially** (`fut.result()`
  blocks the event loop), so excess connections wait in uvicorn's connection
  backlog rather than filling the internal queue — `/metrics queue_depth`
  stayed 0 through EVERY phase including 125 % ladder levels. The in-code
  `OVERLOADED` 429 path (with `Retry-After: 5`) exists but is effectively
  unreachable via HTTP. **Consequence: public rate limiting and abandonment
  timeouts MUST live in the gateway** (as designed). No OOM, no crash, no
  corrupted WAV under any burst; service stayed healthy throughout.

## RESTART / PERSISTENCE

`docker compose restart voice-api` + config-change recreate: model reload +
warmup unattended, `readyz` 200, configuration persisted (compose file),
post-restart synthesis PASS (valid 24 kHz mono PCM16 output). PASS.

## LOGGING AUDIT

Logs contain request status, voice id, char count, wall/audio/RTF and HTTP
codes only. Verified: **zero occurrences of the internal secret** in logs;
no customer text logged; no keys. PASS.

## RESOURCE-SAFETY FINDING (deployment envelope)

Under the 2048 MB limit a fresh-container canonical generation peaked at
**1411.9 MB** (vs 1160.9 MB observed at the 1400 MB limit across a longer-
running process). A 1400 MB cap therefore left as little as ~6 % headroom in
the worst observed case. Production floor raised accordingly:

**DEPLOYED PRODUCTION ENVELOPE: 2048 MB hard limit (swap-free), 1.0 vCPU,
concurrency 1, safe rate ≤ 0.48 RPM for 30 s requests.**
Validated upgrade path: 2.0 vCPU → p50 37 s, sustainable edge ~1.8 RPM,
safe rate ≤ 1.25 RPM.

## FINAL SECURITY CHECK

| Check | Result |
|---|---|
| Raw Voice service publicly exposed | **NO** (127.0.0.1 publish; ufw default-DROP; port closed externally) |
| Internal secret configured | **YES** (root-owned .env chmod 600; value never printed/committed/logged) |
| Secret leaked in logs | **NO** (grep count 0) |
| SSH key leaked | **NO** (key generated locally, private half never transmitted; founder password used once to install it, never stored) |
| Unrelated service modified | **NO** (only new isolated compose project `caedral-voice-api`; discord-bot's pre-existing unhealthy state unchanged) |
| Firewall changed | **NO** (ufw untouched; nothing published on 0.0.0.0) |
| Production mail impacted | **NO** (Stalwart healthy, 0 restarts, all tests) |

## VERDICTS

- REAL VPS VOICE: **READY**
- MAIN PLATFORM INTEGRATION: **READY** (gateway routes to
  `http://127.0.0.1:8600` on this host, forwarding `X-Internal-Secret`;
  enforce gateway-side public rate limits + client abandonment ~240 s)
- SAFE PRODUCTION ENVELOPE: **2 GB RAM / 1 vCPU / concurrency 1 /
  ≤ 0.48 RPM canonical (≤ 1.25 RPM if upgraded to 2 vCPU)**

V1 remains FROZEN. No V2, no Arch migration was started.
