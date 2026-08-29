# CAEDRAL VOICE V1 — OPERATIONS RUNBOOK (server 144.91.116.147)

Internal model service. The gateway owns customer auth/billing/rate limits;
this service is internal-only (published on `127.0.0.1` only).

## Access

```bash
ssh root@144.91.116.147        # key auth (deploy key installed on this machine)
```

## Layout

| Path | Purpose |
|---|---|
| `/opt/caedral/services/voice-api/releases/v1/docker-compose.yml` | service definition |
| `/opt/caedral/services/voice-api/releases/v1/.env` | `CV_INTERNAL_SECRET` (root-only, chmod 600) |
| `/opt/caedral/services/voice-api/releases/v1/build-context/` | production build context (357 MB) |
| Image | `caedral-voice:v1` (787 MB, python:3.13-slim base) |
| Container | `caedral-voice-api`, compose project `caedral-voice-api` |

## Status / logs / control

```bash
cd /opt/caedral/services/voice-api/releases/v1

docker compose ps                       # status
docker logs --tail 100 caedral-voice-api   # logs
docker compose restart voice-api        # restart Voice ONLY
docker compose stop voice-api           # stop Voice only
docker compose start voice-api          # start Voice only
```

Never `docker restart`, `docker system prune`, or touch other compose
projects on this shared host.

## Health

```bash
curl -s http://127.0.0.1:8600/healthz        # liveness -> {"status":"ok"}
curl -s http://127.0.0.1:8600/readyz         # readiness -> {"ready":true} (200)
curl -s http://127.0.0.1:8600/metrics        # queue depth/counters/RSS/last RTF
```

Readiness after cold start ≈ 30–40 s (model load + warmup); the container
healthcheck has `start_period=120s`.

## Synthesis smoke request

```bash
SECRET=$(sed -n 's/^CV_INTERNAL_SECRET=//p' /opt/caedral/services/voice-api/releases/v1/.env)
curl -s -o /tmp/t.wav -w "%{http_code} %{size_download}\n" \
  -X POST http://127.0.0.1:8600/v1/audio/speech \
  -H "Content-Type: application/json" -H "X-Internal-Secret: $SECRET" \
  -d '{"model":"caedral-voice-1","voice":"caedral-f1","input":"Welcome to Caedral.","speed":1.0,"response_format":"wav"}'
```

## Resource usage

```bash
docker stats --no-stream caedral-voice-api
docker exec caedral-voice-api sh -c \
  'cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/cpu.max'
```

Enforced cgroup values (verified inside container, founder-authorized ceiling
2026-08-25): `memory.max=3999997952 (~4 GB)`, `memory.swap.max=0`,
`cpu.max=400000 100000 (4 vCPU)`. This is a CAP, not a reservation — idle
consumption is ~0.6 GB / 0 CPU. Historical floor: never below 2 GB (observed
engine peak 1412 MB); at 10 s-audio requests the peak is ~1198 MB.

## Redeploy / rollback (V1 image)

```bash
# build new image tag alongside, then:
cd /opt/caedral/services/voice-api/releases/v1
docker tag caedral-voice:v1 caedral-voice:v1-backup-$(date +%Y%m%d)   # rollback point
# ... load/import replacement image as caedral-voice:v1 ...
docker compose up -d          # recreates ONLY the voice container
# rollback:
docker tag caedral-voice:v1-backup-<date> caedral-voice:v1 && docker compose up -d
```

Frozen rule: the deployed runtime must hash-match
`artifacts/caedral-voice-v1/V1-FREEZE-MANIFEST.json`
(model `7d5df8ec…`, voices `bca610b8…`). Verify before promoting:

```bash
docker run --rm --entrypoint sh caedral-voice:v1 \
  -c 'sha256sum /app/models/kokoro-v1.0.onnx /app/models/voices-v1.0.bin'
```

## Changing resource profile

Edit `mem_limit` / `memswap_limit` / `cpus` in the compose file, keep them
EQUAL (`mem_limit == memswap_limit` → no swap dependency), then
`docker compose up -d`. Never go below **2 GB** memory for this engine
(observed peak 1412 MB). Measured profiles for ~10 s audio requests
(`artifacts/server-fire-test-4vcpu/`):

| Cap | gen p50 | sustainable edge | safe rate |
|---|---|---|---|
| 4 GB / 1 vCPU | 28.7 s | ~2.1 RPM | ≤ 1.5 RPM |
| 4 GB / 2 vCPU | 10.6 s | ~5.6 RPM | ≤ 4 RPM |
| 4 GB / 4 vCPU, CV_THREADS=6 (superseded) | 6.3 s | ~9.6 RPM | ≤ 7 RPM |
| **4 GB / 4 vCPU, CV_THREADS=4 (deployed 2026-08-26)** | **3.2 s** | **~15.8 RPM** | **≤ 11 RPM** |

Concurrency stays 1 in all cases (extra concurrency adds only latency; dual
sessions measured +3 % throughput for +345 MB — rejected).

## Threading configuration (deployed optimization, founder-approved)

`CV_THREADS=4` is set via `environment:` in the release compose file
(config-only; image `caedral-voice:v1`, model, code and API unchanged).
The image default `CV_THREADS=6` was tuned on a local 16-thread machine and
oversubscribes the production EPYC's 4-vCPU quota (no SMT, AVX2/FMA only).

Validated evidence (2026-08-26, same-day controlled + production confirm):
- e2e p50 6.32 s → **3.17–3.34 s**, p95 7.25 → 3.74 s, RTF ~0.32 (~3.1× realtime)
- CPU-seconds per canonical render 19.0 → 12.5 (−35 %); throttled periods 57 % → 6–10 %
- Quality: class A (canonical waveform corr 1.0, speaker cosine ≥0.999,
  DNSMOS Δ≤0.002, ASR content-word loss 0); 500-request soak PASS with zero drift
- Rollback: set `CV_THREADS=6` in compose `environment:`, `docker compose up -d`
  (pre-change file kept as `docker-compose.yml.pre-t4-rollback`)
- Full research record: repo branch `voice-perf-experimental`,
  `artifacts/perf-optimization/{EXPERIMENTS.md,FINAL-REPORT.md}`
