# CAEDRAL VOICE V1 — PRODUCTION INTEGRATION GUIDE

Audience: main Caedral platform coordinator. The Voice engine is FROZEN;
this document covers integration only. Auth/billing/credits/public rate
limits/accounting are GATEWAY responsibilities; this service is an INTERNAL
model service (same pattern as `model-inference`: FastAPI/uvicorn,
`X-Internal-Secret`, localhost-only publishing).

## Identity

| Field | Value |
|---|---|
| Model ID | `caedral-voice-1` |
| Voices | `caedral-f1`, `caedral-f2`, `caedral-m1`, `caedral-m2` |
| Language | **ENGLISH ONLY** in V1 (non-Latin scripts rejected with 400 `UNSUPPORTED_LANGUAGE`) |
| Output audio | WAV RIFF, 24 kHz, mono, 16-bit PCM |
| Engine | frozen Kokoro-foundation runtime (`V1-FREEZE-MANIFEST.json`, commit `cff0784a7d89`) |
| Session model | ONE ORT session per process, shared by all voices, warm-once |

## Internal endpoint

`POST /v1/audio/speech` (service port default 8000; publish localhost-only)

```json
{
  "model": "caedral-voice-1",
  "voice": "caedral-f1",
  "input": "Welcome to Caedral.",
  "speed": 1.0,
  "response_format": "wav"
}
```

Response: `audio/wav` body + diagnostic headers
`X-Caedral-Audio-Seconds`, `X-Caedral-Wall-Seconds`, `X-Caedral-RTF`.

## Error contract (deterministic JSON)

| HTTP | code | cause |
|---|---|---|
| 400 | `MALFORMED_JSON` | body is not valid JSON |
| 400 | `UNKNOWN_MODEL` | model != `caedral-voice-1` |
| 400 | `UNKNOWN_VOICE` | voice not in the four production ids |
| 400 | `EMPTY_INPUT` | input missing/blank |
| 400 | `INPUT_TOO_LONG` | input > `CV_MAX_INPUT_CHARS` (default 2000) |
| 400 | `UNSUPPORTED_LANGUAGE` | non-Latin script characters |
| 400 | `INVALID_SPEED` | speed outside [0.25, 2.0] |
| 400 | `UNSUPPORTED_FORMAT` | response_format != wav |
| 401 | `UNAUTHORIZED` | missing/wrong `X-Internal-Secret` (when `CV_INTERNAL_SECRET` set) |
| 429 | `OVERLOADED` | bounded queue full; `Retry-After: 5` |
| 503 | `NOT_READY` | model still warming up |
| 504 | `GENERATION_TIMEOUT` | exceeded `CV_GEN_TIMEOUT_S` |
| 500 | `SYNTHESIS_FAILED` | internal error (traceback logged server-side only) |

Health/readiness: `GET /healthz` (process alive), `GET /readyz` (200 only
after model loaded + warmed). `GET /metrics`: queue depth, counters, RSS.

## Resource reality (FIRE TEST, 2026-08-24)

Measured under hard cgroup limits — see
`artifacts/fire-test/FIRE-TEST-REPORT.md`. Both founder-specified profiles
were allowed to fail honestly:

| Deployment | Verdict | Evidence |
|---|---|---|
| 500 MB / 0.5 vCPU | **UNSUPPORTED** — OOM during model load | `profile-500mb-0.5vcpu.json` |
| 1000 MB / 1 vCPU | **UNSUPPORTED** — load OK (511 MB warm) but first ~30 s generation OOM-killed (peak 1023 MB) | `profile-1gb-1vcpu.json` |
| 1400 MB / 1 vCPU | **SUPPORTED** — 10/10 generations, deterministic 1023 MB peak | `profile-aplus-1.4gb-1vcpu.json` |

At 1 vCPU a canonical 30-s request takes p50 66 s / p95 93 s (RTF 2.22/3.11).
Measured sustainable capacity = **0.91 RPM** (queue-stable at the 100 % level;
beyond it throughput pins at the serial limit and latency grows linearly —
demonstrated at the 125 % level with e2e p50 173 s). Concurrency >1 only adds
queue wait (bursts of 2/4 serialized perfectly, zero failures).
Recommended production settings (with ~28 % headroom):

| Setting | Value |
|---|---|
| Memory limit | ≥ 1.4 GB (never below 1.25 GB) |
| CPU | ≥ 1.0 vCPU per instance |
| Max concurrency (synthesis) | **1** (single shared session; more workers only add queue wait) |
| Queue size | 8 (bounded; overflow = clean 429) |
| Recommended sustained rate | **≤ 0.65 RPM** for 30-s requests |
| Generation timeout | 180 s (> measured p95 93 s × 2) |
| Queue timeout | client-side; gateway should time out at 240 s |

Shorter text scales roughly with length: sub-second control sentences render
at RTF 0.45–0.86 even on an idle full machine, so mixed traffic does much
better than the worst-case 30-s number.

## Docker

```bash
bash scripts/stage_prod_context.sh
docker build -f production/Dockerfile -t caedral-voice:v1 production/context

docker run -d --name caedral-voice-api \
  --memory=1400000000b --memory-swap=1400000000b --cpus=1.0 \
  -p 127.0.0.1:8600:8000 \
  -e CV_INTERNAL_SECRET="$MODEL_INFERENCE_SECRET" \
  caedral-voice:v1
```

Image: pinned python:3.13-slim, non-root user, healthcheck wired to
`/readyz`; contains ONLY production files (frozen model+voices baked,
frontend data baked). NO torch, NO whisper/DNSMOS/ASR, NO training stack.
Image size: **787 MB**. Smoke test: 17/17 PASS (fresh constrained container).

Environment variables: `CV_MODEL_PATH`, `CV_VOICES_PATH`, `CV_THREADS=6`
(frozen value — do not change), `CV_MAX_INPUT_CHARS`, `CV_QUEUE_SIZE`,
`CV_GEN_TIMEOUT_S`, `CV_INTERNAL_SECRET`, `CV_PORT`.
No secrets are required at build time; no credentials ever belong in this
service's logs or env beyond the internal shared secret above.

## Compose block (drop-in for main platform docker-compose.yml)

```yaml
  voice-api:
    image: caedral-voice:v1
    container_name: caedral-voice-api
    restart: unless-stopped
    # Localhost-only publish (internal model service, like model-inference).
    ports:
      - "127.0.0.1:8600:8000"
    cpus: 1.0
    mem_limit: 1400000000
    memswap_limit: 1400000000        # no swap spill
    environment:
      - CV_INTERNAL_SECRET=${MODEL_INFERENCE_SECRET:-}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 120s
```

Gateway routing: proxy authenticated customer traffic to
`http://caedral-voice-api:8000/v1/audio/speech`, forwarding the
`X-Internal-Secret` header; apply public rate limits, billing (duration-priced
speech output), and usage accounting there. The gateway already has a
`speech` output modality in pricing/model-routing.

## Validation performed

- Production smoke test (all voices, error contract, overload survival):
  see `production/smoke_test.py`; result recorded in final handoff.
- Packaging memory validation (fresh container, one canonical request):
  recorded in fire-test report §Packaging.

ARCH MIGRATION: DO NOT START (founder gate pending).
