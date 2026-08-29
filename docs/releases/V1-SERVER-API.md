# CAEDRAL VOICE V1 — SERVER API (production deployment)

Deployed instance documented from **actual deployed code** (`production/server.py`
in image `caedral-voice:v1`, commit lineage `cff0784a7d89` → `d5ed26a`) and
verified against the live service on the real server.

## Architecture

```
customer ──> Caedral gateway (auth/billing/accounting/public rate limits)
                 │  forwards X-Internal-Secret
                 ▼
   caedral-voice-api (INTERNAL, 127.0.0.1:8600 on 144.91.116.147)
                 │  ONE ORT session · ONE synth worker thread · bounded queue(8)
                 ▼
        frozen Kokoro-foundation engine (caedral-voice-1)
```

| Field | Value |
|---|---|
| Server | `144.91.116.147` (`mail.caedral.com`), Ubuntu 26.04 LTS |
| Deployment path | `/opt/caedral/services/voice-api/releases/v1` |
| Compose project / service | `caedral-voice-api` / `voice-api` |
| Container | `caedral-voice-api` |
| Image | `caedral-voice:v1` (787 MB, python:3.13-slim, non-root) |
| Publish | **`127.0.0.1:8600` → container 8000 (localhost-only, never public)** |
| Model ID | `caedral-voice-1` |
| Voices | `caedral-f1`, `caedral-f2`, `caedral-m1`, `caedral-m2` |
| Language | **ENGLISH ONLY** (Latin-script gate; other scripts → 400) |
| Engine config | FROZEN: ORT CPU, ENABLE_ALL, intra_op=6, arena off |

## POST /v1/audio/speech

### Headers

| Header | Required | Notes |
|---|---|---|
| `Content-Type` | yes | `application/json` |
| `X-Internal-Secret` | **yes** (deployment has `CV_INTERNAL_SECRET` set) | compared constant-time; wrong/missing → `401 UNAUTHORIZED`. Value lives in `/opt/caedral/services/voice-api/releases/v1/.env`; never log or commit it. |

### JSON body

| Field | Type | Required | Default | Allowed / constraints | Violation behavior |
|---|---|---|---|---|---|
| `model` | string | no | `"caedral-voice-1"` | must be exactly `caedral-voice-1` | 400 `UNKNOWN_MODEL` |
| `voice` | string | **yes** | — | one of `caedral-f1`, `caedral-f2`, `caedral-m1`, `caedral-m2` | 400 `UNKNOWN_VOICE` |
| `input` | string | **yes** | — | non-empty after trim; ≤ **2000 chars** (`CV_MAX_INPUT_CHARS`); Latin-script characters only | empty → 400 `EMPTY_INPUT`; >2000 → 400 `INPUT_TOO_LONG`; non-Latin letters → 400 `UNSUPPORTED_LANGUAGE` |
| `speed` | number | no | `1.0` | **[0.25, 2.0] inclusive** (code-enforced range — NOT the OpenAI 0.25–4.0); booleans rejected | non-number/out-of-range → 400 `INVALID_SPEED` |
| `response_format` | string | no | `"wav"` | only `"wav"` | 400 `UNSUPPORTED_FORMAT` |

Unknown JSON fields are ignored. Input text is trimmed before synthesis.
English-only enforcement is a character-class check (letters outside the
Latin repertoire, including Cyrillic/CJK/Arabic, fail fast).

Example request:

```json
{
  "model": "caedral-voice-1",
  "voice": "caedral-f1",
  "input": "Welcome to Caedral.",
  "speed": 1.0,
  "response_format": "wav"
}
```

### Success response

| Property | Value |
|---|---|
| HTTP status | `200` |
| `Content-Type` | `audio/wav` |
| Audio format | RIFF/WAVE, **24000 Hz**, **mono**, **16-bit PCM** |
| Diagnostic headers | `x-caedral-audio-seconds`, `x-caedral-wall-seconds`, `x-caedral-rtf` (generation wall time excludes queue wait; uvicorn emits them lowercase on the wire) |

## Error contract (verified live)

All errors are deterministic JSON: `{"error": {"code": ..., "message": ...}}`.

| HTTP | code | Meaning | Client retry? |
|---|---|---|---|
| 400 | `MALFORMED_JSON` | body is not valid JSON / not an object | fix request |
| 400 | `UNKNOWN_MODEL` | model ≠ `caedral-voice-1` | fix request |
| 400 | `UNKNOWN_VOICE` | voice not one of the four ids | fix request |
| 400 | `EMPTY_INPUT` | input missing/blank | fix request |
| 400 | `INPUT_TOO_LONG` | > 2000 chars | split/truncate |
| 400 | `UNSUPPORTED_LANGUAGE` | non-Latin script characters | fix request |
| 400 | `INVALID_SPEED` | outside [0.25, 2.0] | fix request |
| 400 | `UNSUPPORTED_FORMAT` | format ≠ wav | use wav |
| 401 | `UNAUTHORIZED` | missing/wrong `X-Internal-Secret` | fix credentials |
| 429 | `OVERLOADED` | bounded queue (8) full; **`Retry-After: 5`** present | retry after delay w/ backoff |
| 503 | `NOT_READY` | model still loading/warming (cold start ≈ 30–40 s) | retry until ready |
| 504 | `GENERATION_TIMEOUT` | exceeded `CV_GEN_TIMEOUT_S` = 300 s | retry once; report if persistent |
| 500 | `SYNTHESIS_FAILED` | internal error (details logged server-side only) | retry once; alert if persistent |

## Health / readiness / metrics (no auth required)

| Endpoint | Healthy response |
|---|---|
| `GET /healthz` | `200 {"status":"ok"}` |
| `GET /readyz` | `200 {"ready":true}` once model loaded + warmed (503 before) |
| `GET /metrics` | JSON: `uptime_s, ready, queue_depth, queue_capacity(8), last_rtf, rss_mb, completed, rejected_400, rejected_429, failed, timeouts` |

## Limits & timing (this deployment)

| Setting | Value | Source |
|---|---|---|
| Max input length | 2000 chars | `CV_MAX_INPUT_CHARS` default |
| Generation timeout | 300 s | `CV_GEN_TIMEOUT_S` default |
| Queue capacity | 8 requests (bounded) | `CV_QUEUE_SIZE` default |
| Synthesis concurrency | **1** (single shared ORT session, strict serialization) | frozen design |
| Queue timeout | none server-side beyond generation timeout; **gateway should abandon at ~240 s** | integration guide |
| Cold start to ready | ≈ 30–40 s observed | this deployment |

## Rate-limit recommendation (real-VPS measured 2026-08-25)

Deployed ceiling since 2026-08-25: **4 GB RAM / 4 vCPU CAP** (founder-authorized;
idle use ~0.6 GB / 0 CPU). Measured on `144.91.116.147` — full data in
`artifacts/server-fire-test/SERVER-FIRE-TEST-REPORT.md` (30 s canonical) and
`artifacts/server-fire-test-4vcpu/SERVER-4VCPU-10S-REPORT.md` (10 s canonical):

| Request class | gen p50 @ 4 vCPU | Measured sustainable edge | Recommended safe rate |
|---|---|---|---|
| ~10 s audio | **6.3 s** (RTF 0.63, faster than realtime) | ~9.6 RPM | **≤ 7 RPM** (≈ 420 req/h) |
| ~30 s audio (long-form worst case) | 37 s* | ~1.8 RPM* | ≤ 1.25 RPM |

*30 s numbers measured at the earlier 2048 MB / 2 vCPU envelope; at the
4-vCPU cap expect roughly half the 1-vCPU wall times, but plan long-form
capacity conservatively until re-measured.

- Concurrency: keep customer-facing effective concurrency at **1**; extra
  concurrent requests only add wait (measured speedup = none).
- Overload behavior note (frozen code): the endpoint admits requests
  serially, so excess concurrent connections wait in the HTTP layer instead
  of receiving fast 429s. The gateway MUST therefore own public rate limits
  and abandonment timeouts (~30 s for short-form is comfortable at the cap;
  keep ~240 s for long-form).

## Resource requirements

- Memory: **≥ 1.4 GB hard limit**, `mem_limit == memswap_limit` (no swap
  dependency). Never below ~1.25 GB (engine peaks ≈ 1023 MB per long request).
- CPU: **≥ 1.0 vCPU** (quota-enforced; thread count stays frozen at 6)
- Disk: image 787 MB + logs

## Examples

`$CAEDRAL_INTERNAL_SECRET` is a placeholder for the value of
`CV_INTERNAL_SECRET` in `/opt/caedral/services/voice-api/releases/v1/.env`
— never hard-code it.

### A. curl (run on the server)

```bash
curl -s -o out.wav \
  -X POST http://127.0.0.1:8600/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: $CAEDRAL_INTERNAL_SECRET" \
  -d '{"model":"caedral-voice-1","voice":"caedral-f1","input":"Welcome to Caedral.","speed":1.0,"response_format":"wav"}'
```

### B. Python (requests / httpx)

```python
import os, requests  # pip install requests

resp = requests.post(
    "http://127.0.0.1:8600/v1/audio/speech",
    headers={"X-Internal-Secret": os.environ["CAEDRAL_INTERNAL_SECRET"]},
    json={
        "model": "caedral-voice-1",
        "voice": "caedral-f1",
        "input": "Welcome to Caedral.",
        "speed": 1.0,
        "response_format": "wav",
    },
    timeout=300,
)
resp.raise_for_status()
open("out.wav", "wb").write(resp.content)
print(resp.headers["x-caedral-rtf"])  # diagnostic RTF
```

### C. JavaScript (fetch, Node 18+)

```javascript
const resp = await fetch("http://127.0.0.1:8600/v1/audio/speech", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Internal-Secret": process.env.CAEDRAL_INTERNAL_SECRET,
  },
  body: JSON.stringify({
    model: "caedral-voice-1",
    voice: "caedral-f1",
    input: "Welcome to Caedral.",
    speed: 1.0,
    response_format: "wav",
  }),
});
if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
const wav = Buffer.from(await resp.arrayBuffer());
```

### D. PowerShell (from an internal admin host over SSH tunnel)

```powershell
$secret = $env:CAEDRAL_INTERNAL_SECRET
$body = @{
  model           = "caedral-voice-1"
  voice           = "caedral-f2"
  input           = "Welcome to Caedral."
  speed           = 1.0
  response_format = "wav"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8600/v1/audio/speech" `
  -Headers @{ "X-Internal-Secret" = $secret } `
  -ContentType "application/json" -Body $body -OutFile "out.wav"
```

### E. OpenAI client compatibility

**Not compatible out of the box.** The schema is OpenAI-audio-speech-shaped,
but: model id is `caedral-voice-1`, speed range is **[0.25, 2.0]** (not
0.25–4.0), only `response_format: "wav"` exists, and the internal-secret
header is mandatory. Pointing a stock OpenAI SDK client at this endpoint will
fail auth and possibly validation. If the gateway wants SDK compatibility it
must translate (map model ids, clamp speed, inject the internal header).

## Security notes

- Raw runtime is published on `127.0.0.1` only; ufw default-DROP keeps the
  port unreachable externally. No public exposure without founder approval.
- Every `/v1/*` call requires `X-Internal-Secret` (constant-time compare).
- Secret storage: root-owned `.env` chmod 600 next to the compose file;
  injected via compose environment interpolation.
- Logs contain voice id, char count, durations, RTF, status codes — **no
  secrets, no API keys, no customer text** (audited post-deploy).
- Container runs as non-root uid 10001; healthcheck wired to `/readyz`.
- Gateway owns customer authentication, billing, accounting and public rate
  limiting; Voice never sees customer credentials.
