# CAEDRAL INFERENCE RUNTIME — DESIGN (Python-first, ONNX-ready)

**Task:** RUNTIME-01 · **Owner:** inference-runtime agent · **Date:** 2026-08-22
**Status:** DRAFT v1 for review · **Scope:** `docs/inference/RUNTIME-DESIGN.md` only (this file)
**Upstream inputs:** CAEDRAL-NAR prototype (`caedral_voice/model/`), text frontend (`caedral_voice/text/`),
voice catalog schema (`configs/voices/README.md`), experiments E0001/E0002/E0006,
Phase C baselines (`docs/benchmarks/BASELINE-MEASURED.md`),
CPU playbook (`docs/research/CPU-INFERENCE-OPTIMIZATION.md`).

Claim labels follow charter §9: `[MEASURED]` / `[REPRODUCED FROM SOURCE]` /
`[REPORTED BY THIRD PARTY]` / `[ESTIMATED]` / `[HYPOTHESIS]` / `[DECISION]`.

---

## 0. Summary

The G1 runtime is a **single-process, CPU-only synthesis service** around the frozen
CAEDRAL-NAR wiring (phoneme frontend → variance predictors → conditional flow decoder →
iSTFT head, 24 kHz out). Time-to-first-audio (TTFA) is manufactured by *our* runtime,
not the model: every baseline tested in Phase C showed native streaming is useless
(Kokoro emits 1 chunk for typical inputs) so **caller-side sentence chunking with
sequential chunk synthesis and incremental PCM handoff is mandatory** `[MEASURED]`
(BASELINE-MEASURED lesson 1). Torch-eager is the v0 backend; the API and session layer
are written ONNX-ready so Phase L swaps `TorchBackend` for `OnnxBackend` without touching
the public surface. Deployment default target: profile B (2 vCPU, ≤2 GB) per MASTER-PLAN §5.

---

## 1. What exists today (design inputs)

| Component | State | Key contract |
|---|---|---|
| Acoustic model | `CaedralNAR` prototype, untrained; `use_flow=True` is the G1 path | `infer(phoneme_ids, speaker_id, style_id, speed, pitch_shift, energy_gain, noise_scale, durations)` → dict with `waveform` `(B, T_y*hop)` fp32 |
| Signal geometry | sr 24000, hop 256, n_fft 1024 | **93.75 mel frames/s**; iSTFT head outputs waveform directly (no separate vocoder network for G1 flow path) |
| Text frontend | `TextFrontend(backend="lexicon")`, GPL-free | `process(text) -> FrontendResult` with `phoneme_tokens` incl. pause marks `\| ‖`; batch-1 cost 0.187 ms/item median `[MEASURED, E0001]` |
| Conditioning | speaker table d=128 + style table (10+neutral), FiLM | `speaker_id` int, `style_id` int; knobs transform tracks exactly (tests/test_model.py) |
| Voice catalog | schema frozen in `configs/voices/README.md` | `prosody_ranges.speed [0.85,1.15]`, `pitch_shift_semitones [-2,2]`, `energy_gain [0.9,1.1]`; rule 2: clamp at API layer |
| Measured latency anchors | E0002/E0006/Vocos runs on this box | see §3.4 tables |

---

## 2. Public API

### 2.1 Signature

```python
# caedral_voice/inference/api.py
def synthesize(
    text: str,
    voice_id: str,
    style: str | None = None,          # None -> voice.default_style
    speed: float | None = None,        # None -> 1.0 (rate multiplier)
    pitch_shift: float | None = None,  # None -> 0.0 (semitones)
    energy_gain: float | None = None,  # None -> 0.0 (dB)
    format: Literal["wav", "pcm-stream"] = "wav",
    *,                                 # keyword-only extras
    seed: int | None = None,           # reproducible latent sampling
    strict: bool = False,              # True: style/knob violations raise instead of clamp
) -> SynthesisResult
```

- `format="wav"` → `SynthesisResult.audio_bytes` holds a complete RIFF/WAVE file
  (16-bit PCM mono 24 kHz). Whole job completes before return (no partial WAV).
- `format="pcm-stream"` → `SynthesisResult.stream` is a lazy iterator yielding raw
  little-endian int16 PCM byte chunks (one yield per synthesized sentence chunk +
  inter-chunk silence). Pull-based ⇒ natural backpressure; nothing is synthesized
  until the consumer asks.

### 2.2 Validation rules

Checked in order; all failures raise `InferenceError(code, message)` (machine-readable
codes, mapped to HTTP status in §7).

| # | Check | Rule | Error code |
|---|---|---|---|
| V1 | text type | must be `str` (bytes/None/int rejected) | `INVALID_TEXT_TYPE` |
| V2 | text length | `len(text) ≤ 100_000` chars — program charter §19 bound | `TEXT_TOO_LONG` |
| V3 | text content | NUL and non-whitespace C0 control chars rejected | `MALFORMED_TEXT` |
| V4 | empty text | empty or whitespace-only | `EMPTY_TEXT` |
| V5 | post-normalization emptiness | `normalize_detailed()` output contains no word/digit token (e.g. input was only emoji/punctuation) | `EMPTY_AFTER_NORMALIZATION` |
| V6 | voice existence | `voice_id` matches `^[a-z0-9][a-z0-9_-]{0,63}$` AND resolves in loaded catalog index | `UNKNOWN_VOICE` |
| V7 | voice availability | catalog entry flagged blocked (provenance not cleared, Phase N gate) | `VOICE_BLOCKED` |
| V8 | style membership | `style` ∈ voice.`allowed_styles`; else fall back to `default_style` + note (raise if `strict=True`) | `STYLE_NOT_ALLOWED` / note |
| V9 | knob types | floats (ints accepted); NaN/±inf rejected outright | `INVALID_KNOB` |
| V10 | knob ranges | clamped to the voice's `prosody_ranges` (§2.3); clamped values recorded in result notes | notes / `INVALID_KNOB` under `strict` |

Long-input behavior: 100 k chars is accepted but processed strictly through the §3
chunk pipeline (bounded per-chunk memory). This is the direct mitigation of the
KittenTTS long-form hard-crash failure mode `[MEASURED, Phase C]`.

### 2.3 Prosody units and clamping `[DECISION]`

Model semantics are fixed by `caedral_nar.py`: `speed` divides clamped durations;
`pitch_shift` rescales F0 by `2^(st/12)`; `energy_gain` adds **dB**. The catalog
schema expresses `energy_gain` as a linear multiplier range `[0.9, 1.1]`. Reconciliation:

- API canonical units: **speed ×** , **pitch semitones**, **energy dB**.
- Catalog loader converts a linear energy range to dB at load time:
  `db = 20·log10(gain)` → e.g. codename defaults become `[-0.92, +0.83] dB`.
  Voice files may alternatively declare `"unit": "db"` explicitly; loader validates.
- Clamping happens exactly once, at the API layer (catalog README rule 2); the model
  never sees out-of-range values. Applied values are echoed in
  `SynthesisResult.metadata["knobs_effective"]` and any clamps in `notes[]`.

### 2.4 Result object

```python
@dataclass
class SynthesisResult:
    voice_id: str
    style_effective: str
    sample_rate: int                  # 24000
    bits_per_sample: int              # 16
    channels: int                     # 1
    audio_bytes: bytes | None         # format="wav"
    stream: Iterator[bytes] | None    # format="pcm-stream"
    metadata: dict                    # n_chunks, knobs_effective, seed, per-stage timings
    notes: list[str]                  # clamps, style fallback, frontend notes
```

### 2.5 Determinism and seeding

Latent sampling (`z = m_e + ns·σ_e·ε`) runs through a per-request `torch.Generator`
seeded from `seed` or the engine default. Same engine build + same seed + same inputs ⇒
bitwise-identical PCM (asserted in tests; mirrors the same-seed bitwise discipline of
E0002/E0006). ONNX backend passes ε as an explicit input tensor so determinism survives
the runtime swap.

---

## 3. TTFA architecture

### 3.1 Pipeline

```
text ──► validate ──► sentence-split chunker ──► per-chunk loop (sequential):
                                                    frontend(G2P) → acoustic infer → iSTFT head
                                                  ──► int16 quantize ──► yield PCM [+ silence pad]
       wav path: consume whole generator → RIFF-wrap → audio_bytes
```

Sequential chunk synthesis (no lookahead batching) in v0: chunk *i+1* starts only after
chunk *i* is yielded. Because per-chunk RTF is far below 1.0 (§3.4), the producer
outruns any listener after the first chunk and the pull-based stream self-throttles.

### 3.2 Chunker spec (`chunker.py`)

- Deterministic, stdlib-only sentence segmentation: split on `[.!?…]+` followed by
  whitespace/newlines; paragraph breaks always split; abbreviation guard set
  (Mr., Mrs., Dr., Prof., St., vs., etc., e.g., i.e., initials, decimals) prevents
  false splits. No ML, no locale surprises, unit-testable goldens.
- **Merge** adjacent sentences while combined length < `min_chars` (default 80) —
  avoids choppy prosody on staccato input.
- **Split** overlong sentences at clause punctuation (`, ; : —`) when above
  `max_chars` (default 400 chars ≈ 380 phonemes at the measured corpus-wide ratio
  0.944 ph/char `[MEASURED, E0001]`), targeting the ≤600-phoneme working point of the
  E0002/E0006 measurement cells.
- Each chunk carries source offsets (`start`, `end`) so results can be traced to
  input spans; chunks are independently normalized + phonemized.
- Per-chunk phoneme ceiling 600 enforced post-G2P; an over-ceiling chunk is
  force-split at the last vowel-to-vowel boundary (fallback: word boundary).

### 3.3 Stream assembly (`stream.py`)

- Quantize fp32 → int16 with peak-normalization guard per chunk (clip counter in
  metadata; no cross-chunk gain riding — loudness consistency is a Phase K problem).
- Inter-chunk silence: exact zeros, default 150 ms (3600 samples @24 kHz)
  `[HYPOTHESIS — placeholder until Phase J/K establishes per-voice long-form defaults,
  as reserved by configs/voices/README.md rule 4]`.
- Chunk-boundary artifacts (prosody discontinuity across sentence joins) accepted for
  G1; overlap-blend schemes deferred to Phase K (documented non-goal now).

### 3.4 TTFA model: formula + measured anchors

**Formula.**

```
TTFA = t_queue + t_frontend(c₁) + t_acoustic(L₁) + t_vocoder(F₁) + t_mux
  c₁      = first sentence chunk;            L₁ = its phoneme count
  F₁      = its mel frames ≈ r̂·L₁  (r̂ = trained frames-per-phoneme, expect ~4–6;
            untrained stub predictors sit at 1.0–1.3 fpp — E0002 R-method / E0006 S2)
  t_mux   = int16 quantize + silence fill + iterator hop (< 1 ms for utterance chunks)
```

**Measured anchors on this machine** (i7-13620H, Win11, py3.13, torch 2.6.0+cpu):

| Stage | Cell | Value | Source |
|---|---|---|---|
| frontend | batch-1 G2P, median/item | **0.187 ms** | `[MEASURED, E0001]` |
| acoustic, tiny (8.35 M) | fwd @50/200/800 ph, 1 thr | 19.4 / 56.0 / 228.6 ms | `[MEASURED, E0002]` |
| acoustic, lite (31.6 M) | fwd @200/800 ph, 1 thr / 4 thr | 194.2 / 891.2 ms · 123.8 / 592.4 ms | `[MEASURED, E0002]` |
| acoustic, lite+flow (18.9 M) | **inference**, matched 1-fpp, @200/800 ph, 1 thr / 4 thr | 87.8 / 424.3 ms · 66.7 / 293.1 ms | `[MEASURED, E0006]` |
| acoustic, lite+flow | inference @200 ph, own predicted durations (262 frames), 1 thr | 117.1 ms | `[MEASURED, E0006 informational]` |
| acoustic, quality (80.7 M) | 800-frame anchor (realistic expansion proxy), 1 thr / 4 thr | 1342.8 / 730.7 ms | `[MEASURED, E0002]` |
| iSTFT head decode (pretrained Vocos-mel-24k, MIT) | decode RTF, 4 thr / 1 thr | **0.012–0.021 / 0.033–0.048** (215–2117-frame mels) | `[MEASURED, benchmarks/results/decoders/vocos-*.jsonl]` |

Note the G1 flow path folds the "vocoder" into the head, so `t_vocoder` above is a
conservative stand-in for head cost carried inside the acoustic pass; the standalone
Vocos rows bound what a split decode would add if the Phase-G/L export separates them.

**Worked example** `[ESTIMATED — arithmetic over the MEASURED cells above]`:
first chunk ≈ 120 phonemes ≈ 540 frames (r̂=4.5) ≈ 5.8 s audio, lite+flow config:

| Thread class | acoustic (frame-slope interp.) | head allowance (Vocos RTF × 5.8 s) | frontend+mux | **TTFA** |
|---|---|---|---|---|
| 1 thr | ≈ 250–300 ms | ≈ 190–280 ms | < 5 ms | **≈ 0.45–0.58 s** |
| 4 thr | ≈ 120–150 ms | ≈ 70–120 ms | < 5 ms | **≈ 0.19–0.27 s** |
| tiny, 1 thr | ≲ 60 ms (≤200-ph cells) | ≈ 190–280 ms | < 5 ms | **≈ 0.25–0.34 s** |

Sanity cross-check: these sit well under the Kokoro baseline where TTFA ≈ full
generation (native streaming yields 1 chunk) — e.g. para_001 (22.6 s audio) took
seconds-to-tens-of-seconds before first audible sample `[MEASURED, Phase C]`.

**Steady-state.** Post-first-chunk starvation requires chunk RTF ≥ 1; worst MEASURED
batch-1 cell anywhere is 0.23 (quality, 800 ph, 1 thr) `[E0002]`, and even E0002's
pessimistic 5–6× expansion extrapolation lands near RTF ≈ 0.9–1.1 only for
quality-class single-pass monsters at ~51 s audio — which the 600-phoneme chunk cap
prevents structurally. Pull-based iteration means no unbounded buffering either way.

---

## 4. Session lifecycle

### 4.1 Load once; share safely

- One process loads **one `Engine` per (config_name, backend)** at startup into a
  process-wide registry; construction is serialized by a lock; after construction the
  object is immutable and shared read-only across request threads.
- Torch path: `model.eval()` + all inference under `torch.inference_mode()`;
  batch-1 tensors are request-local. No weight mutation post-load (knobs are track
  transforms, never weight edits).
- ORT path: `InferenceSession.Run()` is documented thread-safe for concurrent calls
  `[REPORTED — ORT docs, cited in CPU playbook §7.1]`; only construction needs the lock.
- Two named sessions per model when thread classes differ (e.g. `short`=1 intra-op
  thread, `long`=4): weights are mmapped/shared, sessions differ only in options.
  Rationale: 4 threads measurably *hurt* short utterances (T=50 regressions up to +52%)
  while helping long ones (≤1.91×) `[MEASURED, E0002 interp. 3]`.

### 4.2 Backend matrix (ONNX-ready)

| Backend | When | Notes |
|---|---|---|
| `TorchBackend` (eager) | v0, correctness reference | matches all E0002/E0006 measurement conditions; no compile step |
| `OnnxBackend` | Phase L | exported graph(s): encoder+variance, flow-inverse, head (or fused); SEQUENTIAL exec, spinning tuned per playbook §1.2; pre-optimized ORT-format model to cut startup `[REPORTED, playbook §5.2]` |

`Engine.backend` exposes `synthesize_chunk(phoneme_ids, spk, sty, knobs) -> fp32
waveform` — identical shape contract for both, so §2 API, §3 pipeline and tests are
backend-agnostic.

### 4.3 Memory-mapped weights option

- Preferred serialization for inference: **safetensors** (zero-copy mmap, no pickle
  executor) or ONNX external-data layout, where tensor bytes live in disk-backed pages
  the OS manages — helps cold-start page-in overlap `[HYPOTHESIS on Windows page-cache
  behavior, per playbook §5.2 — verify]`. `torch.load(..., mmap=True, weights_only=True)`
  is the fallback for `.pt` checkpoints.
- Consequence: multiple sessions/configs can coexist within RAM envelopes without
  duplicating tensor pages; RSS still counted against envelopes at Phase M
  (lite 334 MB / quality 528 MB fresh-process peaks today `[MEASURED, E0002]`;
  tiny's 235 MB misses the ≤200 MB Phase-M hypothesis purely on the ~215 MB
  CPython+torch floor — profile C/D need the ONNX/small-runtime path, NOT a smaller
  acoustic model `[MEASURED, E0002 interp. 4]`).

### 4.4 Warmup strategy

Per CPU playbook §(c), at boot, before `/healthz` reports ready:

1. Load + hash-verify artifacts (§6).
2. Run 3 representative utterances (short ≤50 chars, medium 200–400, long 800–1200)
   through the full pipeline per configured thread class; discard results.
   Purpose: kernel/code-path selection, arena high-water stabilization, JIT-free but
   allocator-warm state; first-`Run()` cost is the known dominant surprise
   `[REPORTED, playbook §5.2 / ORT thread 14]`.
3. Record warmup timings into the engine's self-report (`GET /healthz` exposes them);
   they are excluded from all steady-state metrics forever (protocol rule).

### 4.5 Memory discipline in the service loop

- Bucketed chunk lengths (§3.2) keep activation working sets bounded and predictable;
  enable arena shrinkage (ORT) / explicit `del`+gc between chunks only if a torture
  test shows RSS ratchet (playbook checklist step 11).
- Long-document requests hold only: input text, chunk queue, current-chunk tensors,
  emitted-bytes-so-far for `wav` mode (unavoidable full-file buffer ≈ 48 kB/s audio →
  100 k-char worst case ≈ 2 h audio ≈ 350 MB transient — acceptable for v0, revisited
  with a streaming-WAV writer if torture shows pressure).

---

## 5. Cancellation semantics

Model: cooperative token (`threading.Event`), checked at every stage boundary —
there is **no safe way to preempt a running forward pass**, so:

```
per chunk: [token?] → frontend → [token?] → acoustic pass (uninterruptible) →
           [token?] → head/quantize → [token?] → yield
between chunks: token check is the primary cancel point
```

- **Bound on cancel latency** = at most one chunk's remaining stage work. Worst
  MEASURED single acoustic pass: 2181 ms (quality, 800 ph, 1 thr) `[E0002]`; at the
  default lite+flow config and ≤600-ph chunks the pass bound interpolates to roughly
  0.3–1 s `[ESTIMATED]`. Documented per-release alongside safe concurrency.
- **pcm-stream:** consumer closes the generator (GeneratorExit) or the token fires →
  loop stops scheduling new chunks, partial stream simply ends (PCM has no trailer);
  cleanup = drop chunk tensors. HTTP v0 detects client disconnects best-effort via
  write errors (BrokenPipeError → token.set()); proper disconnect propagation arrives
  with ASGI (§7.3).
- **wav:** cancellation before completion raises `CANCELLED` and returns nothing
  (no partial file leaves the process).
- After a cancelled job the engine remains valid (stateless chunks); a poison-job
  cannot affect later requests — verified by a post-cancel golden-sample test.

---

## 6. Security

Threat model: hostile/corrupted model artifacts, tampered voice configs, malicious
text payloads, resource-abuse via concurrency. Local-trust process boundary; v0 binds
localhost by default.

1. **No eval / no pickle at inference.** Weights via safetensors (or ONNX protobuf);
   if a legacy `.pt` must be read: `torch.load(..., weights_only=True)` hard-required.
   Configs/manifests parsed with stdlib `json` only (no YAML loaders). No dynamic
   imports, no `eval`/`exec`/`marshal` anywhere in `caedral_voice/inference/*`
   (enforced by grep-gate in CI).
2. **Hash-verified loading.** Every artifact directory ships `manifest.json` with
   sha256 for each file (weights, voice JSONs, lexicon, config JSON). Engine refuses
   to construct on any mismatch (`ARTIFACT_HASH_MISMATCH`), recording expected vs
   actual in the error. Manifest itself is signed-out-of-band for release builds
   (Phase N concern; dev manifests unsigned but committed).
3. **Path-traversal-proof artifact paths.**
   - `voice_id` and artifact names validated against `^[a-z0-9][a-z0-9_-]{0,63}$`
     before touching the filesystem; user strings are never concatenated into paths.
   - All reads resolve via `Path.resolve()` then assert containment:
     `resolved.is_relative_to(artifacts_root.resolve())` plus symlink rejection on
     the final component. Catalog enumeration comes from an allowlist index file,
     never from directory scanning of user-influenced trees.
4. **Input abuse limits.** Charter §19 100 k-char cap (V2) doubles as the CPU/DoS
   bound; per-request phoneme budget derives from it (≈94 k phones at the measured
   0.944 ratio); server admits bounded concurrency (default 2 workers on profile B)
   with a bounded queue — excess gets `OVERLOADED` (503) immediately, never queued
   unboundedly.
5. **No secrets in artifacts; no outbound network** at inference time (frontend
   lexicons are local files; model downloads are an offline packaging step).

---

## 7. Interfaces

### 7.1 CLI (`cli.py`, argparse)

```
python -m caedral_voice.inference.cli synth --voice codename-f01 --text "..." \
        [--style warm] [--speed 1.0] [--pitch-shift 0] [--energy-gain 0] \
        [--format wav|pcm] [--out out.wav|--out - ] [--seed 1234] [--config configs/lite.json]

python -m caedral_voice.inference.cli serve --host 127.0.0.1 --port 8120 --config configs/lite.json
python -m caedral_voice.inference.cli warmup --config configs/lite.json   # §4.4, exits nonzero on failure
python -m caedral_voice.inference.cli voices                              # dump catalog + effective ranges
```

Exit codes mirror `InferenceError.code`s; stdout carries PCM only when `--out -` and
`--format pcm` (binary-safe), human-readable info goes to stderr.

### 7.2 HTTP v0 sketch (stdlib only)

```python
# server.py (sketch — v0 ships exactly this, minus comments)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz": self._json(200, ENGINE.health())
        elif self.path == "/v1/voices": self._json(200, CATALOG.describe_all())

    def do_POST(self):                      # POST /v1/synthesize
        req = self._read_json()             # {text, voice_id, style?, speed?,
                                            #  pitch_shift?, energy_gain?, format?}
        fmt = req.get("format", "wav")
        result = synthesize(req["text"], req["voice_id"], style=req.get("style"),
                            speed=req.get("speed"), pitch_shift=req.get("pitch_shift"),
                            energy_gain=req.get("energy_gain"), format=fmt)
        if fmt == "wav":
            self._bytes(200, result.audio_bytes, "audio/wav")
        else:                               # chunked transfer of the PCM stream
            self.send_response(200); self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Type", "audio/pcm;rate=24000"); self.end_headers()
            try:
                for pcm in result.stream:
                    self.wfile.chunked_write(pcm)
            except (BrokenPipeError, ConnectionResetError):
                TOKEN.set()                 # client left: stop synthesizing (§5)

EXECUTOR = ThreadPoolExecutor(max_workers=2)   # = admitted concurrency (§6.4)
ThreadingHTTPServer(("127.0.0.1", 8120), Handler).serve_forever()
```

| HTTP | Code | Condition |
|---|---|---|
| 400 | `INVALID_*`, `EMPTY_*`, `MALFORMED_TEXT` | validation §2.2 |
| 404 | `UNKNOWN_VOICE` | V6 |
| 413 | `TEXT_TOO_LONG` | V2 (charter §19) |
| 409 | `VOICE_BLOCKED` | V7 |
| 422 | `STYLE_NOT_ALLOWED` (strict) | V8 |
| 503 | `NOT_READY` (warmup incomplete), `OVERLOADED` | §4.4, §6.4 |
| 500 | `INTERNAL` | bug; logged with job id, no stack leak to client |

### 7.3 Upgrade path (later phases)

FastAPI + uvicorn behind the same `api.synthesize`, bounded worker pool sized by the
playbook §7.3 safe-concurrency procedure, SSE/WebSocket for true mid-flight cancel
propagation, then OpenAI-compatible `/audio/speech` shim if demanded. None of these
change the core API; `server.py` is deliberately disposable.

---

## 8. File layout (`caedral_voice/inference/`)

```
caedral_voice/inference/
  __init__.py      # re-export synthesize, SynthesisResult, InferenceError, Engine
  api.py           # synthesize(): validation V1–V10, clamping, result assembly
  chunker.py       # sentence-split chunker (deterministic, offsets, budgets)
  engine.py        # Engine: registry, thread classes, warmup, per-chunk loop
  backend_torch.py # TorchBackend: CaedralNAR.infer wrapper, generator seeding
  backend_onnx.py  # OnnxBackend (Phase L stub interface in v0)
  session.py       # artifact loading: safetensors/torch(mmap)/ORT session builders
  voices.py        # catalog load/validation, unit conversion, prosody clamping
  security.py      # sha256 manifest verify, path containment, ID regex gates
  stream.py        # int16 encode, silence pads, PCMStreamIter, CancellationToken
  errors.py        # InferenceError + code enum (single source for HTTP map)
  cli.py           # argparse entrypoints (synth/serve/warmup/voices)
  server.py        # stdlib http.server v0 (disposable; FastAPI later)
tests/
  test_inference_api.py     # V1–V10 table-driven, clamping, units
  test_inference_stream.py  # TTFA ordering, chunk boundaries, silence, cancel
  test_inference_security.py# traversal attempts, hash mismatch, pickle ban grep
```

Dependency posture: v0 adds **no new third-party imports** beyond the existing env
(torch/numpy already present); safetensors arrives with the first real checkpoint
packaging; `onnxruntime` enters at Phase L only.

---

## 9. Test-plan hooks (what implementation must prove)

1. Determinism: same seed ⇒ bitwise-equal PCM (torch path), ε-tensor equality across backends.
2. Validation: every row of §2.2 has a unit test including the 100 001-char rejection.
3. Streaming: first-yield precedes full-completion; chunk count == chunker count; cancel
   mid-stream stops within one chunk stage-boundary (timing assertion with generous slack).
4. Security: traversal vectors (`../`, URL-encoded, absolute, UNC) rejected; tampered
   byte in weights → engine refuses to start.
5. Concurrency smoke: 2 threads × mixed lengths against one engine; outputs match
   solo runs bitwise (isolation proof).
6. Perf regression tripwires: TTFA and per-chunk RTF medians re-measured per protocol
   §(c) on tagged releases; regressions >25% fail CI locally.

---

## 10. Acceptance mapping (task RUNTIME-01)

| Acceptance item | Section |
|---|---|
| Public API + validation rules (charter §19 bound, blank/malformed, clamped prosody) | §2 |
| TTFA architecture + formula citing E0002/E0006 + Vocos RTF | §3 |
| Session lifecycle (load-once, thread safety, mmap option, warmup) | §4 |
| Cancellation mid-chunk semantics | §5 |
| Security (no eval/pickle, hash-verified load, traversal-proof paths) | §6 |
| CLI + minimal HTTP sketch (stdlib v0, FastAPI later) | §7 |
| File layout `caedral_voice/inference/*.py` | §8 |

## 11. Open questions

1. Inter-chunk silence default (150 ms) and per-voice overrides — needs Phase J/K listening.
2. Style-table bootstrapping changes `voices.py` validation only (schema stable).
3. Does the Phase L ONNX export fuse head into the acoustic graph (kills the separate
   `t_vocoder` term) or keep stages split for quantization granularity? Decide with
   E0002 INVESTIGATE item "compile/export pass" results.
4. Streaming-WAV writer for the 100 k-char extreme if torture shows memory pressure.
5. Safe-concurrency certification per playbook §7.3 — required before any non-localhost claim.

## References

- `experiments/CV-G1-E0002-nar-prototype/conclusion.md` — latency/RSS/thread anchors
- `experiments/CV-G1-E0006-mas-flow-decoder/conclusion.md` — flow inference cells, lite+flow params
- `experiments/CV-G1-E0001-frontend-bakeoff/` — frontend cost + char→phoneme ratio
- `benchmarks/results/decoders/vocos-20260822T150025Z*` / `vocos-20260822T150118Z*` (jsonl+meta) — Vocos decode RTF
- `docs/benchmarks/BASELINE-MEASURED.md` — TTFA/streaming lessons from baselines
- `docs/research/CPU-INFERENCE-OPTIMIZATION.md` §5, §7, appendices — serving stack, mmap, warmup, concurrency
- `configs/voices/README.md` — catalog schema + clamp rule
- `caedral_voice/model/caedral_nar.py` — `infer()` contract; `caedral_voice/text/__init__.py` — frontend facade
