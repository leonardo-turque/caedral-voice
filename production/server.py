"""CAEDRAL VOICE V1 — internal production model service.

Internal model service behind the Caedral gateway: the gateway owns customer
auth, billing, credits, public rate limits and accounting; this service owns
validation, queueing, speech generation and health.

Contract (V1):
  POST /v1/audio/speech   {model, voice, input, speed, response_format}
  GET  /healthz           liveness
  GET  /readyz            readiness (model loaded + warmed)
  GET  /metrics           internal JSON observability

Frozen-engine usage: ONE CaedralVoiceKokoroEngine is built at startup and
shared by all four voices. All synthesis funnels through a single worker
thread (one ORT session, strict serialization) so HTTP concurrency only
affects queue wait, never parallel session execution.

Long-input note: engine.synth() raises in its post-render telemetry for
inputs above 510 phonemes (frozen limitation); the service therefore uses
the engine's own public API pair phonemize() + infer_from_phonemes(), which
is the identical frontend + fused graph with wrapper-side batching.
"""
from __future__ import annotations

import io
import logging
import os
import queue
import threading
import time
import wave
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
# NOTE: fastapi objects are imported at module level on purpose —
# `from __future__ import annotations` makes endpoint annotations strings and
# FastAPI resolves them against module globals; a function-local import of
# `Request` makes FastAPI misread it as a query parameter.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from caedral_voice.inference.kokoro_engine import VOICE_PROFILES  # noqa: E402

MODEL_PATH = os.environ.get("CV_MODEL_PATH", "/app/models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("CV_VOICES_PATH", "/app/models/voices-v1.0.bin")
THREADS = int(os.environ.get("CV_THREADS", "6"))          # frozen production value
MAX_INPUT_CHARS = int(os.environ.get("CV_MAX_INPUT_CHARS", "2000"))
QUEUE_SIZE = int(os.environ.get("CV_QUEUE_SIZE", "8"))
GEN_TIMEOUT_S = float(os.environ.get("CV_GEN_TIMEOUT_S", "300"))
# Internal-service auth (matches main-platform MODEL_INFERENCE_SECRET
# convention): when CV_INTERNAL_SECRET is set, every /v1/* request must carry
# the matching X-Internal-Secret header. The gateway owns customer auth.
INTERNAL_SECRET = os.environ.get("CV_INTERNAL_SECRET", "").strip()
SUPPORTED_MODEL = "caedral-voice-1"
VALID_VOICES = ("caedral-f1", "caedral-f2", "caedral-m1", "caedral-m2")
SPEED_MIN, SPEED_MAX = 0.25, 2.0
FORMATS = ("wav",)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("caedral-voice")

# Latin-script gate: V1 is ENGLISH ONLY. We do not pretend to detect language;
# we reject text whose letters fall outside the Latin repertoire so obviously
# unsupported scripts fail fast with a deterministic error.
_LATIN_OK = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?;:'\"()[]-–—&%$#@*/+=<>~`^\n\r\t"
    "àáâäèéêëìíîïòóôöùúûüñçæœßüöäÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÑÇÆŒ'’‘“”…°"
)


def _err(code: str, message: str):
    return JSONResponse(
        status_code=400,
        content={"error": {"code": code, "message": message}},
    )


class SpeechService:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self._jobs: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._counters = {"completed": 0, "rejected_400": 0, "rejected_429": 0,
                          "failed": 0, "timeouts": 0}
        self._last_rtf = None
        self._started_at = time.time()
        self._engine = None
        self._worker: threading.Thread | None = None

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        t0 = time.perf_counter()
        from caedral_voice.inference.kokoro_engine import CaedralVoiceKokoroEngine
        self._engine = CaedralVoiceKokoroEngine(
            model_path=MODEL_PATH, voices_path=VOICES_PATH, threads=THREADS)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        self._engine.warmup()
        warm_s = time.perf_counter() - t1
        self._worker = threading.Thread(target=self._run_worker,
                                        name="synth-worker", daemon=True)
        self._worker.start()
        self.ready.set()
        log.info("service ready: load %.2fs warmup %.2fs threads=%d",
                 load_s, warm_s, THREADS)

    def _run_worker(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                break
            phonemes, voice_name, speed, fut = item
            try:
                t0 = time.perf_counter()
                audio = self._engine.infer_from_phonemes(
                    phonemes, voice_name, speed=speed)
                wall = time.perf_counter() - t0
                audio_s = max(len(audio) / 24000.0, 1e-9)
                self._last_rtf = wall / audio_s
                fut.set_result((audio, wall, audio_s))
                self._counters["completed"] += 1
            except Exception as exc:  # noqa: BLE001 — deterministic 5xx body
                log.exception("synthesis failed")
                fut.set_exception(exc)
                self._counters["failed"] += 1

    # --- synthesis -----------------------------------------------------------
    def synthesize(self, text: str, voice_id: str, speed: float):
        """Queue one job; returns (audio_f32, wall_s, audio_s). Raises QueueFull."""
        profile = VOICE_PROFILES.get(voice_id)
        if profile is None:  # validated earlier; belt-and-braces
            raise KeyError(voice_id)
        phonemes, _skipped = self._engine.phonemize(text)
        fut: Future = Future()
        try:
            self._jobs.put_nowait((phonemes, profile["kokoro_voice"],
                                   speed, fut))
        except queue.Full:
            raise
        audio, wall, audio_s = fut.result(timeout=GEN_TIMEOUT_S)
        return audio, wall, audio_s


SERVICE = SpeechService()


def wav_bytes(audio_f32: np.ndarray) -> bytes:
    pcm = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def create_app():
    app = FastAPI(title="Caedral Voice V1 (internal)",
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.on_event("startup")
    def _startup() -> None:
        threading.Thread(target=SERVICE.start, name="startup",
                         daemon=True).start()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Response:
        if SERVICE.ready.is_set():
            return JSONResponse({"ready": True})
        return JSONResponse({"ready": False}, status_code=503)

    @app.get("/metrics")
    def metrics() -> dict:
        qdepth = SERVICE._jobs.qsize()
        rss_mb = None
        try:
            import psutil
            rss_mb = psutil.Process().memory_info().rss // 1e6
        except Exception:
            pass
        return {"uptime_s": round(time.time() - SERVICE._started_at, 1),
                "ready": SERVICE.ready.is_set(),
                "queue_depth": qdepth,
                "queue_capacity": QUEUE_SIZE,
                "last_rtf": SERVICE._last_rtf,
                "rss_mb": rss_mb,
                **SERVICE._counters}

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        if INTERNAL_SECRET:
            provided = request.headers.get("X-Internal-Secret", "")
            import secrets as _secrets
            if not _secrets.compare_digest(provided, INTERNAL_SECRET):
                SERVICE._counters["rejected_400"] += 1
                return JSONResponse(
                    {"error": {"code": "UNAUTHORIZED",
                               "message": "missing or invalid "
                                          "X-Internal-Secret"}},
                    status_code=401)
        try:
            body = await request.json()
        except Exception:
            SERVICE._counters["rejected_400"] += 1
            return _err("MALFORMED_JSON", "request body must be valid JSON")

        if not isinstance(body, dict):
            SERVICE._counters["rejected_400"] += 1
            return _err("MALFORMED_JSON", "request body must be a JSON object")

        model = body.get("model", SUPPORTED_MODEL)
        voice = body.get("voice")
        text = body.get("input")
        speed = body.get("speed", 1.0)
        fmt = body.get("response_format", "wav")

        if model != SUPPORTED_MODEL:
            SERVICE._counters["rejected_400"] += 1
            return _err("UNKNOWN_MODEL",
                        f"unknown model {model!r}; supported: {SUPPORTED_MODEL}")
        if voice not in VALID_VOICES:
            SERVICE._counters["rejected_400"] += 1
            return _err("UNKNOWN_VOICE",
                        f"unknown voice {voice!r}; supported: {list(VALID_VOICES)}")
        if not isinstance(text, str) or not text.strip():
            SERVICE._counters["rejected_400"] += 1
            return _err("EMPTY_INPUT", "'input' must be a non-empty string")
        if len(text) > MAX_INPUT_CHARS:
            SERVICE._counters["rejected_400"] += 1
            return _err("INPUT_TOO_LONG",
                        f"input {len(text)} chars exceeds limit {MAX_INPUT_CHARS}")
        bad = sorted({ch for ch in text if ch.isalpha()
                      and ch not in _LATIN_OK})
        if bad:
            SERVICE._counters["rejected_400"] += 1
            return _err("UNSUPPORTED_LANGUAGE",
                        "V1 supports English (Latin script) only; "
                        f"unexpected characters: {''.join(bad[:10])}")
        if not isinstance(speed, (int, float)) or isinstance(speed, bool) \
                or not SPEED_MIN <= float(speed) <= SPEED_MAX:
            SERVICE._counters["rejected_400"] += 1
            return _err("INVALID_SPEED",
                        f"speed must be a number in [{SPEED_MIN}, {SPEED_MAX}]")
        if fmt not in FORMATS:
            SERVICE._counters["rejected_400"] += 1
            return _err("UNSUPPORTED_FORMAT",
                        f"response_format {fmt!r} unsupported; "
                        f"supported: {list(FORMATS)}")

        if not SERVICE.ready.is_set():
            return JSONResponse({"error": {"code": "NOT_READY",
                                           "message": "model warming up"}},
                                status_code=503)
        try:
            audio, wall, audio_s = SERVICE.synthesize(
                text.strip(), voice, float(speed))
        except queue.Full:
            SERVICE._counters["rejected_429"] += 1
            return JSONResponse(
                {"error": {"code": "OVERLOADED",
                           "message": "queue full; retry later"}},
                status_code=429, headers={"Retry-After": "5"})
        except FuturesTimeoutError:
            SERVICE._counters["timeouts"] += 1
            return JSONResponse({"error": {"code": "GENERATION_TIMEOUT",
                                           "message": "generation timed out"}},
                                status_code=504)
        except Exception:
            SERVICE._counters["failed"] += 1
            log.exception("speech request failed")  # server-side visibility;
            # client still gets only the deterministic body below
            return JSONResponse({"error": {"code": "SYNTHESIS_FAILED",
                                           "message": "internal error"}},
                                status_code=500)

        data = await __import__("asyncio").get_running_loop().run_in_executor(
            None, wav_bytes, audio)
        log.info("speech ok voice=%s chars=%d wall=%.2fs audio=%.2fs rtf=%.3f",
                 voice, len(text), wall, audio_s, wall / max(audio_s, 1e-9))
        return Response(content=data, media_type="audio/wav",
                        headers={"X-Caedral-Audio-Seconds": f"{audio_s:.2f}",
                                 "X-Caedral-Wall-Seconds": f"{wall:.2f}",
                                 "X-Caedral-RTF": f"{wall / max(audio_s, 1e-9):.3f}"})

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("CV_PORT", "8000")),
                workers=1, log_level="info")


if __name__ == "__main__":
    main()
