# CAEDRAL VOICE V1 — production image (internal model service)
# Build (from repo root):
#   bash scripts/stage_prod_context.sh
#   docker build -f production/Dockerfile -t caedral-voice:v1 production/context
#
# Reproducible: pinned base + pinned requirements; only production files.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/app/nltk_data

# Runtime shared libraries: soundfile (libsndfile), onnxruntime (libgomp).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements-prod.txt /app/requirements-prod.txt
RUN pip install --no-cache-dir -r /app/requirements-prod.txt

# Frozen engine source (package subset used at runtime)
COPY caedral_voice/ /app/caedral_voice/

# Production service + fire-test probe
COPY app/server.py app/firetest_probe.py /app/

# Frontend support data (deterministic, baked): CMUdict lexicon +
# pronunciation overrides + NLTK data for the g2p_en fallback
COPY data/lexicons/ /app/data/lexicons/
COPY data/pronunciation/ /app/data/pronunciation/
COPY nltk_data/ /app/nltk_data/

# Frozen model assets (SHA-256 recorded in V1-FREEZE-MANIFEST.json)
COPY models/kokoro/kokoro-v1.0.onnx /app/models/kokoro-v1.0.onnx
COPY models/kokoro/voices-v1.0.bin /app/models/voices-v1.0.bin

# Non-root runtime user
RUN useradd --system --create-home --uid 10001 caedral \
    && chown -R caedral:caedral /app
USER caedral

EXPOSE 8000
ENV CV_MODEL_PATH=/app/models/kokoro-v1.0.onnx \
    CV_VOICES_PATH=/app/models/voices-v1.0.bin \
    CV_THREADS=6

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status==200 else 1)"

CMD ["python", "/app/server.py"]
