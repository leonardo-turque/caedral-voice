#!/usr/bin/env bash
# Stage the exact production build context (nothing else leaks in).
set -euo pipefail
cd "$(dirname "$0")/.."

CTX=production/context
rm -rf "$CTX"
mkdir -p "$CTX/app" "$CTX/data" "$CTX/models/kokoro"

cp production/requirements-prod.txt "$CTX/app/"
cp production/server.py production/firetest_probe.py "$CTX/app/"

mkdir -p "$CTX/caedral_voice"
cp -r caedral_voice/text "$CTX/caedral_voice/text"
cp -r caedral_voice/inference "$CTX/caedral_voice/inference"
find "$CTX/caedral_voice" -name "__pycache__" -type d -exec rm -rf {} +

cp -r data/lexicons "$CTX/data/lexicons"
cp -r data/pronunciation "$CTX/data/pronunciation"
cp -r production/nltk_data "$CTX/nltk_data"

cp models/kokoro/kokoro-v1.0.onnx models/kokoro/voices-v1.0.bin "$CTX/models/kokoro/"

echo "staged $CTX:"
du -sh "$CTX"
