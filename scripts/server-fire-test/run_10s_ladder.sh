#!/usr/bin/env bash
# 10s-audio RPM ladder + concurrency bursts at the deployed 4GB/4vCPU cap.
set -u
cd /root/caedral-voice-bench
R=/opt/caedral/services/voice-api/releases/v1
SECRET=$(sed -n "s/^CV_INTERNAL_SECRET=//p" $R/.env)

echo '{"summary":{"gen_p50_s":6.285}}' > probe-10s-4vcpu.json
bash sampler.sh 2400 results/f4l-sampler.log &
SAM=$!
python3 loadgen_server.py --url http://127.0.0.1:8600 \
  --probe probe-10s-4vcpu.json --label f4vcpu-10s --out results --window 300 \
  --body canonical-10s.txt --secret "$SECRET" > results/f4-ladder.log 2>&1
kill $SAM 2>/dev/null
echo done > results/F4L_DONE.marker
