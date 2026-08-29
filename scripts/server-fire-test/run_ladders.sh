#!/usr/bin/env bash
# RPM ladder + sustained windows + concurrency bursts on BOTH candidate
# production envelopes. Phase A at current 2048MB/2vCPU; phase B resets the
# service to the conservative 1400MB/1vCPU envelope and repeats.
set -u
cd /root/caedral-voice-bench
R=/opt/caedral/services/voice-api/releases/v1
SECRET=$(sed -n "s/^CV_INTERNAL_SECRET=//p" $R/.env)

# ---- PHASE A: ladder @ 2048MB / 2 vCPU --------------------------------------
echo '{"summary":{"gen_p50_s":37.23}}' > probe-p3.json
bash sampler.sh 2700 results/p3l-sampler.log &
SAM=$!
python3 loadgen_server.py --url http://127.0.0.1:8600 \
  --probe probe-p3.json --label p3-2048mb-2vcpu --out results --window 300 \
  --secret "$SECRET" \
  > results/p3-ladder.log 2>&1
kill $SAM 2>/dev/null
echo done > results/P3L_DONE.marker

# ---- reset to conservative envelope: 1400MB / 1 vCPU ------------------------
cd $R
sed -i 's/mem_limit: 2048000000/mem_limit: 1400000000/; s/memswap_limit: 2048000000/memswap_limit: 1400000000/; s/cpus: 2.0/cpus: 1.0/' docker-compose.yml
docker compose up -d
sleep 5
for i in $(seq 1 80); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8600/readyz)
  [ "$code" = "200" ] && break
  sleep 3
done
[ "$code" = "200" ] || { echo READY_TIMEOUT > results/LADDER_FAIL.marker; exit 1; }
cd /root/caedral-voice-bench

# ---- PHASE B: ladder @ 1400MB / 1 vCPU --------------------------------------
echo '{"summary":{"gen_p50_s":86.77}}' > probe-p1.json
bash sampler.sh 3300 results/p1l-sampler.log &
SAM=$!
python3 loadgen_server.py --url http://127.0.0.1:8600 \
  --probe probe-p1.json --label p1-1400mb-1vcpu --out results --window 300 \
  --secret "$SECRET" \
  > results/p1-ladder.log 2>&1
kill $SAM 2>/dev/null
echo done > results/P1L_DONE.marker
echo LADDERS_ALL_DONE
