#!/usr/bin/env bash
# P2/P3 fire-test profiles: recreate ONLY the voice container with new limits,
# wait for readiness, run 10 sequential canonical generations each.
set -u
cd /root/caedral-voice-bench
R=/opt/caedral/services/voice-api/releases/v1
SECRET=$(sed -n "s/^CV_INTERNAL_SECRET=//p" $R/.env)

wait_ready() {
  for i in $(seq 1 80); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8600/readyz)
    [ "$code" = "200" ] && return 0
    sleep 3
  done
  return 1
}

# ---- P2: 2048 MB / 1 vCPU ---------------------------------------------------
cd $R
sed -i 's/mem_limit: 1400000000/mem_limit: 2048000000/; s/memswap_limit: 1400000000/memswap_limit: 2048000000/' docker-compose.yml
docker compose up -d
sleep 5
wait_ready || { echo "P2_READY_TIMEOUT" > results/P23_FAIL.marker; exit 1; }
cd /root/caedral-voice-bench
bash sampler.sh 1500 results/p2-sampler.log &
SAM=$!
python3 seq_bench.py --url http://127.0.0.1:8600 --label p2-2048mb-1vcpu --n 10 --secret "$SECRET" --out results > results/p2-seq.log 2>&1
kill $SAM 2>/dev/null
echo done > results/P2_DONE.marker

# ---- P3: 2048 MB / 2 vCPU ---------------------------------------------------
cd $R
sed -i 's/cpus: 1.0/cpus: 2.0/' docker-compose.yml
docker compose up -d
sleep 5
wait_ready || { echo "P3_READY_TIMEOUT" > results/P23_FAIL.marker; exit 1; }
cd /root/caedral-voice-bench
bash sampler.sh 1500 results/p3-sampler.log &
SAM=$!
python3 seq_bench.py --url http://127.0.0.1:8600 --label p3-2048mb-2vcpu --n 10 --secret "$SECRET" --out results > results/p3-seq.log 2>&1
kill $SAM 2>/dev/null
echo done > results/P3_DONE.marker
echo "P23_ALL_DONE"
