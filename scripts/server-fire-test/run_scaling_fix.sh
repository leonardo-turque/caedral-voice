#!/usr/bin/env bash
# Corrected CPU-scaling runs: 10s input, 4 GB RAM fixed, cpus 1 -> 2 -> restore 4.
set -u
BENCH=/root/caedral-voice-bench
R=/opt/caedral/services/voice-api/releases/v1
SECRET=$(sed -n "s/^CV_INTERNAL_SECRET=//p" $R/.env)
cd "$BENCH"

wait_ready() {
  for i in $(seq 1 80); do
    [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 http://127.0.0.1:8600/readyz)" = "200" ] && return 0
    sleep 3
  done
  return 1
}

set_profile() {
  ( cd $R && sed -i "s/cpus: [0-9.]*/cpus: $1/" docker-compose.yml && docker compose up -d >/dev/null ) || return 1
  sleep 5
  wait_ready
}

# ---- 1 vCPU x10 ----
set_profile 1.0 || { echo FAIL > $BENCH/results/SCALING_FAIL.marker; exit 1; }
python3 seq_bench.py --label f1cpu-4000mb-1vcpu --n 10 --warmup 2 \
  --input canonical-10s.txt --secret "$SECRET" --out results > results/f1cpu-seq.log 2>&1
echo done > results/F1C_DONE.marker

# ---- 2 vCPU x10 ----
set_profile 2.0 || { echo FAIL > $BENCH/results/SCALING_FAIL.marker; exit 1; }
python3 seq_bench.py --label f2cpu-4000mb-2vcpu --n 10 --warmup 2 \
  --input canonical-10s.txt --secret "$SECRET" --out results > results/f2cpu-seq.log 2>&1
echo done > results/F2C_DONE.marker

# ---- restore 4 vCPU ----
set_profile 4.0 || { echo FAIL > $BENCH/results/SCALING_FAIL.marker; exit 1; }
sleep 3
echo "restored_4vcpu metrics=$(curl -s http://127.0.0.1:8600/metrics)" >> $BENCH/results/baseline-4vcpu.log
echo ALL_SEQ_DONE > $BENCH/results/FSEQ_ALL.marker
