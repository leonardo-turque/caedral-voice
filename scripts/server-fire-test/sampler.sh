#!/usr/bin/env bash
# Resource sampler — run ON the server host during heavy phases.
# usage: sampler.sh <duration_seconds> <outfile>
set -u
DUR=${1:-300}
OUT=${2:-sampler.log}
END=$(( $(date +%s) + DUR ))
: > "$OUT"
while [ "$(date +%s)" -lt "$END" ]; do
  ts=$(date +%s)
  mem=$(docker exec caedral-voice-api sh -c 'cat /sys/fs/cgroup/memory.current' 2>/dev/null)
  peak=$(docker exec caedral-voice-api sh -c 'cat /sys/fs/cgroup/memory.peak' 2>/dev/null)
  cstat=$(docker exec caedral-voice-api sh -c 'tr "\n" " " < /sys/fs/cgroup/cpu.stat' 2>/dev/null)
  cpuq=$(docker exec caedral-voice-api sh -c 'cat /sys/fs/cgroup/cpu.max' 2>/dev/null)
  load=$(cut -d" " -f1-3 /proc/loadavg | tr " " ",")
  steal=$(awk '/^cpu /{print $9}' /proc/stat)
  echo "$ts mem=${mem:-NA} peak=${peak:-NA} cpumax=${cpuq:-NA} ${cstat} host_load=(${load}) host_steal_jiffies=${steal}" >> "$OUT"
  sleep 10
done
echo "SAMPLER_DONE duration=${DUR}s" >> "$OUT"
