# PHASE O PLAN — Resource-Constraint Test Lab ("Torture Lab") — WINDOWS-FIRST

**Status:** PLANNED (design frozen for review; implementation is Phase O execution)
**Date:** 2026-08-22 · **Owner:** Phase O agent · **Local analysis only** (no downloads beyond already-present tooling)
**Authorities:** Program charter §18 (torture lab) & §19 (deployment profiles × scenarios). The charter text lives outside this repo; the profile and scenario enumerations below are the authoritative §18/§19 lists as tasked for Phase O.
**Reused inputs:** `benchmarks/machine-baseline.json` (§2), `docs/MASTER-PLAN.md` §4–§5, `benchmarks/harness/measure.py` (measurement primitives + Win32 patterns), `docs/benchmarks/BASELINE-MEASURED.md` (expected magnitudes), `benchmarks/corpus/eval-corpus-v1.0.json` (payload items).
**Evidence labels:** every number produced by this lab carries MEASURED / ESTIMATED / REPORTED BY THIRD PARTY per charter §9. Predictions written here are labeled HYPOTHESIS until the lab measures them.

---

## 1. Purpose

Prove — or honestly measure the minimum of — how the Caedral stack behaves under the
charter deployment envelopes (§19 profiles) across the §18 adversarial scenarios.
Targets are *measured against*, never assumed achieved (charter §5 rule). A cell that
cannot meet its bar must still produce a bounded, typed, recorded failure
(**clean fail**) or it counts as a HARD-FAIL against the runtime, not just the target.

Non-goals: quality scoring (Phase C stack covers that); GPU; training workloads;
multi-host distribution.

---

## 2. Machine ground truth (REUSED from `benchmarks/machine-baseline.json`)

| Item | Value |
|---|---|
| CPU | i7-13620H, hybrid 6P+4E = 10 phys cores / **16 logical**, AVX2+FMA3, no AVX-512 |
| RAM | 25,414,197,245 B installed; ~6 GB free at baseline inspection; swap 49,208,274,944 B |
| OS | Windows 11 build 26200 (Home), AMD64 |
| Python | 3.13.15 CPython, repo `.venv` |
| Also present [REPORTED] | Docker Desktop + WSL2 (per MASTER-PLAN §4), no compilers |

Consequences used throughout this plan:
- **16 logical processors ⇒ 1 vCPU ≡ CpuRate 625** (percent×100 units, see §3.1);
  2 vCPU ≡ 1250; 4 vCPU ≡ 2500.
- Hybrid P/E cores ⇒ affinity masks must be explicit and recorded (gotcha G4).
- Huge pagefile ⇒ host-level OOM is nearly unreachable; deterministic boundaries come
  from Job Object limits, not host memory exhaustion (gotcha G3).
- Laptop thermals/power state affect every number ⇒ AC power + cooldown gaps mandatory.

---

## 3. Constraint mechanisms compared (Acceptance item 1)

Three candidate mechanisms for emulating "N vCPU / M RAM" on this Windows-first box.

### 3.1 Candidates

**A. Windows Job Objects via python `ctypes`** (`CreateJobObjectW`,
`SetInformationJobObject`, `AssignProcessToJobObject`, `QueryInformationJobObject`)
- Memory: `JOB_OBJECT_LIMIT_JOB_MEMORY` (job-wide **commit** limit) +
  `JOB_OBJECT_LIMIT_PROCESS_MEMORY` (per-process commit). Exceeding it makes
  allocations *fail* (typed `MemoryError`/allocator-null) instead of thrashing swap.
- CPU: `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION` hard cap, units percent×100
  (range 1–10000 over all logical processors), sub-second enforcement window.
  Plus `LimitAffinity` for spatial pinning to exactly *N* logical processors.
- Accounting: query-back gives `PeakJobMemoryUsed` (peak job commit) and job-wide
  `PeakWorkingSetSize`; combined with psutil RSS/USS samplers.

**B. Docker Desktop / WSL2 cgroups v2** (`--memory`, `--cpus`, cgroup `memory.max`)
- Real Linux kernel semantics: `memory.max` counts anon + page cache, reclaims
  page cache before OOM-kill. This is what a real Linux VPS/container production
  target enforces, so it validates deployment-realistic behavior.

**C. Process priority + affinity masks only** (psutil `nice()`/priority class +
`cpu_affinity()`) — CPU shaping with **no memory enforcement whatsoever**.

### 3.2 Comparison

| Criterion | A. Job Objects | B. WSL2/Docker cgroups | C. Priority+affinity |
|---|---|---|---|
| Memory enforcement | Yes — hard commit limit, deterministic allocation-failure boundary | Yes — cgroup v2 `memory.max`, reclaim→OOM-kill semantics | **None** |
| CPU enforcement | Yes — affinity (spatial, N LPs) and/or CPU-rate % (temporal) | Yes — quota/bandwidth, real Linux scheduler | Partial — affinity yes; priority only re-orders, never reserves |
| Virtualization noise | None (native) | WSL2 VM layer: vmmem dynamic memory, clock-source shift, nested-virt overhead | None |
| Matches production target? | Matches Windows serving reality; commit-limit ≠ Linux OOM semantics | Closest to Linux VPS production semantics | Neither |
| Determinism/reproducibility | High (limits enforced by host kernel on the same hardware being measured) | Medium-low (one shared VM hosts all tests → cross-run interference; Desktop daemon flakiness risk) | High |
| Fragility on this box | Low — pure ctypes + stdlib subprocess | Higher — VM boot, `.wslconfig` VM cap nests above container limits, daemon restarts | Lowest |
| Clean-fail observability | Typed alloc failures + `PeakJobMemoryUsed` query-back | OOM-kill events (exit 137-class) + dmesg/cgroup.events | n/a (no boundary to fail at) |
| Effort | Moderate (ctypes structs, one-time) | Moderate-high (WSL config + Docker orchestration) | Trivial |

### 3.3 Decision

**PRIMARY: Mechanism A — Windows Job Objects via ctypes.**
Justification: (1) Windows-first honesty — measurements come from the host kernel on
the same silicon that will serve, zero VM layer shifting timers/schedulers;
(2) it is the only native option that enforces BOTH axes deterministically;
(3) commit-limit overflow yields typed allocation failures — exactly the observable
the clean-fail criteria need (a cgroup OOM-kill is a blunter instrument);
(4) no dependency on Docker Desktop daemon stability, which MASTER-PLAN §4 itself
flags as conditional ("if stable").

**SECONDARY / cross-validation subset: Mechanism B — WSL2/Docker cgroups.** Not for
the full 60-cell matrix (instability + comparability costs) but for a confirmation
subset (all 10 scenarios at profiles C-class and D/E-class, i.e. 1 vCPU rows) to
verify that verdicts transfer to real Linux cgroup semantics before any production
claim. Results are tagged with `constraint_mechanism` and are never numerically
compared across mechanisms (different kernel accounting).

**FALLBACK: Mechanism C — priority + affinity (psutil-only).** Activates if ctypes
Job Object control proves unstable (struct/API breakage). Covers CPU-side scenarios
only; memory scenarios (S1 partial, S9) are then marked NOT ENFORCEABLE — those cells
report resource telemetry without a pass/fail boundary and the gap is documented per
charter honesty rules.

Escalation path: run the §10.1 shakedown on Mechanism A first; only if a blocking
defect appears does the matrix run on C, with B re-attempted for the validation subset.

---

## 4. Deployment profiles (charter §19 enumeration, byte-exact)

Caps interpreted at the **strictest reading**: SI decimal MB/GB (256 MB = 256×10⁶ B),
which is tighter than a binary-MiB reading of the same names.

| Profile | vCPU | RAM cap (bytes) | Charter kin | Affinity strategy (canonical) | Threads (intra-op) |
|---|---|---|---|---|---|
| P1 | 1 | 256,000,000 | §5-E extreme research | 1 P-core logical processor | 1 |
| P2 | 1 | 500,000,000 | §5-D torture | 1 P-core LP | 1 |
| P3 | 1 | 1,000,000,000 | §5-C ultra-low-cost | 1 P-core LP | 1 |
| P4 | 2 | 1,000,000,000 | C/B hybrid row | 2 LPs, distinct physical P cores (no HT siblings) | 2 |
| P5 | 2 | 2,000,000,000 | §5-B economical prod | 2 distinct P-core LPs | 2 |
| P6 | 4 | 4,000,000,000 | ⊂ §5-A flagship | 4 distinct P-core LPs | 4 |

- Exact LP indices come from a topology probe (GetLogicalProcessorInformationEx,
  sketched in §7); if P/E identification fails, fall back to LP `{0}`/`{0,2}`/`{0,2,4,6}`
  and label the mask ESTIMATED. Shakedown runs each 1-vCPU profile once pinned to an
  E-core as well, to quantify hybrid spread once, then fixes the canonical mask.
- Expected-fit context from MEASURED baselines (not requirements): torch-runtime floor
  ~215 MB (E0002), Caedral tiny RSS 235 MB, lite ~334 MB, Kokoro ~1.25 GB. HYPOTHESIS:
  P1/P2 admit tiny-class models only; P3 borderline for lite; Kokoro-class needs P5+.

---

## 5. Verdict taxonomy and global invariants

Every matrix cell ends in exactly one verdict:

- **PASS** — functional output valid AND resource bars met AND latency bar met.
- **CLEAN-FAIL** — a bar was not met, but the failure satisfied ALL clean-fail
  requirements below. Acceptable outcome; publishes the measured minimum (charter §9.5).
- **HARD-FAIL** — anything else. Counts against the runtime even if the profile target
  was unrealistic: hang past watchdog, access violation/segfault, silent bad output,
  cap evasion, leaked handles/processes/locks, host destabilization.

**Clean-fail requirements (global, all scenarios):**
1. Bounded duration — terminated by or before its watchdog (§6 per-scenario values).
2. Explicit typed failure surfaced in the result record (exception class + message +
   stage), or a job-limit kill recorded with `PeakJobMemoryUsed` attached.
3. No host impact: harness responsive; constrained process tree fully reaped
   (verified via `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` + psutil scan for stragglers).
4. No cap evasion: either usage stayed under the enforced limit, or evasion
   (e.g., mmap'd weights escaping commit accounting — gotcha G1) was detected and
   reported in the same result record.
5. Any audio produced during a failing run is quarantined and marked INVALID — it
   never enters quality scoring.
6. Result row persisted with full provenance (git commit, mask, threads, mechanism,
   timestamp, corpus hash) regardless of verdict.

**Shared gates:** audio sanity gate = decodes; duration within ±5% of expectation;
RMS > silence floor (−60 dBFS); no NaN/Inf samples; clipped-sample fraction < 1%.
Watchdog default where not stated: 600 s.

---

## 6. Test matrix — 6 profiles × 10 scenarios (Acceptance item 2)

Legend: ✔ full criteria apply; (n) notes. All 60 cells execute; no cell is skipped —
an inapplicable-looking cell is itself a measurement (its predicted outcome is
HYPOTHESIS until run).

| Scenario \ Profile | P1 1/256M | P2 1/500M | P3 1/1G | P4 2/1G | P5 2/2G | P6 4/4G |
|---|---|---|---|---|---|---|
| S1 cold boot | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S2 huge paragraph | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S3 100 consecutive gens | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S4 100k-char protection | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S5 malformed/blank text | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S6 cancel mid-synthesis | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S7 restart | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S8 corrupted model file | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S9 memory pressure | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| S10 concurrency overload | ✔ (ramp capped by cap) | ✔ | ✔ | ✔ | ✔ | ✔ (full 1→16) |

Per-profile expectations are derived from §4 fit context, not baked into criteria:
identical criteria apply everywhere; the *verdict* differs.

### S1 Cold boot
Fresh process under profile constraints → load model → synthesize one short corpus
item (conv_001-class). Two sub-runs: OS-cache-warm and cache-cooled (standby-list
flush via brief external memory hog — Windows cannot drop caches directly, gotcha G8).
- PASS: load completes; TTFA_cold ≤ 3× TTFA_warm + 2 s; peak job commit ≤ cap;
  audio gate green; process exits 0.
- CLEAN-FAIL: typed load/OOM error within watchdog (180 s), limits respected.
- HARD-FAIL: hang, AV, cap evasion undetected, garbage output reported as success.

### S2 Huge paragraph
Inputs: frozen-corpus paragraph item (para_001) plus synthetic 5k / 10k / 25k-char
paragraphs assembled from shuffled corpus sentences. Exercises caller-side chunking
(Baseline lesson #1: mandatory for long-form).
- PASS: completes within 600 s; peak commit ≤ cap; concatenated audio duration within
  ±10% of Σ per-chunk expectations; every chunk passes audio gate.
- CLEAN-FAIL: typed rejection ("input exceeds policy") or typed OOM at a documented
  input size — the largest passing size becomes the published minimum.
- HARD-FAIL: crash/hang (KittenTTS-style unbounded shape growth is the historical
  cautionary example), NaN output, silent truncation without policy.

### S3 100 consecutive generations
Same process, 100 back-to-back generations of mixed corpus items.
- Leak gates (commit-based, gotcha G3): linear slope of commit (psutil `vms`) over
  gens 51–100 ≤ 0.5 MB/gen AND absolute growth ≤ 50 MB; post-gc RSS returns to within
  10% of steady state.
- Drift gate: median RTF(last 20) / median RTF(first 20) ∈ [0.8, 1.25].
- PASS: all gates + every output passes audio gate + watchdog 2400 s respected.
- CLEAN-FAIL: leak/drift gate trips with clean outputs throughout — publish measured
  sustainable count (largest N with all gates green, N ≥ 1).
- HARD-FAIL: crash, hang, invalid output presented as success.

### S4 100k-char protection
Feed exactly 100,000 chars (and a 100,001-char probe) to the pipeline entrypoint.
Frontend contract: reject BEFORE tensor allocation.
- PASS: typed rejection (or explicit logged truncate-policy) within 1 s wall;
  peak commit delta < 10 MB; exit handled cleanly.
- CLEAN-FAIL: rejection slower than 1 s but < watchdog (60 s), still bounded+explicit.
- HARD-FAIL: processing proceeds silently, proportional allocation occurs, OOM-crash
  without typed surface, or truncated audio returned without policy notice.

### S5 Malformed/blank text
Suite: `""`; `" "`; `"\n\t"`; `"\x00"`; `"a"×10,000` (single word); U+FFFD ×1000;
emoji-only `"😀"`×50; bidi-mixed RTL+LTR; digits-only 5k; punctuation-only; single
char; whitespace-heavy 5k; lone surrogates `"\ud800"`.
- PASS: every case returns typed rejection or sane audio; whole suite ≤ 120 s.
- CLEAN-FAIL: individual case(s) rejected late but explicitly; per-case verdicts
  recorded individually (suite verdict = worst case).
- HARD-FAIL: any crash, hang, or NaN/garbage audio from any case.

### S6 Cancel mid-synthesis
Start long synthesis; issue cancellation at 10% and separately at 50% progress.
Two modes: cooperative flag (checked between pipeline stages) and forced
(process-tree terminate).
- PASS: ack ≤ 500 ms (cooperative; ≤ 2 s hard cap if stage-boundary deferral) /
  ack ≤ 2 s (forced); worker CPU < 5% within 1 s of ack; commit ≤ baseline+10%
  within 5 s; the immediately-following generation succeeds (state reset proven).
- CLEAN-FAIL: ack within watchdog (300 s) with full cleanup verified afterwards.
- HARD-FAIL: orphaned compute threads spinning, memory not released, next generation
  corrupted, deadlock.

### S7 Restart
10 cycles: 5× graceful shutdown → relaunch; 5× hard TerminateProcess mid-generation →
relaunch; final 11th cycle relaunched while S9-style pressure is active.
- Per-cycle checks: model loads (S1 gates); model file deletable/renamable after
  shutdown (Windows file-lock check, gotcha G7); no zombie children; restarted RSS
  equals fresh cold-boot RSS within 10%; queued items either completed or explicitly
  reported lost — never silently dropped.
- PASS: all 11 cycles green. CLEAN-FAIL: some cycle recovers only via manual
  intervention step that is itself scripted+documented (e.g., lock-release retry with
  backoff). HARD-FAIL: unrecoverable state, permanent lock, handle-count growth
  across cycles.

### S8 Corrupted model file
Mutations applied to copies: M1 truncate 50%; M2 truncate last 4 KB; M3 zero first
64 B (header); M4 bit-flip random 1 KiB mid-payload; M5 swap two 1 KiB regions;
M6 zero-byte file; M7 valid WAV renamed to model extension.
- PASS: loader detects every mutation via checksum mismatch or structured error,
  within 5 s, with typed error naming file + mutation id; NEVER produces audio.
  (If the current loader lacks pre-load hashing, the actionable outcome is to add it.)
- CLEAN-FAIL: detection slower than 5 s but bounded and typed.
- HARD-FAIL: audio emitted from any mutated file, or untyped crash.

### S9 Memory pressure
External hog process (§7 `pressure.py`) commits+touches pages to reach, from live
`available_bytes`: L1 = consume 50% of available; L2 = leave 1024 MB free;
L3 = leave 512 MB free. Abort a level if available < 2 GB at start (laptop safety).
Workload: standard generation loop under profile constraints.
- Host invariant (all levels): harness + OS stay responsive; no other-app disruption
  beyond expected paging; hog fully released afterwards.
- PASS: workload completes within SLO at all three levels.
- CLEAN-FAIL: typed MemoryError/job-limit kill at a documented level; no partial
  artifacts left behind (checkpoint/temp scan).
- HARD-FAIL: hang, AV, host destabilization, leftover corrupt temp state.
- Note (G3): with 49 GB swap, true host OOM is out of reach — job limits are the real
  boundary; levels calibrate against available RAM, not total.

### S10 Concurrency overload
Protocol and formulas in §8. Ramp k = 1→16 concurrent streams inside one constrained
job per profile.
- PASS: k* (saturation point) ≥ predicted N_safe(p) (§8) and all invariants hold up
  to k*.
- CLEAN-FAIL: beyond saturation, streams degrade gracefully — queueing, typed
  backpressure errors, bounded latencies; no invariant broken at any k up to 16.
- HARD-FAIL: any hang, corruption, cross-stream interference (audio from stream i
  appearing in stream j), unbounded queue growth, or invariant breach.

---

## 7. Harness layout (Acceptance item 3) — `benchmarks/torture/*.py`

All sketches below are DESIGN SKETCHES (implementation happens in Phase O; numeric
Win32 constants to be verified against headers during the shakedown). Reuses
`benchmarks/harness/measure.py` primitives (`PerfResult`, `win_process_peaks`, `rtf`).

```
benchmarks/torture/
  __init__.py
  jobobject.py      # ctypes Job Object wrapper (primary mechanism)
  topology.py       # P/E-core probe -> canonical affinity masks
  profiles.py       # §4 table as data
  payloads.py       # corpus loaders + synthetic/adversarial text suites
  monitor.py        # commit/RSS/USS sampler + PeakJobMemoryUsed query-back
  pressure.py       # external memory hog for S9/S1-cooling
  scenarios.py      # S1..S10 implementations, each -> ScenarioResult
  ramp.py           # §8 concurrency ramp driver
  report.py         # JSONL -> markdown rollup with evidence labels
  run_matrix.py     # CLI: --profiles --scenarios --mechanism --experiment-id
```

### jobobject.py (core sketch)

```python
"""Job Object constraint wrapper. Sketch — verify constants vs winbase.h."""
import ctypes, subprocess
from ctypes import wintypes

JobObjectExtendedLimitInformation   = 9
JobObjectCpuRateControlInformation  = 20   # verify in shakedown

JOB_OBJECT_LIMIT_AFFINITY        = 0x0008  # verify
JOB_OBJECT_LIMIT_PROCESS_MEMORY  = 0x0100  # verify
JOB_OBJECT_LIMIT_JOB_MEMORY      = 0x0200  # verify
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000  # verify
CPU_RATE_ENABLE = 0x1; CPU_RATE_HARD_CAP = 0x4 # verify

class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]

class IoCounters(ctypes.Structure):
    _fields_ = [(k, ctypes.c_ulonglong) for k in
                ("ReadOperationCount","ReadTransferCount","WriteOperationCount",
                 "WriteTransferCount","OtherOperationCount","OtherTransferCount")]

class ProcMemCounts(ctypes.Structure):     # mirrors measure.py _PMC
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]

class ExtLimits(ctypes.Structure):
    _fields_ = [("Basic", BasicLimits), ("Io", IoCounters),
                ("ProcessMemoryInfo", ProcMemCounts),
                ("ProcessMemoryLimit", ctypes.c_size_t),  # per-process COMMIT cap
                ("JobMemoryLimit", ctypes.c_size_t),      # job-wide COMMIT cap
                ("PeakJobMemoryUsed", ctypes.c_size_t)]   # peak COMMIT, job-wide

class CpuRateControl(ctypes.Structure):
    _fields_ = [("ControlFlags", wintypes.DWORD),
                ("CpuRate", wintypes.DWORD)]  # layout verify: expect 8-byte struct

class Job:
    def __init__(self, mem_bytes=None, per_process_bytes=None,
                 affinity_mask=None, cpu_rate=None):
        self.h = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        self.h = ctypes.c_void_p(self.h)          # HANDLE restype trap (see measure.py)
        ext = ExtLimits()
        ext.Basic.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if mem_bytes:
            ext.Basic.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY
            ext.JobMemoryLimit = mem_bytes
        if per_process_bytes:
            ext.Basic.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
            ext.ProcessMemoryLimit = per_process_bytes
        if affinity_mask:
            ext.Basic.LimitFlags |= JOB_OBJECT_LIMIT_AFFINITY
            ext.Basic.Affinity = affinity_mask
        if not ctypes.windll.kernel32.SetInformationJobObject(
                self.h, JobObjectExtendedLimitInformation,
                ctypes.byref(ext), ctypes.sizeof(ext)):
            raise ctypes.WinError()
        if cpu_rate is not None:                  # percent x100 over all LPs
            crc = CpuRateControl(ControlFlags=CPU_RATE_ENABLE | CPU_RATE_HARD_CAP,
                                 CpuRate=cpu_rate)
            if not ctypes.windll.kernel32.SetInformationJobObject(
                    self.h, JobObjectCpuRateControlInformation,
                    ctypes.byref(crc), ctypes.sizeof(crc)):
                raise ctypes.WinError()
```

Field note (implementation must match): `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`
ends with `ProcessMemoryLimit; JobMemoryLimit; PeakJobMemoryUsed;` — memory caps
live on the EXTENDED struct, not the basic one.

```python
def spawn_constrained(cmd, *, mem_bytes, affinity_mask, cpu_rate=None):
    """Create child CREATE_SUSPENDED -> assign to job -> resume.
    Suspended-start closes the race where the child spawns helpers before
    assignment (gotcha G5)."""
    hjob = Job(mem_bytes=mem_bytes, affinity_mask=affinity_mask, cpu_rate=cpu_rate)
    proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_SUSPENDED)
    if not ctypes.windll.kernel32.AssignProcessToJobObject(
            hjob.h, ctypes.c_void_p(proc._handle)):
        raise ctypes.WinError()
    resume_all_threads(proc)                  # via NtResumeProcess or thread walk
    return hjob, proc

def job_peak_commit(hjob) -> int:             # accounting for results
    ext = ExtLimits()
    ctypes.windll.kernel32.QueryInformationJobObject(
        hjob.h, JobObjectExtendedLimitInformation,
        ctypes.byref(ext), ctypes.sizeof(ext), None)
    return ext.PeakJobMemoryUsed              # commit-based (gotcha G1)
```

### Other modules (responsibilities + key signatures)

```python
# topology.py
def pe_topology() -> dict[lp_index, "P"|"E"]   # GetLogicalProcessorInformationEx
                                               # (RelationProcessorCore + efficiency class)
def canonical_mask(vcpus: int, kind="P") -> int

# profiles.py
PROFILES = {"P1": dict(vcpus=1, cap=256_000_000, threads=1), ...}   # §4 verbatim

# payloads.py
def corpus_item(name) -> dict          # conv_001 / para_001 from frozen corpus
MALFORMED_SUITE: list[tuple[str, str]]  # §6-S5 enumerated cases, id + payload
def synthetic_paragraph(chars: int) -> str
CORRUPTION_MUTATIONS = ["M1_trunc50", ..., "M7_wav_renamed"]

# monitor.py  — sampler thread (per job): poll psutil children rss/vms/uss @50 ms,
# QueryInformationJobObject for PeakJobMemoryUsed at end; emit timeline JSONL.
# Leak metrics computed on COMMIT (vms/PagefileUsage), not RSS (gotcha G3).

# pressure.py — `hog(target_free_bytes)`: allocate+touch 64 MB chunks until
# psutil.virtual_memory().available <= target; context manager releases in finally.

# scenarios.py — every scenario: run(ctx: ProfileCtx) -> ScenarioResult
@dataclass
class ScenarioResult:
    scenario: str; profile: str; verdict: str        # PASS|CLEAN_FAIL|HARD_FAIL
    metrics: dict; failures: list[str]; provenance: dict

# run_matrix.py CLI
#   python -m benchmarks.torture.run_matrix --profiles P1..P6 --scenarios S1..S10
#       --mechanism jobobj [--experiment-id CV-G1-E00XX]
# Writes benchmarks/results/torture/<id>/results.jsonl + meta.json
# (meta mirrors existing harness convention: git commit, machine snapshot, corpus hash).
```

---

## 8. Concurrency ramp 1→16 and safe-concurrency estimation (Acceptance item 4)

**Legs:** k ∈ {1, 2, 4, 8, 12, 16}, each leg ≥ 60 s or ≥ 2× longest item, two arrival
patterns per leg (simultaneous burst; staggered 1/s). Workers are separate processes
in ONE constrained job (Windows has no fork-CoW, gotcha G6 — each worker pays private
memory; weights may share only via file-backed sections). Between legs: drain, gc,
verify job commit returns to idle floor (leak gate between legs).

**Thread policy:** canonical (threads = profile vCPUs). Optional variant B leg:
threads = max(1, vCPUs // k) to test oversubscription relief (thread sweep showed
weak scaling beyond ~4 threads — Baseline-MEASURED).

**Measured quantities per leg:** per-stream wall latency, TTFA, per-stream RTF,
aggregate throughput (audio-seconds/wall-second), error count, peak job commit,
p95 latency.

**Derived (measured):**
- Efficiency `E(k) = agg_throughput(k) / (k · agg_throughput(1))`
- Saturation point `k*` = smallest k where ANY trip: `E(k) < 0.80`;
  p95 latency > 3× solo median; TTFA > 2× solo TTFA; error rate > 0; any S-invariant
  HARD-FAIL.

**Predictive formula (computed before the ramp, from solo-leg measurements):**

```
ρ      = solo CPU-seconds per audio-second at profile p's canonical threads
         (= harness cpu_seconds / audio_seconds, median of ≥5 solo runs)
U      = utilization budget = 0.90 (headroom for scheduler/GC/OS)
V_p    = profile vCPUs
M_idle = job commit with one worker loaded and idle   (fixed cost incl. weights)
b̄      = median INCREMENTAL commit per additional worker
         (= PeakJobMemoryUsed(k) − PeakJobMemoryUsed(k−1), legs ≥2)

N_cpu  = floor(U · V_p / ρ)                    # CPU-bound bound
N_mem  = floor((cap_p − M_idle) / b̄)           # memory-bound bound
N_safe(p) = max(1, min(N_cpu, N_mem))
```

**Calibration rule:** if measured `k*` deviates from `N_safe` by more than ±1,
recalibrate ρ and b̄ from the new legs and re-run k ∈ {k*−1, k*, k*+2}; converged
pairs are reported together (`predicted vs measured`). Both numbers publish — the
formula makes future profiles predictable, the ramp keeps it honest.

Profile-specific expectation (HYPOTHESIS, from thread sweep): P1–P3 saturate at
k*=1–2; P6 should reach materially higher k*; the 16-stream leg is EXPECTED to be
past saturation on most profiles — graceful degradation there is the clean-fail
behavior under test, not a failure of the lab.

---

## 9. Known Windows gotchas (Acceptance item 5)

**G1 — Working-set vs commit: Job memory limits charge COMMIT, not working set.**
`JobMemoryLimit`/`ProcessMemoryLimit` bound commit charge (private + pagefile-backed).
File-backed mappings (mmap-loading an ONNX/weights file) do NOT count toward commit —
a 256 MB profile could admit a multi-hundred-MB mmap'd model, then die on working-set
trim/thrash with NO typed boundary. Mitigations: prefer read-into-anonymous-memory
loading for torture runs; always log BOTH `PeakJobMemoryUsed` (commit) and
job `PeakWorkingSetSize` + psutil RSS; detect divergence > 15% between commit-peak
and WS-peak and flag possible evasion in the result record (clean-fail req. #4).

**G2 — Two different "peaks".** `PeakWorkingSetSize` (process working set) and
`PeakJobMemoryUsed` (job commit peak) answer different questions; neither substitutes
for the other. Report both; RSS-only reporting under-reports true footprint on Windows.

**G3 — Swap hides host OOM; commit is the honest ceiling.** 49 GB pagefile means
host-RAM exhaustion essentially cannot happen; deterministic boundaries must come from
job limits. Additionally, under host pressure Windows trims working sets aggressively,
so RSS-based leak detection false-negatives — S3 leak gates therefore run on commit
(`vms`/PagefileUsage), never RSS alone.

**G4 — Hybrid P/E cores + Thread Director.** Without affinity the scheduler floats
threads across P and E cores; results are irreproducible and the thread sweep already
showed bimodality. Canonical masks (§4) are mandatory; E-core variants measured once
during shakedown and documented. Never interpret an unpinned number as a profile result.

**G5 — Assignment race.** A child started running freely may spawn helpers before
`AssignProcessToJobObject`; always CREATE_SUSPENDED → assign → resume, and set
`KILL_ON_JOB_CLOSE` so a crashing harness still reaps the tree. Nested jobs (harness
itself inside a CI job) are supported on Win11 but must be tested in shakedown.

**G6 — No fork CoW.** Concurrent workers are full processes; each pays private
committed copy of interpreter/runtime unless pages come from shared file-backed
sections. Hence per-worker increment b̄ in §8 is measured, never assumed = model size.

**G7 — File locking.** Windows locks open files harder than POSIX; S7 restart checks
model-file rename/delete explicitly; loaders should open with full sharing flags or
use copy-then-load to keep hot-reload/restart viable.

**G8 — Cache control.** No user-mode "drop caches"; cold-boot variance is fought with
standby-list flushing via a brief memory hog + N-repeat medians. Antivirus scanning of
first model load adds one-time outliers — excluded by warmup discipline.

**G9 — CPU-rate granularity.** Rate-control enforcement window is sub-second
(documented scheduling-window semantics; exact window REPORTED BY THIRD PARTY at
~100 ms), producing bursty delivery: rate-capped RTF numbers are noisier/pessimistic
vs affinity-pinned equivalents. If rate mode is exercised, report it as a separate
variant, never merged with affinity results.

**G10 — Laptop environment.** AC power required; cooldown gaps between heavy legs;
log `psutil.cpu_freq()` per leg to expose thermal drift; Defender exclusions for the
venv are a user-approved config change, otherwise accept and document scan jitter.
HANDLE-typed ctypes calls must set `.restype = c_void_p` (trap already documented in
`measure.py`).

---

## 10. Execution order & deliverables

1. **Shakedown (Mechanism A proof):** implement `jobobject.py` + `topology.py`;
   validate constants/struct sizes empirically (SetInformationJobObject error codes);
   measure P-vs-E spread once; verify nested-job behavior; confirm commit-limit
   produces typed failure in a toy allocator loop.
2. **Harness completion** (remaining modules) + unit smoke: constrain a trivial
   sleeper process at 128 MB / 1 LP and observe the kill.
3. **Matrix run:** all 60 cells, Mechanism A. Persist
   `benchmarks/results/torture/<experiment-id>/{results.jsonl, meta.json}`.
4. **Cross-validation subset** (Mechanism B): 10 scenarios × {P2, P3} under
   Docker/WSL2 cgroups; verdict-transfer comparison table (verdicts only, not times).
5. **Fallback activation** only per §3.3 escalation.
6. **Report:** `report.py` rollup appended to this folder as
   `TORTURE-LAB-RESULTS.md` with per-cell verdicts, measured minima, predicted-vs-
   measured concurrency tables, evidence labels; charter §9.5 minima published for
   every unreachable target.

Proposed experiment IDs (confirm against EXPERIMENT-INDEX.md at launch to avoid
collisions): **CV-G1-E0007** (shakedown) and **CV-G1-E0008** (matrix + report).

### Open questions for coordinator
- Confirm strictest-reading caps (SI decimal) match charter intent.
- Cancel semantics: is 500 ms cooperative ack the product requirement, or stage-boundary?
- Does the Phase-J runtime expose a health endpoint S7 can probe, or is restart
  process-lifecycle-only for now?
