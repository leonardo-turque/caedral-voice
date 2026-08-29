# Colab r3 — the ONE cell

## Before you open Colab (one-time, per package build)

Drag **all three** files from `artifacts/colab-r3/` into Google Drive folder
**`MyDrive/caedral/`** (create it if missing) and WAIT until both zips finish
uploading — no "uploading" badge left:

1. `code-package.zip`
2. `teacher-kokoro.zip`
3. `COLAB_BOOTSTRAP.py`

## In Colab

1. `Runtime > Change runtime type > GPU` (T4 is enough; anything better is
   faster). Then paste this single cell and run it:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!cp "/content/drive/MyDrive/caedral/COLAB_BOOTSTRAP.py" /content/
!python /content/COLAB_BOOTSTRAP.py
```

That's it. The bootstrap verifies your uploads against embedded SHA256s,
downloads LibriTTS-R itself, rebuilds + audits the preprocessed corpus,
runs smoke/AMP/benchmark gates, and only then prints
`COLAB R3 VALIDATED -- SAFE TO CUT OVER` and starts the 20k-step run.
If any gate fails it prints a diagnosis and exits non-zero — your local CPU
r3 keeps running regardless.

## Notes

* Interrupted (Runtime crashed, you closed the tab)? Just rerun the same
  cell. Every stage is idempotent; training auto-resumes from
  `experiments/colab-r3/checkpoints/last.pt`.
* Checkpoints sync continuously to `MyDrive/caedral-r3/checkpoints/`
  (`last.pt`, `prev-last.pt`, `best.pt`, `metrics.jsonl`). See RESUME.md for
  bringing them back to the local machine.
* Optional env flags: `CAEDRAL_FORCE_GATES=1` re-runs all gates even when a
  resumable checkpoint exists.
