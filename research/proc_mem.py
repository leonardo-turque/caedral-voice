import json
import psutil

vm = psutil.virtual_memory()
print(f"RAM avail: {vm.available/1e9:.1f} GB")
for pid in (6968, 41672, 1764, 38412, 50376):
    try:
        p = psutil.Process(pid)
        cl = " ".join(p.cmdline())
        print(pid, f"{p.memory_info().rss/1e9:.2f} GB", cl[60:160])
    except psutil.NoSuchProcess:
        print(pid, "gone")
