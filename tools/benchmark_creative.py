"""Wall-clock phase timer for complete creative-edit benchmarks (inspection -> verified delivery).

    python tools/benchmark_creative.py mark run1.json inspect start   # ... end
    python tools/benchmark_creative.py note run1.json path=compose cache=cold
    python tools/benchmark_creative.py summary run1.json run2.json run3.json

Phases: inspect, prepare, render, qa, review, deliver (a parallel span ends its name with '||').
Pauses awaiting a user choice are recorded as phase 'user-wait' and excluded from the total."""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.review import measured_phases  # noqa: E402


def _load(f):
    p = Path(f)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"marks": [], "notes": {}}


def main(argv):
    cmd, f = argv[0], argv[1]
    if cmd == "mark":
        d = _load(f)
        d["marks"].append([argv[2], argv[3], time.time()])
        Path(f).write_text(json.dumps(d, indent=1), encoding="utf-8")
    elif cmd == "note":
        d = _load(f)
        d["notes"] |= dict(kv.split("=", 1) for kv in argv[2:])
        Path(f).write_text(json.dumps(d, indent=1), encoding="utf-8")
    elif cmd == "summary":
        rows = []
        for run in argv[1:]:
            d, open_, spans, wait = _load(run), {}, [], 0.0
            for name, edge, t in d["marks"]:
                if edge == "start":
                    open_[name] = t
                else:
                    a = open_.pop(name)
                    if name == "user-wait":
                        wait += t - a
                    else:
                        spans.append((name, a, t))
            t0 = min(a for _, a, _ in spans)
            r = measured_phases([(n, a - t0, b - t0) for n, a, b in sorted(spans, key=lambda s: s[1])],
                                d["notes"].get("tokens"))
            r["total_seconds"] = round(r["total_seconds"] - wait, 3)
            rows.append({"run": run, "notes": d["notes"], "user_wait_seconds": round(wait, 3)} | r)
        keys = sorted({k for r in rows for k in r if k.endswith("_seconds")})
        stats = {k: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                 for k in keys if (v := [r[k] for r in rows if k in r])}
        print(json.dumps({"runs": rows, "stats": stats}, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
