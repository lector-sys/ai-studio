"""Render cost of the same ten-card edit through existing separate ops (10 chained `overlay` steps +
`audio_clean`) for comparison with `compose`. Not colour-managed for HDR (overlay composites sRGB cards
into HLG without conversion), so this measures work only; its output is not a valid delivery.
    python tools/bench_separate_ops.py <compose job.json> <workspace> <output.mp4>"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio import jobs  # noqa: E402

src = jobs.load(sys.argv[1])
st = src.steps[0]
sources = dict(src.sources)
steps, cur = [], next(iter(src.sources))
for n, L in enumerate(st.params["layers"]):
    card = st.inputs[L["input"]]
    steps.append({"id": f"o{n}", "op": "overlay", "inputs": [cur, card],
                  "params": {"x": L["x"], "y": L["y"], "width": L["width"], "start": L["start"], "end": L["end"]}})
    cur = f"o{n}"
steps.append({"id": "clean", "op": "audio_clean", "inputs": [cur], "params": st.params.get("audio") or {}})
job = jobs.Job(name="separate-ops", sources=sources, steps=steps, output=sys.argv[3], profile=src.profile)
t = time.time()
r = jobs.run(job, ws=sys.argv[2], log=lambda m: print(m, file=sys.stderr))
print(json.dumps({"total_seconds": round(time.time() - t, 1), "encode_seconds": r["encode_seconds"],
                  "steps": [[s["step"], s["seconds"]] for s in r["steps"]], "ok": r["ok"]}, indent=1))
