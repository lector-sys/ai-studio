#!/usr/bin/env bash
# One timed phase of the creative-edit benchmark (see tools/benchmark_creative.py).
#   tools/bench_phase.sh <run> <ws-name> inspect|prepare|render|qa|deliver
# The review phase is timed by the operator around the delegated reviewers:
#   python tools/benchmark_creative.py mark <timing.json> review start|end
set -euo pipefail
RUN=$1 WSNAME=$2 PHASE=$3
ROOT=D:/projects/AI_studio
PY=$ROOT/.venv/Scripts/python.exe
STUDIO=$ROOT/.venv/Scripts/studio.exe
OUT="D:/edited(studio)/Creative toolkit benchmark/$RUN"
T="$OUT/timing.json"
export STUDIO_WORKSPACE=$ROOT/workspace/$WSNAME
mkdir -p "$OUT"
mark() { "$PY" "$ROOT/tools/benchmark_creative.py" mark "$T" "$PHASE" "$1"; }
mark start
case $PHASE in
  inspect)
    "$STUDIO" inspect D:/footage/IMG_3532.MOV --hash > "$OUT/source_inspection.json"
    for a in almond oval square; do "$STUDIO" inspect "D:/footage/$a.jpeg" --hash > "$OUT/asset_$a.json"; done ;;
  prepare)
    sed "s|D:/edited(studio)/Creative toolkit benchmark/run-1/|$OUT/|" "$ROOT/examples/creative_benchmark/request.json" > "$OUT/request.json"
    "$STUDIO" recipe image-popup --request "$OUT/request.json" -o "$OUT/job.json" --assets "$STUDIO_WORKSPACE/cards" > "$OUT/recipe.json" ;;
  render)
    STUDIO_HW_ENCODE=1 "$STUDIO" render "$OUT/job.json" > "$OUT/render.json" 2> "$OUT/render.log" ;;
  qa)
    LOG=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['log'])" "$OUT/render.json")
    "$STUDIO" review "$LOG" -o "$OUT/review" > "$OUT/qa.json" ;;
  deliver)
    "$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "D:/projects/AI_studio")
from studio.media import sha256
out = Path(sys.argv[1])
r = json.loads((out / "render.json").read_text(encoding="utf-8"))
q = json.loads((out / "qa.json").read_text(encoding="utf-8"))
reviews = json.loads(Path(q["reviews"]).read_text(encoding="utf-8"))["reviews"]
files = {"master": r["output"], "viewing_copy": q["viewing_copy"]}
hashes = {k: sha256(v) for k, v in files.items() if v}
assert hashes["master"] == r["sha256"] == q["output_sha256"], "master hash changed after render"
src = sha256("D:/footage/IMG_3532.MOV")
assert src == "7e77980a566944074d2cb687396d492dc4c331569aee284414cf753ceb16de48", "original changed"
(out / "delivery.json").write_text(json.dumps({"files": files, "sha256": hashes, "source_sha256": src,
    "technical_ok": q["technical_ok"], "reviews": reviews}, indent=1), encoding="utf-8")
print(json.dumps({"technical_ok": q["technical_ok"], "reviews": {k: v["status"] for k, v in reviews.items()}}))
EOF
    ;;
esac
mark end
