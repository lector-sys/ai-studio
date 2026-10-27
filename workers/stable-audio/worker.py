"""AI Studio command worker: Stable Audio 3 Small (music / SFX) via the official LiteRT CPU scripts.
Request JSON: kind (music | sfx), prompt, duration (s), seed. Output: the script's native 44.1 kHz stereo WAV.
Pinned source: workers/stable-audio/src (Stability-AI/stable-audio-3 @ 77e0398, optimized/tflite).
Weights: stabilityai/stable-audio-3-optimized (Stability AI Community License), cached in models/hf."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
tfl = here / "src" / "optimized" / "tflite"
ap = argparse.ArgumentParser()
ap.add_argument("--request", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
req = json.loads(Path(a.request).read_text(encoding="utf-8"))
kind = req.get("kind")
if kind not in ("music", "sfx"):
    sys.exit(f"stable-audio worker: kind must be music or sfx, got {kind!r}")
prompt = (req.get("prompt") or "").strip()
if not prompt:
    sys.exit("stable-audio worker: empty prompt")
seconds = float(req.get("duration") or (20 if kind == "music" else 5))  # README: music wants >= ~20 s
seed = int(req["seed"]) if req.get("seed") is not None else int(hashlib.sha256(prompt.encode()).hexdigest()[:7], 16)
threads = min(8, os.cpu_count() or 8)
dit = "sm-music" if kind == "music" else "sm-sfx"
env = dict(os.environ, HF_HOME=str(here / "models" / "hf"), PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_TELEMETRY="1")
cmd = [sys.executable, str(tfl / "scripts" / "sa3_tflite.py"), "--prompt", prompt, "--dit", dit, "--decoder", "same-s",
       "--dit-precision", "w8a8", "--decoder-precision", "w8a8", "--seconds", str(seconds), "--seed", str(seed),
       "--threads", str(threads), "--out", str(Path(a.out).resolve())]
r = subprocess.run(cmd, cwd=tfl, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                   stdin=subprocess.DEVNULL)
if r.returncode:
    sys.exit(f"sa3_tflite failed ({r.returncode}): {(r.stderr or r.stdout)[-2000:]}")
print(json.dumps({"device": "cpu", "model": f"sa3-{dit} (DiT w8a8)", "decoder": "same-s (w8a8)", "seconds": seconds,
                  "seed": seed, "threads": threads, "source": "Stability-AI/stable-audio-3@77e0398 optimized/tflite"}))
