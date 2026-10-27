"""AI Studio command worker: Kokoro-82M (ONNX, CPU) narration.
Request JSON fields: text (or prompt), voice (default af_heart), lang (default en-us), speed (1.0)."""
import argparse
import json
import sys
from pathlib import Path

import onnxruntime
import soundfile as sf
from kokoro_onnx import Kokoro

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--request", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
req = json.loads(Path(a.request).read_text(encoding="utf-8"))
text = (req.get("text") or req.get("prompt") or "").strip()
if not text:
    sys.exit("kokoro worker: no text to speak")
voice, lang, speed = req.get("voice") or "af_heart", req.get("lang") or "en-us", float(req.get("speed") or 1.0)
k = Kokoro(str(here / "kokoro-v1.0.onnx"), str(here / "voices-v1.0.bin"))
if voice not in k.get_voices():
    sys.exit(f"kokoro worker: unknown voice {voice!r}; have {sorted(k.get_voices())}")
samples, rate = k.create(text, voice=voice, speed=speed, lang=lang)
sf.write(a.out, samples, rate, subtype="PCM_24", format="WAV")  # native rate, no resampling
print(json.dumps({"device": "cpu", "providers": onnxruntime.get_available_providers()[:1], "model": "kokoro-v1.0.onnx",
                  "voice": voice, "lang": lang, "speed": speed, "sample_rate": rate}))
