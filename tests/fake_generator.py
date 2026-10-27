"""Fake command worker for generation-contract tests. Behaviour is chosen by the request prompt."""
import argparse
import json
import math
import struct
import subprocess
import sys
import time
import wave

ap = argparse.ArgumentParser()
ap.add_argument("--request")
ap.add_argument("--out")
a = ap.parse_args()
req = json.load(open(a.request, encoding="utf-8"))
mode = req.get("prompt") or "tone"
if mode == "fail":
    sys.exit("fake worker: model exploded")
if mode == "sleep":
    time.sleep(60)
if mode == "nothing":
    sys.exit(0)
if mode == "corrupt":
    open(a.out, "wb").write(b"RIFF\x00\x00\x00\x00WAVEjunk" * 10)
elif mode == "video":
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24", "-f", "lavfi", "-i",
                    "sine=f=330:r=48000", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    a.out[:-4] + ".mp4"], check=True)
    print(json.dumps({"device": "cpu", "output": a.out[:-4] + ".mp4"}))
    sys.exit(0)
else:  # tone: a short sine as 24 kHz mono PCM (length from duration, or from the text like real TTS)
    rate = 24000
    secs = float(req.get("duration") or (0.3 + 0.05 * len(req.get("text") or "")))
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                               for i in range(int(secs * rate))))
print(json.dumps({"device": "cpu", "model": "fake-tone", "echo_text": req.get("text")}))
