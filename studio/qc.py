"""Technical QA: decode integrity, frame count, resolution, fps, duration, A/V sync, optional SSIM."""
import re
import subprocess

from .media import ffprobe_json, info, tool


def _ssim(out, ref, w, h) -> float | None:
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", out, "-i", ref, "-lavfi",
                        f"[1:v]scale={w}:{h}[r];[0:v][r]ssim", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"All:([\d.]+)", r.stderr)
    return float(m.group(1)) if m else None


def validate(out, against=None, duration=None, tolerance=None, size=None, fps=None, ssim=False, decode=True) -> dict:
    o = info(out)
    checks = []

    def chk(name, ok, **detail):
        checks.append({"check": name, "ok": bool(ok), **detail})

    ov = o["video"]
    chk("has_video", ov is not None)
    if ov:
        rate = ov["fps"] or 30
        tol = tolerance or max(1.5 / rate, 0.05)
        if decode:
            r = subprocess.run([tool("ffmpeg"), "-v", "error", "-nostdin", "-i", out, "-f", "null", "-"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            errs = [line for line in r.stderr.splitlines() if line.strip()]
            chk("decodes_cleanly", r.returncode == 0 and not errs, errors=errs[:5])
        pk = ffprobe_json(["-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", out])
        n = int(pk["streams"][0]["nb_read_packets"])
        expected = round(ov["duration"] * rate)
        chk("frame_count", abs(n - expected) <= 2, frames=n, expected=expected)
        ref = info(against) if against else None
        rv = (ref or {}).get("video") or {}
        want = size or ([rv["width"], rv["height"]] if rv else None)
        if want:
            chk("resolution", [ov["width"], ov["height"]] == list(want), got=[ov["width"], ov["height"]], want=list(want))
        want_fps = fps or rv.get("fps")
        if want_fps:
            chk("fps", abs(ov["fps"] - want_fps) < 0.01, got=round(ov["fps"], 3), want=round(want_fps, 3))
        if duration is not None:
            chk("duration", abs(ov["duration"] - duration) <= tol, got=round(ov["duration"], 3), want=duration)
        if ref and ref["audio"]:
            chk("audio_present", o["audio"] is not None)
        if o["audio"]:
            a = o["audio"]
            dd, ds = abs(a["duration"] - ov["duration"]), abs(a["start_time"] - ov["start_time"])
            chk("av_sync", dd <= max(tol, 0.1) and ds <= 0.05, duration_delta=round(dd, 3), start_delta=round(ds, 3))
        if ssim and against and rv:
            s = _ssim(out, against, ov["width"], ov["height"])
            chk("ssim", s is not None and s >= 0.9, ssim=s)
    return {"ok": all(c["ok"] for c in checks), "file": out, "checks": checks,
            "summary": {"duration": ov and ov["duration"], "size": ov and [ov["width"], ov["height"]],
                        "fps": ov and ov["fps"], "vcodec": ov and ov["codec"], "pix_fmt": ov and ov["pix_fmt"],
                        "audio": o["audio"] and o["audio"]["codec"]}}


def clock_match(src, out) -> dict:
    """Compare every output frame time with the source's (an unchanged-timeline edit).
    Passes within max(one source tick, 0.5 ms): lossless .mkv intermediates store milliseconds."""
    from fractions import Fraction

    from .media import frame_clock
    a, b = frame_clock(src), frame_clock(out)
    ta, tb = Fraction(a["time_base"]), Fraction(b["time_base"])
    sa = [p * ta for p in a["pts"]]
    sa = [t - sa[0] for t in sa]
    sb = [p * tb - b["pts"][0] * tb for p in b["pts"]] if b["pts"] else []
    n = min(len(sa), len(sb))
    dev = max((abs(x - y) for x, y in zip(sa[:n], sb[:n])), default=Fraction(0))
    limit = max(ta, Fraction(1, 2000))
    return {"ok": len(sb) == len(sa) and dev <= limit, "frames_src": len(sa), "frames_out": len(sb),
            "max_dev_seconds": float(dev), "max_dev_ticks": float(dev / ta), "time_base": a["time_base"]}


def _pcm(path, rate=48000, start=None, dur=None, mono=True):
    import numpy as np
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin"] + (["-ss", str(start)] if start else []) + ["-i", str(path)]
    cmd += (["-t", str(dur)] if dur else []) + ["-map", "0:a:0", "-ac", "1" if mono else "2", "-ar", str(rate),
                                                "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"audio decode failed: {r.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(r.stdout, np.float32)


def audio_metrics(path) -> dict:
    """Stream facts + measured peak, clipped samples and integrated loudness (EBU R128)."""
    import numpy as np
    a = info(path)["audio"]
    if not a:
        return {"has_audio": False}
    x = _pcm(path, a["sample_rate"], mono=False)
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:a:0",
                        "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    m = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", r.stderr)
    peak = float(np.abs(x).max()) if x.size else 0.0
    return {"has_audio": True, "codec": a["codec"], "sample_rate": a["sample_rate"], "channels": a["channels"],
            "layout": a["layout"], "samples": x.size // 2, "duration": round(x.size / 2 / a["sample_rate"], 4),
            "peak_dbfs": round(20 * np.log10(peak), 2) if peak > 0 else None,
            "clipped_samples": int((np.abs(x) >= 0.999).sum()),
            "loudness_lufs": float(m[-1]) if m and m[-1] != "-inf" else None, "silent": peak < 1e-4}


def audio_alignment(reference, output, window=3.0, max_lag=0.2, rate=48000) -> dict:
    """Lag of `output` vs `reference` at start/middle/end windows by normalized cross-correlation.
    Silence, weak or ambiguous matches are inconclusive (lag None): never evidence of sync."""
    import numpy as np
    total = min(info(reference)["duration"] or 0, info(output)["duration"] or 0)
    starts = sorted({0.0, max(0.0, total / 2 - window / 2), max(0.0, total - window - max_lag)})
    k = int(max_lag * rate)
    rows = []
    for s in starts:
        a, b = _pcm(reference, rate, s, window), _pcm(output, rate, s, window)
        n = min(a.size, b.size)
        a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        if n <= 2 * k or np.sqrt((a * a).mean()) < 1e-3 or np.sqrt((b * b).mean()) < 1e-3:
            rows.append({"at": round(s, 3), "lag_samples": None, "confidence": 0.0, "conclusive": False})
            continue
        core = a[k:n - k]
        from scipy.signal import correlate
        c = correlate(b, core, mode="valid", method="fft")  # c[j] = core . b[j:j+len(core)], lag j-k
        e = np.concatenate([[0], np.cumsum(b.astype(np.float64) ** 2)])
        c /= np.linalg.norm(core) * np.sqrt(np.maximum(e[core.size:] - e[:-core.size], 0)) + 1e-12
        best = int(np.argmax(c))
        second = np.max(np.delete(c, range(max(0, best - 48), best + 49))) if c.size > 97 else 0
        conf = float(c[best])
        ok = conf >= 0.6 and conf - second >= 0.1
        rows.append({"at": round(s, 3), "lag_samples": best - k if ok else None, "confidence": round(conf, 3),
                     "conclusive": bool(ok)})
    lags = [r["lag_samples"] for r in rows if r["conclusive"]]
    return {"conclusive": bool(rows) and all(r["conclusive"] for r in rows),
            "lag_samples": max(lags, key=abs) if lags and len(lags) == len(rows) else None,
            "windows": rows, "rate": rate}


def validate_audio(out, duration=None, tolerance=None, decode=True) -> dict:
    """Audio-only QA: a decodable audio stream, positive rate/channels, measured length, peak and clipping.
    Silence is reported (intentional in music/SFX), not treated as corruption."""
    checks = []

    def chk(name, ok, **detail):
        checks.append({"check": name, "ok": bool(ok), **detail})

    a = info(out)["audio"]
    chk("has_audio", a is not None)
    m = {}
    if a:
        chk("format", (a["sample_rate"] or 0) > 0 and (a["channels"] or 0) > 0, rate=a["sample_rate"], channels=a["channels"])
        if decode:
            r = subprocess.run([tool("ffmpeg"), "-v", "error", "-nostdin", "-i", out, "-map", "0:a:0", "-f", "null", "-"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            errs = [line for line in r.stderr.splitlines() if line.strip()]
            chk("decodes_cleanly", r.returncode == 0 and not errs, errors=errs[:5])
        m = audio_metrics(out)
        chk("not_empty", m["samples"] > 0, samples=m["samples"])
        chk("no_clipping", m["clipped_samples"] == 0, clipped=m["clipped_samples"], peak_dbfs=m["peak_dbfs"])
        if duration is not None:
            chk("duration", abs(m["duration"] - float(duration)) <= (tolerance or 0.05), got=m["duration"], want=duration)
    return {"ok": all(c["ok"] for c in checks), "file": str(out), "checks": checks,
            "summary": {k: m.get(k) for k in ("duration", "sample_rate", "channels", "peak_dbfs", "loudness_lufs", "silent")}}
