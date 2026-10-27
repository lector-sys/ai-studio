"""Shared review evidence for a rendered job: one bundle (inspection, cue/checkpoint table, native frames +
100% crops, entrance/exit neighbours, SDR viewing copy, audio excerpts, technical checks, provenance)
that several independent reviewers use at once. Files are content-addressed, so a second call with the
same output reuses them. Automated checks are never recorded as human review."""
import json
import os
import math
import re
import subprocess
import time
from bisect import bisect_left
from fractions import Fraction
from hashlib import sha256 as _h
from pathlib import Path

from . import jobs
from .media import ffmpeg, frame_clock, info, sha256, tool
from .qc import audio_alignment, audio_metrics, clock_match

TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,tonemap=tonemap=hable:desat=0,"
           "zscale=p=bt709:t=bt709:m=bt709:r=limited,format=yuv420p")
REVIEWS = ["early-frames-and-framing", "late-frames-and-moving-edges", "background-detail", "playback-and-listening"]


def measured_phases(phases, tokens=None) -> dict:
    """phases: [(name, start_s, end_s)] in order; overlapping spans must be named '<x>||' (parallel)."""
    out, prev = {}, None
    for name, a, b in phases:
        if not (math.isfinite(a) and math.isfinite(b)) or b < a:
            raise ValueError(f"phase {name}: needs finite start <= end")
        if prev is not None and a < prev - 1e-9 and not name.endswith("||"):
            raise ValueError(f"phase {name} overlaps the previous one; mark parallel spans with '||'")
        out[f"{name.rstrip('|')}_seconds"] = round(out.get(f"{name.rstrip('|')}_seconds", 0) + b - a, 3)
        prev = max(prev or b, b)
    out["total_seconds"] = round(max(b for _, _, b in phases) - min(a for _, a, _ in phases), 3)
    out["tokens"] = tokens
    return out


def _frames_at(clock, times):
    """Index of the first frame at/after each time (s, from stream start)."""
    tb = Fraction(clock["time_base"])
    ts = [(p - clock["pts"][0]) * tb for p in clock["pts"]]
    return [min(len(ts) - 1, bisect_left(ts, Fraction(t).limit_denominator(10 ** 6))) for t in times], ts


def _extract(path, frames, out_dir, tag, hdr, crops):
    """Decode `path` once; save each wanted frame index (full, tone-mapped for viewing if HDR) and its
    crops. One sequential decode beats per-frame seeks here (measured: 9 s vs 21 s on 4K HEVC)."""
    want = sorted(set(frames))
    # inside quotes: commas are literal. ffmpeg's parser rejects >100 flat `+` terms, so group them by 50.
    terms = [f"eq(n,{n})" for n in want]
    sel = "+".join("(" + "+".join(terms[k:k + 50]) + ")" for k in range(0, len(terms), 50))
    view = f"select='{sel}'" + (f",{TONEMAP}" if hdr else "")
    parts = [f"[0:v]{view},split={1 + len(crops)}[full]" + "".join(f"[c{i}]" for i in range(len(crops)))]
    parts += [f"[c{i}]crop={w}:{h}:{x}:{y}[o{i}]" for i, (_, x, y, w, h) in enumerate(crops)]
    args = ["-i", path, "-filter_complex", ";".join(parts), "-map", "[full]", "-fps_mode", "passthrough",
            out_dir / f"{tag}-%05d.png"]
    for i, (name, *_r) in enumerate(crops):
        args += ["-map", f"[o{i}]", "-fps_mode", "passthrough", out_dir / f"{tag}-{name}-%05d.png"]
    ffmpeg(args)
    files = {}
    for k, n in enumerate(want, 1):  # rename sequence numbers to frame indices
        for suffix in [""] + [f"-{c[0]}" for c in crops]:
            src = out_dir / f"{tag}{suffix}-{k:05d}.png"
            dst = out_dir / f"{tag}{suffix}-f{n:05d}.png"
            if src.exists():
                src.replace(dst)
                files.setdefault(n, []).append(dst.name)
    return files


def preserved_outside(source, rendered, boxes) -> dict:
    """Every decoded frame of `rendered` equals `source` outside the layer boxes (per-frame YUV checksums)."""
    def md5s(path):
        fill = ",".join(f"drawbox=x={x}:y={y}:w={w}:h={h}:t=fill:c=black" for x, y, w, h in boxes) or "null"
        r = subprocess.run([tool("ffmpeg"), "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-vf",
                            f"format=yuv420p10le,{fill}", "-f", "framecrc", "-"], capture_output=True, text=True)
        return [line.rsplit(",", 1)[1].strip() for line in r.stdout.splitlines() if line and not line.startswith("#")]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        a, b = ex.map(md5s, [source, rendered])
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"ok": len(a) == len(b) and not diff and bool(a), "frames": len(b), "source_frames": len(a),
            "differing_frames": diff[:20], "masked_boxes": boxes}


def psnr(reference, distorted) -> dict:
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", distorted, "-i", reference, "-lavfi",
                        "[0:v]settb=1/1000,setpts=N,format=yuv420p10le[a];[1:v]settb=1/1000,setpts=N,format=yuv420p10le[b];"
                        "[a][b]psnr=stats_file=-",  # pair by frame index: the .mkv clock is ms-rounded
                        "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    ys = [float(m) for m in re.findall(r"psnr_y:([\d.]+|inf)", r.stdout) if m != "inf"]
    avg = re.search(r"PSNR y:([\d.]+|inf)", r.stderr)
    return {"y_average_db": avg and avg.group(1), "y_min_db": min(ys) if ys else None, "frames": len(ys)}


def _layer_boxes(step, base_v):
    """Union boxes (source px) each layer can touch, including its slide-in travel."""
    out = []
    for L in step.params.get("layers", []):
        x, y = int(L["x"]) - 2, int(L["y"]) - 2
        h = int(L["height"]) + abs(int(L.get("slide", 46) if L.get("animation", "slide-scale") == "slide-scale" else 0)) + 4
        x, y = max(0, x), max(0, y)
        box = [x, y, min(base_v["width"] - x, int(L["width"]) + 4), min(base_v["height"] - y, h)]
        if box not in out:  # each drawbox costs a full-frame pass
            out.append(box)
    return out


def bundle(job: jobs.Job, result: dict, root) -> dict:
    """Build (or reuse) the review bundle for a rendered job result."""
    t0 = time.time()
    root = Path(root)
    out_path = result["output"]
    primary = next(iter(job.sources))
    src = job.sources[primary].path
    sv, ov = info(src)["video"], info(out_path)["video"]
    key = _h(json.dumps([sha256(out_path), sha256(src), job.model_dump(mode="json")], sort_keys=True,
                        default=str).encode()).hexdigest()[:16]
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "bundle.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("key") == key:
            old["reused"] = True
            return old
    ev = root / f"evidence-{key}"
    ev.mkdir(exist_ok=True)
    clock = frame_clock(out_path)
    comp = next((s for s in job.steps if s.op == "compose"), None)
    dur = ov["duration"]
    checkpoints = []  # (label, time) -- entrance/exit neighbours for each layer, else evenly spaced
    if comp:
        for n, L in enumerate(comp.params["layers"]):
            checkpoints += [(f"L{n}-before", L["start"], -1), (f"L{n}-first", L["start"], 0),
                            (f"L{n}-entered", L["start"] + L.get("entrance", .22), 0),
                            (f"L{n}-mid", (L["start"] + L["end"]) / 2, 0), (f"L{n}-last", L["end"], -1),
                            (f"L{n}-after", L["end"], 0)]
    else:
        checkpoints = [(f"t{i}", dur * i / 8, 0) for i in range(9)]
    idx, ts = _frames_at(clock, [t for _, t, _ in checkpoints])
    frames = [max(0, min(len(ts) - 1, i + off)) for i, (_, _, off) in zip(idx, checkpoints)]
    W, H = ov["width"], ov["height"]
    crops = [("bg-crop", 0, H - min(H, 720), min(W, 720), min(H, 720))]
    if comp:
        L0 = comp.params["layers"][0]
        cw, ch = min(W, int(L0["width"]) + 80), min(H, int(L0["height"]) + 80)
        crops.append(("card-crop", max(0, min(W - cw, int(L0["x"]) - 40)), max(0, min(H - ch, int(L0["y"]) - 40)), cw, ch))
    same_timeline = result.get("mode", "final") == "final" and not any(
        s.op in jobs.FORMAT_CHANGING | jobs.RETIMING for s in job.steps)
    step_rec = (result.get("steps") or [{}])[-1]
    mid = step_rec.get("output") if comp and Path(step_rec.get("output") or "").exists() else None
    view = root / f"{Path(out_path).stem}_SDR_viewing_copy.mp4" if ov["hdr"] else None

    def viewing_copy():
        if view.exists():
            return
        part = view.with_name(view.stem + ".part.mp4")
        ffmpeg(["-i", out_path, "-map", "0:v:0", "-map", "0:a?", "-vf",
                f"scale=-2:'min(1080,ih)':flags=lanczos,{TONEMAP}", "-c:v", "libx264", "-preset", "fast",
                "-crf", "17", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", part])
        part.replace(view)

    def audio():
        au_src, au_out = audio_metrics(src), audio_metrics(out_path)
        res = {"source": au_src, "output": au_out}
        if au_src.get("has_audio") and au_out.get("has_audio"):
            res["alignment"] = audio_alignment(src, out_path)
            for name, at in (("start", 0), ("middle", max(0, dur / 2 - 5)), ("end", max(0, dur - 10))):
                for tag, f in (("edit", out_path), ("source", src)):
                    ffmpeg(["-ss", at, "-t", 10, "-i", f, "-map", "0:a:0", "-c:a", "pcm_s24le",
                            ev / f"audio-{name}-{tag}.wav"])
        return res

    # Every heavy check is an independent full-file decode: run them side by side.
    from concurrent.futures import ThreadPoolExecutor

    class Timed:
        def __init__(self, ex):
            self.ex = ex

        def submit(self, fn, *a):
            def run():
                t = time.time()
                try:
                    return fn(*a)
                finally:
                    fut.seconds = time.time() - t
            fut = self.ex.submit(run)
            fut.seconds = None
            return fut

    with ThreadPoolExecutor(int(os.environ.get("STUDIO_REVIEW_WORKERS", 3))) as pool:  # 4K decodes thrash past ~3
        ex = Timed(pool)
        jobs_ = {"edit": ex.submit(_extract, out_path, frames, ev, "edit", ov["hdr"], crops),
                 "source": ex.submit(_extract, src, frames, ev, "source", sv["hdr"], crops),
                 "audio": ex.submit(audio)}
        if same_timeline:
            jobs_["clock"] = ex.submit(clock_match, src, out_path)
        if mid:
            jobs_["preserved_outside_layers"] = ex.submit(preserved_outside, src, mid, _layer_boxes(comp, sv))
            jobs_["compression"] = ex.submit(psnr, mid, out_path)
        if view:
            jobs_["view"] = ex.submit(viewing_copy)
        done = {k: f.result() for k, f in jobs_.items()}
    timings = {k: round(f.seconds, 1) for k, f in jobs_.items()}
    fo, fs = done.pop("edit"), done.pop("source")
    done.pop("view", None)
    # Playback reviewer's overview: 1 frame/s contact sheet + source/edit waveforms (not a listening pass).
    sheet = ev / "playback-contact-sheet.png"
    cols = 10
    rows = max(1, math.ceil(dur / cols))
    ffmpeg(["-i", str(view or out_path), "-vf", f"fps=1,scale=-2:240,tile={cols}x{rows}", "-frames:v", 1,
            "-update", 1, sheet])
    if info(out_path)["audio"] and not info(src)["audio"]:  # generated/silent source: edit waveform only
        ffmpeg(["-i", out_path, "-lavfi", "[0:a:0]showwavespic=s=2400x240:split_channels=0", "-frames:v", 1,
                "-update", 1, ev / "waveform-edit.png"])
    elif info(out_path)["audio"]:
        ffmpeg(["-i", src, "-i", out_path, "-filter_complex",
                "[0:a:0]showwavespic=s=2400x240:split_channels=0[a];[1:a:0]showwavespic=s=2400x240:split_channels=0[b];"
                "[a][b]vstack", "-frames:v", 1, "-update", 1, ev / "waveforms-source-top-edit-bottom.png"])
    table = [{"checkpoint": c, "frame": f, "time": float(ts[f]), "edit": fo.get(f, []), "source": fs.get(f, [])}
             for (c, _, _), f in zip(checkpoints, frames)]
    checks = {"validation": result.get("validation"),
              "clock": {"ok": True, "n/a": "timeline intentionally changed or preview"}} | done
    reviews = {k: {"status": "pending", "reviewer": None, "findings": []} for k in REVIEWS}
    old_reviews = root / "reviews.json"
    if old_reviews.exists():  # keep findings already recorded for this exact output
        prev = json.loads(old_reviews.read_text(encoding="utf-8"))
        if prev.get("key") == key:
            reviews = prev["reviews"]
    old_reviews.write_text(json.dumps({"key": key, "reviews": reviews}, indent=1), encoding="utf-8")
    out = {"key": key, "output": out_path, "output_sha256": sha256(out_path), "source": src, "source_sha256": sha256(src),
           "inspection": {"source": sv, "output": ov, "source_audio": info(src)["audio"], "output_audio": info(out_path)["audio"]},
           "checkpoints": table, "evidence_dir": str(ev), "viewing_copy": str(view) if view else None,
           "checks": checks, "provenance": {"log": result.get("log"), "manifest": result.get("manifest")},
           "technical_ok": all(c.get("ok", True) for c in [checks["clock"], checks.get("preserved_outside_layers", {})])
           and bool((result.get("validation") or {}).get("ok")),
           "reviews": str(old_reviews), "seconds": round(time.time() - t0, 1), "check_seconds": timings,
           "reused": False}
    manifest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def record(root, kind, reviewer, status, findings) -> dict:
    """Record one human/delegated review result (status: pass | fail | needs-work)."""
    f = Path(root) / "reviews.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    if kind not in data["reviews"]:
        raise ValueError(f"unknown review {kind}; have {sorted(data['reviews'])}")
    if status not in ("pass", "fail", "needs-work"):
        raise ValueError("status must be pass | fail | needs-work")
    data["reviews"][kind] = {"status": status, "reviewer": reviewer, "findings": findings,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    f.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


if __name__ == "__main__":  # self-check of the phase accounting
    r = measured_phases([("inspect", 0, 2), ("prepare", 2, 8), ("render", 8, 25), ("qa", 25, 32),
                         ("review", 32, 70), ("deliver", 70, 72)])
    assert r["total_seconds"] == 72 and r["review_seconds"] == 38 and r["tokens"] is None
    p = measured_phases([("qa", 25, 32), ("review||", 28, 70)])
    assert p["total_seconds"] == 45
    print("review ok")
