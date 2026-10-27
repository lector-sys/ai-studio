"""Job manifests, content-addressed step cache (resume), output profiles and provenance."""
import inspect
import json
import os
import time
from datetime import datetime
from hashlib import sha256 as _h
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from . import __version__, compose, generate, restore, speech, vision  # noqa: F401  (modules register their ops)
from .media import audio_maps, ffmpeg, ffmpeg_version, hdr10, info, publish, sha256, tool
from .ops import OPS, Ctx
from .qc import validate, validate_audio
from .sched import gpu_lock

_SHARED = b"".join((Path(__file__).parent / f).read_bytes() for f in ("ops.py", "media.py"))


def op_version(name) -> str:
    """Cache keys include the code that renders a step (its module + the shared helpers),
    so editing an op invalidates its stale outputs without flushing every other op's cache."""
    return _h(_SHARED + Path(inspect.getsourcefile(OPS[name])).read_bytes()).hexdigest()[:12]

# Single-input, time-preserving ops that may run on overlapping chunks (resumable per chunk on long footage).
CHUNKABLE = {"mask", "denoise", "sharpen", "upscale", "stylize", "face_restore"}
FORMAT_CHANGING = {"crop", "resize", "rotate", "fps", "mask", "upscale", "interpolate", "extend", "generate",
                   "deinterlace"}

OUTPUT_EXT = {"generate_audio": ".wav"}  # step outputs that are not lossless .mkv video
# Ops that change frame times without changing the format (the source clock no longer applies).
RETIMING = {"trim", "speed", "reverse", "freeze", "concat", "transition", "cut"}

_AAC = ["-c:a", "aac", "-b:a", "320k"]
PROFILES = {
    "4k60-master": {"desc": "Preserve source resolution/fps; HEVC 10-bit CRF16 (use for 4K60 masters)",
                    "v": ["-c:v", "libx265", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"],
                    "hw": ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "16", "-qp_p", "18",
                           "-pix_fmt", "p010le", "-tag:v", "hvc1"], "a": _AAC},
    "native-hdr-master": {"desc": "Source-preserving HEVC Main10 (HDR/10-bit): AMD AMF constant QP 10/12 with "
                                  "STUDIO_HW_ENCODE=1, else x265 CRF 12",
                          "v": ["-c:v", "libx265", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p10le",
                                "-tag:v", "hvc1"],
                          "hw": ["-c:v", "hevc_amf", "-profile:v", "main10", "-quality", "quality", "-rc", "cqp",
                                 "-qp_i", "10", "-qp_p", "12", "-pix_fmt", "p010le", "-tag:v", "hvc1"], "a": _AAC},
    "h264-master": {"desc": "Preserve source resolution/fps; H.264 8-bit CRF16, maximum compatibility",
                    "v": ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"],
                    "hw": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "16", "-qp_p", "18",
                           "-pix_fmt", "yuv420p"], "a": _AAC},
    "av1-master": {"desc": "Preserve source resolution/fps; SVT-AV1 10-bit",
                   "v": ["-c:v", "libsvtav1", "-preset", "5", "-crf", "24", "-pix_fmt", "yuv420p10le"], "a": _AAC},
    "prores-hq": {"desc": "ProRes 422 HQ intermediate (.mov)",
                  "v": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "a": ["-c:a", "pcm_s24le"], "ext": ".mov"},
    "alpha-master": {"desc": "ProRes 4444 with alpha (.mov) for background removal / keyed output",
                     "v": ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"],
                     "a": ["-c:a", "pcm_s24le"], "alpha": True, "ext": ".mov"},
    "archive": {"desc": "Lossless FFV1 + FLAC (.mkv); also used for mask exports",
                "v": ["-c:v", "ffv1", "-level", "3"], "a": ["-c:a", "flac"], "alpha": True, "ext": ".mkv"},
    "preview": {"desc": "Cheap proxy-resolution H.264 for review",
                "v": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"],
                "vf": "scale=-2:'min(540,ih)'", "a": ["-c:a", "aac", "-b:a", "128k"], "scales": True},
    "wav": {"desc": "Audio-only PCM WAV (24-bit, native rate); chosen automatically for audio-only results",
            "a": ["-c:a", "pcm_s24le"], "ext": ".wav", "audio_only": True},
    "social-vertical": {"desc": "1080x1920 fill-crop H.264 for Shorts/Reels/TikTok",
                        "v": ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p"],
                        "vf": "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1",
                        "a": ["-c:a", "aac", "-b:a", "192k"], "size": [1080, 1920]},
}


class Source(BaseModel):
    path: str
    sha256: str | None = None


class Step(BaseModel):
    id: str
    op: str
    inputs: list[str]
    params: dict = Field(default_factory=dict)
    chunk: float | None = None  # seconds per chunk for long footage (0 = never; default STUDIO_CHUNK_SECONDS)


class Validation(BaseModel):
    against: str | None = None  # source id; default: primary source when no step changes format
    duration: float | None = None
    tolerance: float | None = None
    ssim: bool = False
    decode: bool = True


class Job(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    version: int = 1
    name: str = "job"
    sources: dict[str, Source]
    steps: list[Step]
    output: str
    profile: str = "4k60-master"
    proxy_height: int = 540
    retries: int = 1
    checks: Validation = Field(default_factory=Validation, alias="validate")


def workspace(root=None) -> Path:
    ws = Path(root or os.environ.get("STUDIO_WORKSPACE") or Path(__file__).resolve().parents[1] / "workspace")
    for d in ("proxies", "cache", "renders", "manifests", "logs"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    return ws


def load(path) -> Job:
    return Job.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _key(*parts) -> str:
    return _h(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


FILE_FIELDS = {"file", "font", "lut", "cue_file", "workflow"}


def resource_hashes(params) -> dict:
    """Content hashes of every file a step's params point at (fonts, LUTs, subtitles, models...), so
    replacing a file at the same path invalidates cached output. Documented file fields must exist."""
    out = {}

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and x:
            p = Path(x)
            if p.is_file():
                out[str(p.resolve())] = sha256(p)
            elif key in FILE_FIELDS:
                raise FileNotFoundError(f"param {key}: {x} does not exist")
    walk(params)
    return out


def _worker_id(params):
    """A generation step's output depends on its worker's config, models and runtime, not just its params."""
    if not params.get("worker"):
        return None
    from .workers import fingerprint, worker_config
    return fingerprint(worker_config(params["worker"]))


def _registry(ws) -> Path:
    return ws / "logs" / "outputs.json"


def proxy(path, digest, height, ws) -> str:
    out = ws / "proxies" / f"{digest[:16]}-{height}.mp4"
    if not out.exists():
        tmp = out.with_name(out.stem + ".part.mp4")
        ffmpeg(["-i", path, "-map", "0:v:0", *audio_maps(path), "-vf", f"scale=-2:'min({height},ih)'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", tmp])
        tmp.replace(out)
    return str(out)


def _hw_ok(enc: str) -> bool:
    """Capability probe: only use a hardware encoder that actually encodes a test frame."""
    try:
        ffmpeg(["-f", "lavfi", "-i", "color=s=640x360:d=0.1", "-c:v", enc, "-f", "null", "-"])  # AMF HEVC rejects tiny frames
        return True
    except RuntimeError:
        return False


def _hdr_params(enc, hdr) -> list:
    """Carry HDR10 mastering display + content light level into the master encode."""
    d, cll = hdr.get("display"), hdr.get("cll")
    if enc == "libx265":  # chroma in 0.00002 units, luminance in 0.0001 cd/m2
        xp = "hdr10=1:repeat-headers=1"
        if d:
            c = [round(v * 50000) for v in d[:8]]
            xp += (f":master-display=G({c[0]},{c[1]})B({c[2]},{c[3]})R({c[4]},{c[5]})WP({c[6]},{c[7]})"
                   f"L({round(d[8] * 10000)},{round(d[9] * 10000)})")
        return ["-x265-params", xp + (f":max-cll={cll[0]},{cll[1]}" if cll else "")]
    sp = []  # SVT-AV1: same layout, plain decimals
    if d:
        sp.append("mastering-display=G({:.5f},{:.5f})B({:.5f},{:.5f})R({:.5f},{:.5f})WP({:.5f},{:.5f})"
                  "L({:.4f},{:.4f})".format(*d))
    if cll:
        sp.append(f"content-light={cll[0]},{cll[1]}")
    return ["-svtav1-params", ":".join(sp)] if sp else []


def encode_final(src, out, prof, ref, keep_clock=False):
    """keep_clock: the timeline is unchanged, so restore the reference's exact time base (frame times
    pass through; the lossless .mkv intermediates only store milliseconds)."""
    if prof.get("audio_only"):
        return ffmpeg(["-i", src, "-map", "0:a:0"] + prof["a"] + [out])
    v = info(src)["video"]
    alpha = (v["pix_fmt"] or "").startswith(("yuva", "rgba", "bgra", "gbrap", "ya"))
    args = ["-i", src]
    if alpha and not prof.get("alpha"):
        if not prof.get("scales"):
            raise ValueError("output has alpha: use --profile alpha-master (or archive)")
        # preview of an alpha result: flatten over mid-gray so removed areas are visible
        args += ["-filter_complex", f"color=c=gray:s={v['width']}x{v['height']}:r={v['fps_str']}[bg];"
                 f"[bg][0:v]overlay=shortest=1,{prof.get('vf', 'null')}[v]", "-map", "[v]"]
    else:
        args += ["-map", "0:v:0"] + (["-vf", prof["vf"]] if prof.get("vf") else [])
    args += ["-map", "0:a?"]
    venc = prof["v"]
    if os.environ.get("STUDIO_HW_ENCODE") == "1" and prof.get("hw") and _hw_ok(prof["hw"][1]):
        venc = prof["hw"]
    hdr = hdr10(ref["path"]) if (ref.get("video") or {}).get("hdr") else None
    if hdr and venc[1] in ("libx265", "libsvtav1"):
        venc = venc + _hdr_params(venc[1], hdr)
    args += venc + prof["a"]
    rv = ref.get("video") or {}
    if rv.get("color_space") == "gbr":  # RGB source (e.g. rendered graphics): its tags don't apply to YUV output
        rv = dict(rv, color_space="bt709", color_primaries="bt709", color_transfer="bt709", color_range="tv")
    for k, flag in (("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"),
                    ("color_space", "-colorspace"), ("color_range", "-color_range")):
        if rv.get(k) and rv[k] not in ("unknown", "reserved"):
            args += [flag, rv[k]]
    tb = rv.get("time_base") or ""
    if keep_clock and tb.startswith("1/") and not prof.get("scales"):
        args += ["-fps_mode:v", "passthrough", "-enc_time_base:v", tb]
        if Path(out).suffix.lower() in (".mp4", ".mov", ".m4v"):
            args += ["-video_track_timescale", tb[2:]]
    if Path(out).suffix.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+faststart"]
    return ffmpeg(args + [out])


def _attempt(fn, tmp, ctx, retries, label, log) -> int:
    """Run one op invocation with the job's retry policy; returns the attempt that succeeded."""
    err = None
    for attempt in range(1, retries + 2):
        ctx.meta.clear()
        log(f"{label}: running (attempt {attempt})")
        try:
            fn()
            return attempt
        except Exception as e:  # noqa: BLE001 - retry policy covers any worker failure
            err = e
            Path(tmp).unlink(missing_ok=True)
            log(f"{label} failed: {e}")
    raise RuntimeError(f"{label} failed after {retries + 1} attempts: {err}")


def _chunked(step, src, tmp, ctx, chunk_s, retries, log) -> int:
    """Run a CHUNKABLE op over overlapping frame ranges. Each chunk gets a 1 s lead-in (temporal state
    like mask smoothing warms up) that is trimmed off; finished chunks survive an interruption, so a long
    job resumes mid-step. Audio is taken from the step input, untouched."""
    v = info(src)["video"]
    fps = v["fps"]
    n = v["frames"] or round(v["duration"] * fps)
    per, lead = max(1, round(chunk_s * fps)), round(fps)
    base = Path(tmp).with_name(Path(tmp).name.replace(".part.mkv", ""))
    parts, metas, worst = [], [], 1
    for c, a in enumerate(range(0, n, per)):
        b, a0 = min(n, a + per), max(0, a - lead)
        part = base.with_name(f"{base.name}-c{c:04}.mkv")
        if not (part.exists() and part.with_suffix(".json").exists()):
            seg, raw = part.with_name(part.stem + ".in.mkv"), part.with_name(part.stem + ".raw.mkv")
            ffmpeg((["-ss", (a0 - 0.5) / fps] if a0 else []) + ["-i", src, "-map", "0:v:0", "-frames:v", b - a0,
                                                                "-c:v", "ffv1", "-level", "3", seg])
            worst = max(worst, _attempt(lambda: OPS[step.op](ctx, [str(seg)], step.params, str(raw)), raw, ctx, retries,
                                        f"[{step.id}] {step.op} chunk {c + 1}/{-(-n // per)}", log))
            ffmpeg(["-i", raw, "-map", "0:v:0", "-vf", f"trim=start_frame={a - a0},setpts=PTS-STARTPTS",
                    "-c:v", "ffv1", "-level", "3", part.with_suffix(".part.mkv")])
            part.with_suffix(".part.mkv").replace(part)
            part.with_suffix(".json").write_text(json.dumps(ctx.meta, default=str), encoding="utf-8")
            seg.unlink()
            raw.unlink()
        else:
            log(f"[{step.id}] {step.op} chunk {c + 1}: cached")
        parts.append(part)
        metas.append(json.loads(part.with_suffix(".json").read_text(encoding="utf-8")))
    lst = base.with_name(base.name + "-chunks.txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    ffmpeg(["-f", "concat", "-safe", 0, "-i", lst, "-i", src, "-map", "0:v:0"]
           + ([] if step.op == "mask" else [*audio_maps(src, 1), "-c:a", "flac"]) + ["-c:v", "copy", tmp])
    for p in [*parts, *(p.with_suffix(".json") for p in parts), lst]:
        p.unlink()
    ctx.meta.clear()
    ctx.meta.update(metas[0], chunks=len(parts))
    ms = [m["mask"] for m in metas if "mask" in m]
    if ms:  # frame-weighted mask quality across chunks
        f = sum(m["frames"] for m in ms)
        agg = {k: round(sum(m[k] * m["frames"] for m in ms) / f, 4)
               for k in ("coverage_mean", "temporal_iou_mean", "empty_frame_ratio")}
        ctx.meta["mask"] = ms[0] | agg | {"frames": n, "low_confidence": any(m["low_confidence"] for m in ms)
                                          or agg["temporal_iou_mean"] < 0.8 or agg["empty_frame_ratio"] > 0.1}
    return worst


def run(job: Job, mode="final", profile=None, ws=None, log=print) -> dict:
    """Execute a job. mode = final | preview. Completed steps are cached by content hash,
    so re-running (or resuming an interrupted job) skips finished work."""
    ws = workspace(ws)
    prof_name = "preview" if mode == "preview" else (profile or job.profile)
    if prof_name not in PROFILES:
        raise ValueError(f"unknown profile {prof_name}; have {sorted(PROFILES)}")
    prof = PROFILES[prof_name]
    if not job.steps:
        raise ValueError("job has no steps")

    paths, keys, originals = {}, {}, set()
    for sid, s in job.sources.items():
        p = Path(s.path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"source {sid}: {p}")
        digest = sha256(p)
        if s.sha256 and s.sha256 != digest:
            raise ValueError(f"source {sid} changed since the manifest was written (sha256 mismatch)")
        s.sha256, s.path = digest, str(p)  # saved manifests stay valid from any working directory
        originals.add(p)
        if mode == "preview" and info(p)["video"] and not info(p)["is_image"]:
            paths[sid], keys[sid] = proxy(str(p), digest, job.proxy_height, ws), f"proxy{job.proxy_height}:{digest}"
        else:
            paths[sid], keys[sid] = str(p), digest
    primary = next(iter(job.sources), None)  # None: pure generation job
    pv = info(job.sources[primary].path)["video"] if primary else None
    scale = info(paths[primary])["video"]["height"] / pv["height"] if pv else 1.0

    job.output = str(Path(job.output).resolve())
    out = Path(job.output)
    if mode == "preview":
        out = out.with_name(out.stem + ".preview.mp4")
    if prof.get("ext") and out.suffix.lower() != prof["ext"]:
        out = out.with_suffix(prof["ext"])  # e.g. ProRes can't go in .mp4
    out = out.resolve()
    if out in originals:
        raise ValueError("refusing to overwrite a source file; pick another output path")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Plan every step's cache key up front: graph errors and output conflicts surface before any rendering.
    plan = []
    for step in job.steps:
        if step.op not in OPS:
            raise ValueError(f"step {step.id}: unknown op {step.op}; have {sorted(OPS)}")
        missing = [i for i in step.inputs if i not in keys]
        if missing or step.id in keys:
            raise ValueError(f"step {step.id}: unknown inputs {missing}" if missing else f"duplicate id {step.id}")
        keys[step.id] = _key(op_version(step.op), step.op, step.params, resource_hashes(step.params),
                             [keys[i] for i in step.inputs], round(scale, 6), _worker_id(step.params))
        plan.append(step)
    if OUTPUT_EXT.get(job.steps[-1].op) == ".wav":  # audio-only result: one PCM WAV, audio QA
        prof_name, prof = "wav", PROFILES["wav"]
        out = out.with_suffix(".wav")
        if out in originals:
            raise ValueError("refusing to overwrite a source file; pick another output path")
    fingerprint = _key(keys[job.steps[-1].id], prof_name, prof, job.checks.model_dump(),
                       os.environ.get("STUDIO_HW_ENCODE"), ffmpeg_version())
    reg = _registry(ws)
    if mode == "final" and out.exists():
        rec = json.loads(reg.read_text(encoding="utf-8")).get(str(out)) if reg.exists() else None
        if rec and rec["fingerprint"] == fingerprint and rec["sha256"] == sha256(out) and Path(rec["log"]).exists():
            log(f"{out.name}: identical request already delivered; hash verified")
            return json.loads(Path(rec["log"]).read_text(encoding="utf-8"))["result"] | {"reused": True}
        raise FileExistsError(f"output already exists: {out}; choose a new output name (deliveries are never replaced)")

    ctx = Ctx(workdir=ws / "cache", scale=scale)
    records = []
    from .workers import gpu_workers_in
    with gpu_lock(ws, force=gpu_workers_in(plan)):  # no-op on CPU unless a GPU worker generates
        for step in plan:
            key = keys[step.id]
            dst = ws / "cache" / f"{step.op}-{key[:20]}{OUTPUT_EXT.get(step.op, '.mkv')}"
            side = dst.with_suffix(".json")
            rec = json.loads(side.read_text(encoding="utf-8")) if dst.exists() and side.exists() else None
            # ponytail: size check only; a full hash of a 4K lossless intermediate costs ~10 s per hit
            if rec and rec.get("bytes", dst.stat().st_size) == dst.stat().st_size:
                rec |= {"cached": True}
                os.utime(side)  # mtime = last use, so `studio cache prune` keeps what's still in use
                log(f"[{step.id}] {step.op}: cached")
            else:
                if rec:
                    log(f"[{step.id}] {step.op}: cached output damaged; re-rendering")
                tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
                t0, ins = time.time(), [paths[i] for i in step.inputs]
                ctx.commands, ctx.meta = [], {}
                chunk_s = step.chunk if step.chunk is not None else float(os.environ.get("STUDIO_CHUNK_SECONDS", 60))
                v0 = info(ins[0])["video"] if len(ins) == 1 else None
                if (step.op in CHUNKABLE and v0 and chunk_s > 0 and (v0["duration"] or 0) > chunk_s * (1 if step.chunk else 2)
                        and not (step.params.get("box") or step.params.get("point"))):  # a box is frame-0 state
                    attempt = _chunked(step, ins[0], tmp, ctx, chunk_s, job.retries, log)
                else:
                    attempt = _attempt(lambda: OPS[step.op](ctx, ins, step.params, str(tmp)), tmp, ctx, job.retries,
                                       f"[{step.id}] {step.op}", log)
                tmp.replace(dst)
                rec = {"step": step.id, "op": step.op, "params": step.params,
                       "inputs": {i: keys[i] for i in step.inputs}, "resources": resource_hashes(step.params),
                       "output": str(dst), "sha256": sha256(dst), "bytes": dst.stat().st_size,
                       "seconds": round(time.time() - t0, 2), "attempts": attempt, "meta": ctx.meta,
                       "commands": ctx.commands, "studio": __version__, "op_version": op_version(step.op)}
                side.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                rec["cached"] = False
            paths[step.id] = str(dst)
            records.append(rec)

    if not prof.get("audio_only") and not info(paths[job.steps[-1].id])["video"]:
        raise ValueError("the last step has no video: end audio-only jobs with generate_audio, or mix it into a clip")
    log(f"encoding {out.name} with profile {prof_name}")
    t0 = time.time()
    part = out.with_name(f"{out.stem}.part-{os.getpid()}-{int(time.time())}{out.suffix}")
    if part.resolve() in originals:
        raise ValueError(f"temporary name collides with a source: {part}")
    same_timeline = primary and not any(s.op in FORMAT_CHANGING | RETIMING for s in job.steps)
    final_cmd = encode_final(paths[job.steps[-1].id], str(part), prof, info(paths[primary]) if primary else {},
                             keep_clock=bool(same_timeline))
    encode_s = round(time.time() - t0, 2)

    c = job.checks
    against = c.against or (primary if primary and not any(s.op in FORMAT_CHANGING for s in job.steps) else None)
    if prof.get("audio_only"):
        report = validate_audio(str(part), duration=c.duration, tolerance=c.tolerance, decode=c.decode)
    else:
        report = validate(str(part), against=paths[against] if against else None, duration=c.duration,
                          tolerance=c.tolerance, size=prof.get("size"), ssim=c.ssim, decode=c.decode)
    if mode == "final" and report["ok"]:
        publish(part, out)  # never replaces a file that appeared during the encode
    else:  # previews iterate in place; a failed final is kept for inspection, never at the delivery name
        if mode == "final":
            out = out.with_name(f"{out.stem}.failed-{datetime.now():%Y%m%d-%H%M%S}{out.suffix}")
        if out in originals:
            raise ValueError(f"refusing to overwrite a source file: {out}")
        part.replace(out)
    report["file"] = str(out)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = ws / "manifests" / f"{job.name}-{stamp}.json"
    manifest.write_text(job.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
    result = {"ok": report["ok"], "output": str(out), "sha256": sha256(out), "mode": mode, "profile": prof_name,
              "steps": [{k: r.get(k) for k in ("step", "op", "cached", "seconds", "meta", "output")} for r in records],
              "encode_seconds": encode_s, "validation": report, "manifest": str(manifest),
              "ffmpeg": tool("ffmpeg")}
    logf = ws / "logs" / f"{job.name}-{stamp}.json"
    result["log"] = str(logf)
    logf.write_text(json.dumps(result | {"result": result, "fingerprint": fingerprint, "step_records": records,
                                         "final_command": final_cmd}, indent=2, default=str), encoding="utf-8")
    if mode == "final" and report["ok"]:
        entries = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {}
        entries[str(out)] = {"fingerprint": fingerprint, "sha256": result["sha256"], "log": str(logf)}
        tmpreg = reg.with_name(f"outputs.{os.getpid()}.tmp")
        tmpreg.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        tmpreg.replace(reg)  # ponytail: read-modify-write without a lock; two finals at once may drop an entry
    return result


def prune(days=None, ws=None) -> dict:
    """Delete cached step outputs and proxies unused for `days` (None = everything). Never touches sources."""
    ws = workspace(ws)
    cutoff = time.time() - days * 86400 if days is not None else float("inf")
    freed, files = 0, 0
    for f in [*ws.glob("cache/*"), *ws.glob("proxies/*")]:
        if f.is_file() and f.stat().st_mtime < cutoff:
            side = f.with_suffix(".json")  # a step output's age is its sidecar's (touched on every cache hit)
            if f.suffix == ".mkv" and side.exists() and side.stat().st_mtime >= cutoff:
                continue
            freed += f.stat().st_size
            files += 1
            f.unlink()
    return {"files": files, "freed_gb": round(freed / 1e9, 3)}
