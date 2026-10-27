"""Agent reliability: static plan checks + resource estimates, the automatic preview -> refine ->
render loop, capability routing, and a benchmark suite."""
import copy
import inspect
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from . import jobs
from .jobs import FORMAT_CHANGING, PROFILES, Job
from .media import ffmpeg, info
from .ops import OPS
from .sched import VRAM_MB, budget, gpu_active

AI_OPS = {"mask", "cut", "captions", "inpaint", "generate", "generate_audio", "face_restore"}  # may load models (cut: only for fillers/phrases)
MODEL_LADDER = {"person": ["u2net_human_seg", "birefnet-portrait", "isnet-general-use"],
                "salient": ["isnet-general-use", "birefnet-general", "u2net"],
                "object": ["sam"]}


def _doc_params(fn) -> set:
    """Words in the op's documented params section (the docstring is the registry contract).
    Over-inclusive on purpose: it only drives unknown-param warnings."""
    doc = " ".join((fn.__doc__ or "").split())
    return set(re.findall(r"[A-Za-z_]\w*", doc.split("params:", 1)[1] if "params:" in doc else ""))


def check(job: Job) -> dict:
    """Dry-run a manifest: graph/op/source errors, param warnings, and cost estimates. Nothing renders."""
    errors, warnings, ids = [], [], set(job.sources)
    frames = px = 0
    audio_only = set()  # ids whose media has no picture (audio files, generate_audio steps)
    for sid, s in job.sources.items():
        if not Path(s.path).exists():
            errors.append(f"source {sid} missing: {s.path}")
            continue
        v = info(s.path)["video"]
        if not v:
            audio_only.add(sid)
        if v and not info(s.path)["is_image"]:
            frames = max(frames, int((v["duration"] or 0) * (v["fps"] or 0)))
            px = max(px, v["width"] * v["height"])
            if v["vfr"] and any(st.op in ("mask", "inpaint", "insert") for st in job.steps):
                warnings.append(f"source {sid} is VFR: conform with an `fps` step before masks/tracking")
    if job.profile not in PROFILES:
        errors.append(f"unknown profile {job.profile}")
    models = []
    for st in job.steps:
        if st.op not in OPS:
            errors.append(f"step {st.id}: unknown op {st.op}")
            continue
        if st.id in ids:
            errors.append(f"duplicate id {st.id}")
        for i in st.inputs:
            if i not in ids:
                errors.append(f"step {st.id}: input {i} is not a source or an earlier step")
        ids.add(st.id)
        if st.op == "generate_audio":
            audio_only.add(st.id)
        fed = [i for n, i in enumerate(st.inputs) if i in audio_only and not (st.op == "audio" and n == 1)]
        if fed:
            errors.append(f"step {st.id} ({st.op}): needs video but {fed} is audio-only "
                          "(mix generated audio into a clip with an `audio` step: inputs [video, audio])")
        if st.params.get("worker"):
            from .workers import readiness, worker_config
            try:
                r = readiness(worker_config(st.params["worker"]))
            except ValueError as e:
                errors.append(f"step {st.id}: {e}")
            else:
                kind = st.params.get("kind", "video" if st.op == "generate" else "voice")
                k = r["kinds"].get(kind, {})
                if r["state"] == "unavailable":
                    errors.append(f"step {st.id}: worker {st.params['worker']} unavailable: {r['reason']}")
                elif k.get("state") != "ready":
                    warnings.append(f"step {st.id}: worker {st.params['worker']} {kind} is configured but unverified "
                                    f"({k.get('why')}); run `studio workers smoke` and record a review")
        known = _doc_params(OPS[st.op])
        unknown = [k for k in st.params if k not in known]
        if unknown:
            warnings.append(f"step {st.id} ({st.op}): params not in the op's docs: {unknown}")
        if st.op == "compose" and all(i in job.sources for i in st.inputs):
            from .compose import validate_compose
            try:  # colour/format/geometry preflight; inputs made by earlier steps are checked at render
                validate_compose([job.sources[i].path for i in st.inputs], st.params)
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                errors.append(f"step {st.id} (compose): {e}")
        if st.op == "mask":
            t = st.params.get("target", "person")
            models.append("grabcut" if st.params.get("backend") == "grabcut" else
                          f"rembg:{st.params.get('model') or MODEL_LADDER.get(t, ['sam'])[0]}")
        if st.op in ("captions",) or (st.op == "cut" and (st.params.get("fillers") or st.params.get("phrases"))):
            models.append(f"whisper:{st.params.get('model', 'base')}")
        if st.op == "face_restore":
            models += ["onnx:yunet", f"onnx:{st.params.get('model', 'gfpgan_1.4')}"]
        if st.params.get("backend") == "onnx":
            default = {"interpolate": "rife_4.9", "upscale": "real_esrgan_x2"}.get(st.op)
            models.append(f"onnx:{Path(str(st.params.get('model') or default)).name}")
    if Path(job.output).exists():
        warnings.append(f"output {job.output} exists: a different final render to it will be refused "
                        "(deliveries are never replaced); choose a new name")
    alpha = any(st.op == "background" and st.params.get("mode") == "remove" for st in job.steps[-1:])
    if alpha and not PROFILES.get(job.profile, {}).get("alpha"):
        errors.append("last step outputs alpha: use profile alpha-master or archive")
    vram = sum(VRAM_MB.get(m, VRAM_MB.get(m.split(":")[0], 0)) for m in set(models) if m != "grabcut")
    # lossless intermediates are ~1.5 bytes/px/frame at qp0 for camera footage (rough)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "steps": len(job.steps),
            "ai_steps": [s.id for s in job.steps if s.op in AI_OPS], "models": sorted(set(models)),
            "estimate": {"frames": frames, "megapixels": round(px / 1e6, 2),
                         "cache_gb_per_step": round(frames * px * 1.5 / 1e9, 2),
                         "vram_mb": vram if gpu_active() else 0, "vram_budget_mb": budget() if gpu_active() else None,
                         "device": "gpu" if gpu_active() else "cpu",
                         "format_changing": [s.id for s in job.steps if s.op in FORMAT_CHANGING]}}


def _refine(job: Job, result: dict) -> list[str]:
    """Adjust low-confidence mask steps in place; returns what changed (empty = nothing left to try)."""
    changes = []
    by_id = {s.id: s for s in job.steps}
    for r in result["steps"]:
        m = (r.get("meta") or {}).get("mask")
        if not m or not m.get("low_confidence"):
            continue
        st = by_id[r["step"]]
        p = st.params
        if int(p.get("stride", 1)) > 1:
            p["stride"] = 1
            changes.append(f"{st.id}: stride -> 1")
            continue
        ladder = MODEL_LADDER.get(p.get("target", "person"), [])
        cur = p.get("model") or (ladder[0] if ladder else None)
        nxt = ladder[ladder.index(cur) + 1] if cur in ladder and ladder.index(cur) + 1 < len(ladder) else None
        if nxt:
            p["model"] = nxt
            changes.append(f"{st.id}: model {cur} -> {nxt}")
        elif float(p.get("smooth", 0.3)) < 0.6:
            p["smooth"] = 0.6
            changes.append(f"{st.id}: smooth -> 0.6")
    return changes


def auto(job: Job, max_rounds=3, log=print, ws=None) -> dict:
    """Preview, detect failed/low-confidence steps, refine and retry, then render + validate the master."""
    pre = check(job)
    if not pre["ok"]:
        return {"ok": False, "stage": "check", "check": pre}
    job = copy.deepcopy(job)
    rounds = []
    for n in range(1, max_rounds + 1):
        log(f"auto: preview round {n}")
        r = jobs.run(copy.deepcopy(job), mode="preview", ws=ws, log=log)
        low = [s["step"] for s in r["steps"] if ((s.get("meta") or {}).get("mask") or {}).get("low_confidence")]
        rounds.append({"round": n, "preview": r["output"], "ok": r["ok"], "low_confidence": low,
                       "failed_checks": [c for c in r["validation"]["checks"] if not c["ok"]]})
        if not r["validation"]["ok"]:
            return {"ok": False, "stage": "preview", "rounds": rounds, "preview": r}
        if not low:
            break
        changes = _refine(job, r)
        rounds[-1]["changes"] = changes
        if not changes:
            log("auto: nothing left to refine; rendering best effort (flagged)")
            break
    refined = Path(job.output).with_suffix(".refined.json")
    refined.write_text(job.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
    log("auto: final render")
    final = jobs.run(job, ws=ws, log=log)
    still_low = [s["step"] for s in final["steps"] if ((s.get("meta") or {}).get("mask") or {}).get("low_confidence")]
    return {"ok": final["ok"] and not still_low, "stage": "final", "rounds": rounds, "manifest": str(refined),
            "low_confidence": still_low, "final": final}


BENCH = [("trim", {"start": 0.5, "end": 1.5}), ("resize", {"width": 1280, "height": 720}),
         ("speed", {"factor": 2}), ("color", {"contrast": 1.1, "temperature": 5600}), ("fade", {"in": 0.5}),
         ("denoise", {}), ("sharpen", {}), ("stabilize", {}), ("upscale", {"factor": 2}),
         ("interpolate", {"fps": 60, "backend": "blend"}), ("stylize", {"style": "cinematic"}),
         ("extend", {"mode": "canvas", "width": 1080, "height": 1920}), ("loudnorm", {})]


def bench(width=1920, height=1080, seconds=2, ai=False, log=print) -> dict:
    """Time every op on a synthetic clip in a throwaway workspace. Results -> workspace/bench/."""
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        src = t / "bench.mp4"
        ffmpeg(["-f", "lavfi", "-i", f"testsrc2=s={width}x{height}:r=30", "-f", "lavfi", "-i",
                f"sine=f=440:d={seconds}", "-t", seconds, "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", src])
        cases = BENCH + ([("mask", {"target": "person", "stride": 2})] if ai else [])
        rows = []
        for op, p in cases:
            job = Job(name=f"bench-{op}", sources={"s": {"path": str(src)}},
                      steps=[{"id": op, "op": op, "inputs": ["s"], "params": p}],
                      output=str(t / f"{op}.mkv"), profile="archive", validate={"decode": False})
            t0 = time.time()
            try:
                r = jobs.run(job, ws=t / "ws", log=lambda m: None)
                ok, step_s = r["ok"], r["steps"][0]["seconds"]
            except Exception as e:  # noqa: BLE001 - report, keep benchmarking
                ok, step_s = False, None
                log(f"bench {op}: {e}")
            rows.append({"op": op, "ok": ok, "step_seconds": step_s, "total_seconds": round(time.time() - t0, 2),
                         "realtime_x": round(seconds / step_s, 2) if step_s else None})
            log(f"bench {op}: {rows[-1]}")
    out = {"clip": f"{width}x{height}@30 {seconds}s", "device": "gpu" if gpu_active() else "cpu",
           "cpu_threads": os.cpu_count(), "results": rows}
    f = jobs.workspace() / "bench" / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out | {"file": str(f)}


def op_signature(name) -> dict:
    fn = OPS[name]
    return {"doc": " ".join((fn.__doc__ or "").split()), "module": inspect.getmodule(fn).__name__.split(".")[-1],
            "ai": name in AI_OPS}
