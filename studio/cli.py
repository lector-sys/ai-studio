"""`studio` command: the stable contract the coding agent calls. All output is JSON on stdout;
progress goes to stderr; exit code 1 means the operation or its validation failed."""
import json
import sys
from pathlib import Path

import typer

from . import jobs, media
from .jobs import Job

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
P = typer.Option([], "-p", "--param", help="key=value (value parsed as JSON when possible)")


def _out(obj, ok=True):
    print(json.dumps(obj, indent=2, default=str))
    raise typer.Exit(0 if ok else 1)


def _log(msg):
    print(msg, file=sys.stderr)


def _params(pairs) -> dict:
    out = {}
    for kv in pairs:
        k, _, v = kv.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _nums(s):
    return [float(x) for x in s.split(",")] if s else None


def _run(sources, steps, output, profile, preview, validate=None):
    job = Job(name=Path(output).stem, sources={f"s{i}": {"path": str(p)} for i, p in enumerate(sources)},
              steps=steps, output=str(output), profile=profile, validate=validate or {})
    r = jobs.run(job, mode="preview" if preview else "final", log=_log)
    _out(r, r["ok"])


@app.command()
def inspect(path: Path, keyframes: bool = False, hash: bool = False, json_: bool = typer.Option(True, "--json")):
    """Codecs, streams, resolution, fps, duration, HDR/color, audio (+ keyframes/hash on request)."""
    d = media.info(path)
    if keyframes:
        d["keyframes"] = media.keyframes(path)
    if hash:
        d["sha256"] = media.sha256(path)
    _out(d)


@app.command()
def devices():
    """Discover accelerators (ffmpeg hwaccels/encoders, ONNX providers, torch, GPUs)."""
    from .devices import discover
    _out(discover())


@app.command()
def ops():
    """List every operation with its parameters (the capability registry)."""
    from .ops import OPS
    _out({k: " ".join((f.__doc__ or "").split()) for k, f in sorted(OPS.items())})


@app.command()
def profiles():
    """List output profiles."""
    _out({k: v["desc"] for k, v in jobs.PROFILES.items()})


@app.command()
def edit(op: str, inputs: list[Path], output: Path = typer.Option(..., "-o"), param: list[str] = P,
         profile: str = "4k60-master", preview: bool = False, duration: float = None):
    """One deterministic op, e.g. `studio edit trim in.mp4 -o out.mp4 -p start=1 -p end=4`.
    Multi-input ops (concat, transition, overlay, audio, chromakey) take extra inputs in order."""
    step = {"id": op, "op": op, "inputs": [f"s{i}" for i in range(len(inputs))], "params": _params(param)}
    _run(inputs, [step], output, profile, preview, {"duration": duration})


def _alias(op):
    def cmd(inputs: list[Path], output: Path = typer.Option(..., "-o"), param: list[str] = P,
            profile: str = "4k60-master", preview: bool = False):
        step = {"id": op, "op": op, "inputs": [f"s{i}" for i in range(len(inputs))], "params": _params(param)}
        _run(inputs, [step], output, profile, preview)
    cmd.__doc__ = f"Shortcut for `studio edit {op}`."
    return cmd


app.command("color")(_alias("color"))
app.command("audio")(_alias("audio"))


@app.command()
def mask(target: str, input: Path, output: Path = typer.Option(..., "-o"), box: str = None, point: str = None,
         param: list[str] = P, preview: bool = False):
    """Segmentation matte video. target = person | salient | object (object needs --box x,y,w,h or
    --point x,y in source px). Prints mask quality stats (temporal IoU, empty frames, low_confidence)."""
    p = _params(param) | {"target": target}
    p |= {"box": _nums(box)} if box else {}
    p |= {"point": _nums(point)} if point else {}
    _run([input], [{"id": "mask", "op": "mask", "inputs": ["s0"], "params": p}], output, "archive", preview)


@app.command()
def track(input: Path, box: str = typer.Option(..., help="x,y,w,h in source px on frame 0"),
          output: Path = typer.Option(None, "-o"), height: int = 512):
    """Track a box through the clip (optical flow); JSON list of per-frame [x,y,w,h]."""
    from .vision import track as _track
    boxes = _track(str(input), _nums(box), height)
    if output:
        output.write_text(json.dumps(boxes))
    _out({"frames": len(boxes), "first": boxes[0] if boxes else None, "last": boxes[-1] if boxes else None,
          "file": str(output) if output else None, **({} if output else {"boxes": boxes})})


@app.command()
def background(mode: str, input: Path, output: Path = typer.Option(..., "-o"), bg: Path = None,
               mask_file: Path = typer.Option(None, "--mask", help="reuse an existing mask video"),
               target: str = "person", box: str = None, param: list[str] = P, profile: str = None,
               preview: bool = False):
    """Background remove | blur | replace. Builds mask (unless --mask) then composites at source res.
    `-p mask.model=birefnet-portrait -p mask.stride=2` tunes the mask step; other -p go to compositing."""
    params = _params(param)
    mp = {k[5:]: v for k, v in params.items() if k.startswith("mask.")} | {"target": target}
    mp |= {"box": _nums(box)} if box else {}
    bp = {k: v for k, v in params.items() if not k.startswith("mask.")} | {"mode": mode}
    sources, steps = [input], []
    if mask_file:
        sources.append(mask_file)
        mask_id = "s1"
    else:
        steps.append({"id": "mask", "op": "mask", "inputs": ["s0"], "params": mp})
        mask_id = "mask"
    ins = ["s0", mask_id]
    if bg:
        sources.append(bg)
        ins.append(f"s{len(sources) - 1}")
    steps.append({"id": "bg", "op": "background", "inputs": ins, "params": bp})
    prof = profile or ("alpha-master" if mode == "remove" else "4k60-master")
    _run(sources, steps, output, prof, preview, {"against": "s0"})


@app.command()
def preview(job: Path):
    """Low-cost proxy render of a job manifest (outputs <output>.preview.mp4)."""
    r = jobs.run(jobs.load(job), mode="preview", log=_log)
    _out(r, r["ok"])


@app.command()
def render(job: Path, profile: str = None):
    """Final render of a job manifest against master media."""
    r = jobs.run(jobs.load(job), profile=profile, log=_log)
    _out(r, r["ok"])


@app.command()
def validate(output: Path, against: Path = None, duration: float = None, size: str = None, fps: float = None,
             ssim: bool = False, no_decode: bool = False):
    """Technical QA of a render (optionally against its source)."""
    from .qc import validate as _validate
    r = _validate(str(output), str(against) if against else None, duration, None,
                  [int(x) for x in size.split("x")] if size else None, fps, ssim, not no_decode)
    _out(r, r["ok"])


# --- Phase 3: intelligent editorial ----------------------------------------------------------------

def _ranges(s):
    """'1.2-3.4,5-6' -> [[1.2, 3.4], [5, 6]]"""
    return [[float(x) for x in r.split("-", 1)] for r in s.split(",")] if s else []


@app.command()
def transcribe(input: Path, output: Path = typer.Option(None, "-o", help=".json (default: print)"),
               model: str = "base", language: str = None, translate: bool = False):
    """Local STT with word timestamps (faster-whisper; --translate gives English)."""
    from .speech import transcribe as _t
    t = _t(str(input), model, language, "translate" if translate else "transcribe", jobs.workspace() / "cache")
    if output:
        output.write_text(json.dumps(t, indent=1), encoding="utf-8")
        _out({"file": str(output), "language": t["language"], "segments": len(t["segments"]), "model": t["model"]})
    _out(t)


@app.command()
def captions(input: Path, output: Path = typer.Option(..., "-o", help=".srt/.vtt/.json sidecar, or a video to burn into"),
             model: str = "base", language: str = None, translate: bool = False, max_chars: int = 42,
             style: str = None, profile: str = "4k60-master", preview: bool = False):
    """Captions: sidecar .srt/.vtt (or .json transcript), or burned in when -o is a video."""
    from . import speech
    task = "translate" if translate else "transcribe"
    if output.suffix.lower() in (".srt", ".vtt", ".json"):
        t = speech.transcribe(str(input), model, language, task, jobs.workspace() / "cache")
        fmt = {".srt": speech.to_srt, ".vtt": speech.to_vtt}.get(output.suffix.lower())
        output.write_text(fmt(t, max_chars=max_chars) if fmt else json.dumps(t, indent=1), encoding="utf-8")
        _out({"file": str(output), "cues": len(speech.cues(t, max_chars=max_chars)), "language": t["language"]})
    p = {"model": model, "language": language, "task": task, "max_chars": max_chars} | ({"style": style} if style else {})
    _run([input], [{"id": "captions", "op": "captions", "inputs": ["s0"], "params": p}], output, profile, preview)


@app.command()
def cut(input: Path, output: Path = typer.Option(..., "-o"), remove: str = typer.Option(None, help="1.2-3.4,5-6"),
        keep: str = typer.Option(None, help="ranges to keep instead"), silence: bool = False,
        min_silence: float = 0.5, noise: float = -35, fillers: bool = False,
        phrase: list[str] = typer.Option([], help="transcript phrase to remove (repeatable)"),
        model: str = "base", profile: str = "4k60-master", preview: bool = False):
    """Ripple-delete ranges: explicit, silences, filler words, or transcript phrases."""
    p = {"remove": _ranges(remove), "keep": _ranges(keep), "phrases": phrase, "fillers": fillers, "model": model,
         "silence": {"noise": noise, "min": min_silence} if silence else False}
    _run([input], [{"id": "cut", "op": "cut", "inputs": ["s0"], "params": p}], output, profile, preview)


@app.command()
def index(folder: Path, model: str = "base", no_transcribe: bool = False):
    """Index a media folder (metadata, scene cuts + thumbnails, transcripts) for `studio search`."""
    from .speech import index as _index
    _out(_index(folder, jobs.workspace(), model, not no_transcribe, _log))


@app.command()
def search(query: str, limit: int = 20):
    """Search indexed media by spoken words / file names; returns file + timecode hits."""
    from .speech import search as _search
    _out({"query": query, "hits": _search(query, jobs.workspace(), limit)})


# --- Phase 5: object + generative ------------------------------------------------------------------

@app.command("object")
def object_(action: str, input: Path, output: Path = typer.Option(..., "-o"), box: str = None, point: str = None,
            with_: Path = typer.Option(None, "--with", help="replacement/insert image or video"),
            method: str = "plate", backend: str = typer.Option(None, help="mask backend: sam | grabcut"),
            track: bool = True, param: list[str] = P, profile: str = "4k60-master", preview: bool = False):
    """Object remove | replace | insert. Box/point are source px on frame 0. remove = mask + inpaint
    (clean plate); replace = remove + tracked insert of --with; insert = tracked insert of --with."""
    if action not in ("remove", "replace", "insert"):
        raise ValueError("action must be remove | replace | insert")
    if action != "insert" and not (box or point):
        raise ValueError(f"object {action} needs --box x,y,w,h or --point x,y")
    if action != "remove" and not (with_ and box):
        raise ValueError(f"object {action} needs --with <image/video> and --box x,y,w,h")
    sources, steps, cur = [input], [], "s0"
    if action != "insert":
        mp = {"target": "object"} | ({"box": _nums(box)} if box else {"point": _nums(point)})
        mp |= {"backend": backend} if backend else {}
        steps += [{"id": "mask", "op": "mask", "inputs": ["s0"], "params": mp},
                  {"id": "remove", "op": "inpaint", "inputs": ["s0", "mask"],
                   "params": {"method": method} | _params(param)}]
        cur = "remove"
    if with_ and action != "remove":
        sources.append(with_)
        steps.append({"id": "insert", "op": "insert", "inputs": [cur, "s1"],
                      "params": {"box": _nums(box), "track": track}})
    _run(sources, steps, output, profile, preview, {"against": "s0"})


@app.command()
def generate(kind: str, prompt: str = typer.Option("", "--prompt", help="prompt, or the narration text for voice"),
             output: Path = typer.Option(..., "-o"), backend: str = None,
             worker: str = typer.Option(None, help="configured local worker (`studio workers`)"),
             init: Path = None, reference: list[str] = typer.Option([], help="name=path reference image (repeatable)"),
             text: str = typer.Option(None, help="voice: text to speak (default: --prompt)"), voice: str = None,
             width: int = 1920, height: int = 1080,
             duration: float = typer.Option(None, help="s; image/video default 5; voice: measured from synthesis"),
             fps: str = "30", seed: int = None, profile: str = "h264-master"):
    """Generate an image, video, voice, music or sfx (procedural | command | comfyui | configured worker).
    Never replaces an existing file; provenance is written beside the output as <file>.gen.json."""
    from .generate import AUDIO_KINDS, generate_media, provenance
    if kind not in ("image", "video", *AUDIO_KINDS):
        raise ValueError("kind must be image | video | voice | music | sfx")
    refs = dict(r.split("=", 1) for r in reference)
    if kind != "video":
        res = generate_media(prompt, kind, output.with_suffix(".png" if kind == "image" else ".wav"), width, height,
                             duration, fps, backend, seed, str(init) if init else None, references=refs, worker=worker,
                             voice=voice, text=text or (prompt if kind == "voice" else None))
        _out({"file": res, "provenance": provenance(res)})
    if refs:
        raise ValueError("video references go through a configured worker in a job manifest (generate step)")
    p = {"prompt": prompt, "kind": "video", "width": width, "height": height, "duration": duration or 5, "fps": fps,
         "seed": seed, "backend": backend} | ({"worker": worker} if worker else {})
    _run([init] if init else [], [{"id": "gen", "op": "generate", "inputs": ["s0"] if init else [], "params": p}],
         output, profile, False, {"duration": duration or 5})


@app.command()
def workers(action: str = typer.Argument("list", help="list | smoke | review"), name: str = typer.Argument(None),
            kind: str = None, prompt: str = typer.Option("", "--prompt"), text: str = None, voice: str = None,
            duration: float = None, width: int = 1024, height: int = 1024, fps: str = "16", seed: int = 7,
            reference: list[str] = typer.Option([], help="name=path"), reviewer: str = None,
            status: str = typer.Option(None, help="review: pass | fail | needs-work"),
            finding: list[str] = typer.Option([], help="review finding (repeatable)")):
    """Local generation workers. `list`: state per worker (unavailable + reason / configured / ready).
    `smoke NAME --kind K --prompt ...`: one real, validated output recorded as evidence (review pending).
    `review NAME --kind K --reviewer R --status pass`: record the content review that makes it ready."""
    from datetime import datetime

    from . import workers as W
    if action == "list":
        _out(W.listing())
    if not name:
        raise ValueError(f"workers {action} needs a worker name")
    cfg = W.worker_config(name)
    kind = kind or cfg["kinds"][0]
    if action == "review":
        if not (reviewer and status):
            raise ValueError("review needs --reviewer and --status")
        _out(W.record_review(cfg, kind, reviewer, status, finding))
    if action != "smoke":
        raise ValueError("action must be list | smoke | review")
    from .generate import AUDIO_KINDS, generate_media, provenance
    d = jobs.workspace() / "workers" / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}-{kind}-{datetime.now():%Y%m%d-%H%M%S}"
    res = generate_media(prompt, kind, out, width, height, duration if kind != "video" else (duration or 2), fps,
                         seed=seed, references=dict(r.split("=", 1) for r in reference), worker=name, voice=voice,
                         text=text or (prompt if kind == "voice" else None))
    prov = provenance(res)
    ev = W.record_smoke(cfg, prov | {"kind": kind})
    _out({"output": res, "evidence": ev, "next": f"review it, then: studio workers review {name} --kind {kind} "
                                                 "--reviewer <who> --status pass|fail|needs-work --finding ..."})


# --- Creative recipes ------------------------------------------------------------------------------

@app.command()
def recipes():
    """List the creative recipes (inputs, capabilities, steps they produce)."""
    from .recipes import listing
    _out(listing())


@app.command()
def recipe(name: str, request: Path = typer.Option(..., "--request", help="request .json"),
           output: Path = typer.Option(..., "-o", help="manifest .json to write"),
           assets: Path = typer.Option(None, help="card/caption asset folder (default: <manifest>.assets)")):
    """Expand a recipe request into a checked job manifest (renders only static card assets)."""
    from .agent import check as _check
    from .recipes import expand
    job = expand(name, json.loads(request.read_text(encoding="utf-8")),
                 assets or output.with_name(output.stem + ".assets"))
    c = _check(job)
    if c["ok"]:
        tmp = output.with_name(output.name + ".part")
        tmp.write_text(job.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
        tmp.replace(output)
    _out({"manifest": str(output) if c["ok"] else None, "steps": [s.op for s in job.steps], "check": c}, c["ok"])


@app.command()
def review(log: Path, output: Path = typer.Option(..., "-o", help="review bundle folder")):
    """Prepare shared review evidence from a render log (frames, crops, checks, viewing copy, audio)."""
    from .review import bundle
    rec = json.loads(log.read_text(encoding="utf-8"))
    b = bundle(jobs.load(rec["manifest"]), rec.get("result", rec), output)
    _out(b, b["technical_ok"])


@app.command("review-record")
def review_record(bundle_dir: Path, kind: str, reviewer: str = typer.Option(...),
                  status: str = typer.Option(..., help="pass | fail | needs-work"),
                  finding: list[str] = typer.Option([], help="one finding (repeatable)")):
    """Record a delegated/human review result in a bundle's reviews.json."""
    from .review import record
    _out(record(bundle_dir, kind, reviewer, status, finding))


# --- Phase 6: agent reliability --------------------------------------------------------------------

@app.command()
def capabilities():
    """Routing table: which backends work on this machine, plus every op's contract."""
    from .devices import capabilities as _c
    _out(_c())


@app.command()
def schema():
    """JSON schema of the job manifest (what a planner must emit)."""
    _out(Job.model_json_schema(by_alias=True))


@app.command()
def check(job: Path):
    """Dry-run a manifest: graph errors, unknown params, models, VRAM/disk/frame estimates."""
    from .agent import check as _check
    r = _check(jobs.load(job))
    _out(r, r["ok"])


@app.command()
def auto(job: Path, max_rounds: int = 3):
    """Preview -> detect low-confidence/failed steps -> refine (model ladder, stride) -> final render."""
    from .agent import auto as _auto
    r = _auto(jobs.load(job), max_rounds, _log)
    _out(r, r["ok"])


@app.command()
def bench(size: str = "1920x1080", seconds: float = 2, ai: bool = False):
    """Benchmark every op on a synthetic clip (results saved to workspace/bench)."""
    from .agent import bench as _bench
    w, h = (int(x) for x in size.split("x"))
    _out(_bench(w, h, seconds, ai, _log))


@app.command("models")
def models_(action: str = typer.Argument("list", help="list | download"), name: list[str] = typer.Argument(None)):
    """Model registry: `studio models` lists named models; `studio models download gfpgan_1.4 rife_4.9`
    fetches them ahead of time (otherwise they download on first use)."""
    from . import models
    if action == "download":
        _out({n: models.path(n) for n in (name or list(models.MODELS))})
    _out(models.listing())


@app.command()
def cache(action: str = typer.Argument(..., help="prune"), older_than: float = typer.Option(
        None, help="days unused; omit to clear everything")):
    """Free disk: `studio cache prune --older-than 7` deletes step outputs/proxies unused for 7 days."""
    if action != "prune":
        raise ValueError("action must be prune")
    _out(jobs.prune(older_than))


# --- Phase 7: optional UI --------------------------------------------------------------------------

@app.command()
def ui(port: int = 8765, no_browser: bool = False):
    """Local web UI (127.0.0.1 only): jobs, preview/render, frame box picker, transcript, logs."""
    from .ui import serve
    serve(port, not no_browser)


def main():
    try:
        app()
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
