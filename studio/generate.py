"""Generative editing: object removal (inpaint / clean plate), tracked insert/replace, masked
relighting, stylization, background/shot extension, and pluggable image/video generators.

Generators are adapters behind one function, so any local model plugs in without contract changes:
  procedural - built-in ffmpeg gradients/noise plates (always available, deterministic by seed; image/video only)
  command    - a configured worker (studio/workers.json or workspace/workers.json: argv with {request} {out})
               or legacy STUDIO_GEN_CMD, e.g. "python my_gen.py --prompt {prompt} --out {out}"
  comfyui    - a local ComfyUI server: a configured worker (workflow, bindings, uploads, designated output
               node) or legacy STUDIO_COMFYUI_URL + STUDIO_COMFYUI_WORKFLOW with {{prompt}}/... placeholders
"""
import json
import os
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
import warnings
from hashlib import sha256 as _h
from pathlib import Path

import cv2
import numpy as np

from .media import IMAGE_EXTS, ffmpeg, fpath, info, read_frames
from .ops import grade, has_audio, op
from .restore import map_frames, onnx_image


def _mask_iter(path, W, H):
    """Mask frames (bool, master res); repeats the last one if the mask is a frame short."""
    last = np.zeros((H, W), bool)
    for f in read_frames(path, W, H):
        last = f[..., 0] > 127
        yield last
    while True:
        yield last


def _plate(src, mask, W, H, samples=15):
    """Clean plate: per-pixel median of the frames where that pixel is not masked (static camera)."""
    v = info(src)["video"]
    n = v["frames"] or int((v["duration"] or 1) * (v["fps"] or 30))
    every = max(1, n // samples)
    fs, ms = [], []
    for i, (f, m) in enumerate(zip(read_frames(src, W, H), _mask_iter(mask, W, H))):
        if i % every == 0:
            fs.append(f)
            ms.append(m)
    fs, ms = np.stack(fs), np.stack(ms)
    plate, seen = np.zeros((H, W, 3), np.uint8), ~ms.all(0)
    for y in range(0, H, 64):  # row chunks keep 4K memory bounded
        x = fs[:, y:y + 64].astype(np.float32)
        x[ms[:, y:y + 64]] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns = never-visible pixels
            plate[y:y + 64] = np.nan_to_num(np.nanmedian(x, 0)).astype(np.uint8)
    return plate, seen


@op("inpaint")
def inpaint(ctx, ins, p, out):
    """Object removal. inputs: video, mask (white = remove; from `mask target=object`). params:
    method = plate (clean plate from other frames, best for a static camera; falls back to telea where
    never visible) | telea | ns (spatial inpaint); dilate (source px, 8); radius (px, 5)."""
    v = info(ins[0])["video"]
    W, H = v["width"], v["height"]
    method, r = p.get("method", "plate"), int(p.get("radius", 5))
    k = max(1, ctx.px(p.get("dilate", 8)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    plate, seen = _plate(ins[0], ins[1], W, H) if method == "plate" else (None, None)
    masks, flag = _mask_iter(ins[1], W, H), cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    area = []

    def fix(f, i):
        m = cv2.dilate(next(masks).astype(np.uint8) * 255, kern)
        area.append(float((m > 0).mean()))
        if not m.any():
            return f
        f = f.copy()
        if plate is not None:
            use = (m > 0) & seen
            f[use] = plate[use]
            m = np.where(seen, 0, m).astype(np.uint8)
            if not m.any():
                return f
        ys, xs = np.nonzero(m)
        pad = 4 * r + k
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        f[y0:y1, x0:x1] = cv2.inpaint(f[y0:y1, x0:x1], m[y0:y1, x0:x1], r, flag)
        return f

    map_frames(ctx, ins[0], out, fix)
    ctx.meta["inpaint"] = {"method": method, "mask_area_mean": round(float(np.mean(area)), 4),
                           "plate_coverage": round(float(seen.mean()), 4) if seen is not None else None}


@op("insert")
def insert(ctx, ins, p, out):
    """Tracked insert / replacement element. inputs: video, layer (image or video; alpha respected).
    params: box [x, y, w, h] placement on frame 0 (source px); track (follow the surface inside the
    box via optical flow, default true); opacity (0-1); start, end (s)."""
    from .vision import track
    v = info(ins[0])["video"]
    box = p["box"]
    boxes = track(ins[0], box, scale=ctx.scale) if p.get("track", True) else [box]
    cmd = ctx.workdir / f"insert-{_h(json.dumps(boxes).encode()).hexdigest()[:16]}.cmd"
    fps = v["fps"] or 30
    cmd.write_text("".join(f"{i / fps:.5f} overlay@ins x {ctx.px(b[0])}, overlay@ins y {ctx.px(b[1])};\n"
                           for i, b in enumerate(boxes)), encoding="utf-8")
    img = info(ins[1])["is_image"]
    lay = f"[1:v]format=rgba,scale={ctx.even(box[2])}:{ctx.even(box[3])}"
    if float(p.get("opacity", 1)) < 1:
        lay += f",colorchannelmixer=aa={p['opacity']}"
    en = ""
    if p.get("start") is not None or p.get("end") is not None:
        en = f":enable='between(t,{p.get('start') or 0},{p.get('end') if p.get('end') is not None else 1e9})'"
    fc = (f"{lay}[ov];[0:v]sendcmd=f={fpath(cmd)}[m];[m][ov]overlay@ins=x={ctx.px(box[0])}:y={ctx.px(box[1])}"
          f":format=auto{':shortest=1' if img else ':eof_action=pass'}{en}[v]")
    ctx.meta["insert"] = {"tracked": len(boxes) > 1, "first": boxes[0], "last": boxes[-1]}
    ctx.encode(["-i", ins[0]] + (["-loop", "1", "-i", ins[1]] if img else ["-i", ins[1]]), fc, "[v]", "0:a?", out)


@op("relight")
def relight(ctx, ins, p, out):
    """Selective grade / relighting through a mask. inputs: video, mask (white = subject). params:
    subject {brightness, contrast, saturation, gamma, temperature}, background {same}, feather (px, 4),
    invert. Use for skin-tone-safe grading, subject pop, or matching a new background's light."""
    v = info(ins[0])["video"]
    W, H = v["width"], v["height"]
    fmt = "gbrp10le" if any(b in (v["pix_fmt"] or "") for b in ("10", "12")) else "gbrp"
    feather = ctx.px(p.get("feather", 4))
    fc = (f"[0:v]setpts=PTS-STARTPTS,split[a][b];[a]{grade(p.get('subject', {}))},format={fmt}[fg];"
          f"[b]{grade(p.get('background', {}))},format={fmt}[bg];"
          f"[1:v]setpts=PTS-STARTPTS,scale={W}:{H}:flags=bicubic,format=gray"
          + (",negate" if p.get("invert") else "") + (f",gblur=sigma={feather}" if feather else "")
          + f",format={fmt}[m];[bg][fg][m]maskedmerge[v]")
    ctx.encode(["-i", ins[0], "-i", ins[1]], fc, "[v]", "0:a?", out)


def _rgb(c) -> np.ndarray:
    """'#c01e19', 'c01e19' or [r, g, b] -> uint8 RGB."""
    if isinstance(c, str):
        c = c.lstrip("#")
        c = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return np.uint8([[c]])


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _ground(small, spare=None):
    """Top edge (row per column) of the in-focus surface touching the bottom of the frame (sand, a road,
    a floor): local texture finds the sharp ground; everything below that edge counts as ground."""
    from scipy.ndimage import gaussian_filter1d, median_filter
    H = small.shape[0]
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mu * mu, 0))
    if spare is not None:  # spared subjects (e.g. insects on the edge) must not raise the ground line
        std = std * (1 - cv2.dilate(spare, np.ones((5, 5), np.uint8)))
    tex = _ramp(cv2.GaussianBlur(std, (0, 0), 5), 1.0, 2.0) > 0.5
    n, lab, st, _ = cv2.connectedComponentsWithStats(tex.astype(np.uint8))
    rb = np.isin(lab, [i for i in range(1, n) if st[i][1] + st[i][3] >= H - 2])
    top = median_filter(np.where(rb.any(0), rb.argmax(0), H), 31, mode="nearest").astype(np.float32)
    top = gaussian_filter1d(top, small.shape[1] * 0.02, mode="nearest")  # a gently curving line, no spikes
    return top


@op("recolor")
def recolor(ctx, ins, p, out):
    """Change the colour of one part of the shot, keeping its light and texture (LAB chroma swap).
    inputs: video[, mask (white = recolour)]. params: to (target colour, "#c01e19" or [r,g,b]);
    region = all | ground (in-focus surface touching the bottom edge: sand, road, floor) | mask (2nd input);
    sat_max (only recolour near-neutral pixels below this HSV saturation 0-1, sparing colourful things
    like clothes; default none); spare_warm (LAB b* threshold, e.g. 3: spares warm-toned pixels such as
    skin, insects, wood, judged per pixel at full resolution so thin legs/hair survive; default none);
    hue [lo, hi] (only these hues, degrees 0-360); strength (0-1, 1);
    smooth (temporal smoothing of the selection 0-1, 0.6)."""
    target = cv2.cvtColor(_rgb(p.get("to", "#c01e19")), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    region, k, smooth = p.get("region", "all"), float(p.get("strength", 1)), float(p.get("smooth", 0.6))
    v = info(ins[0])["video"]
    W, H = v["width"], v["height"]
    sw, sh = (960, round(960 * H / W)) if W > 960 else (W, H)  # selection is computed small, applied at full res
    masks = _mask_iter(ins[1], sw, sh) if region == "mask" else None
    state = {"top": None, "w": None}
    cover = []

    warm = p.get("spare_warm")

    def fix(f, i):
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32)
        spare = None
        if warm is not None:  # full-res, so 10 px legs at 4K aren't averaged away
            b = cv2.GaussianBlur(lab[..., 2] - 128, (0, 0), 1.5)
            spare = cv2.dilate(_ramp(b, float(warm), float(warm) + 3), np.ones((5, 5), np.uint8))
            spare = cv2.GaussianBlur(spare, (0, 0), 1.5)
        small = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA) if (sw, sh) != (W, H) else f
        w = np.ones((sh, sw), np.float32)
        if region == "ground":
            top = _ground(small, None if spare is None else cv2.resize(spare, (sw, sh), interpolation=cv2.INTER_AREA))
            state["top"] = top if state["top"] is None else smooth * state["top"] + (1 - smooth) * top
            band = sh * 0.03  # fade into the defocused zone instead of a hard line
            w = _ramp(np.arange(sh, dtype=np.float32)[:, None] - state["top"][None, :], -band, band)
        elif region == "mask":
            w = cv2.GaussianBlur(next(masks).astype(np.float32), (0, 0), 1.5)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        if p.get("sat_max") is not None:
            s = float(p["sat_max"])
            w *= 1 - _ramp(cv2.GaussianBlur(hsv[..., 1] / 255, (0, 0), 1.5), s, s + 0.08)
        if p.get("hue"):
            lo, hi = (x / 2 for x in p["hue"])  # OpenCV hue is 0-180
            h = hsv[..., 0].astype(np.float32)
            w *= ((h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)).astype(np.float32)
        if region != "ground":  # ground already smoothed its edge; this keeps other selections from flickering
            state["w"] = w if state["w"] is None else smooth * state["w"] + (1 - smooth) * w
            w = state["w"]
        cover.append(float(w.mean()))
        if (sw, sh) != (W, H):
            w = cv2.resize(w, (W, H), interpolation=cv2.INTER_LINEAR)
        if spare is not None:
            w = w * (1 - spare)
        lab[..., 1:] += (k * w)[..., None] * (target[1:] - lab[..., 1:])
        return cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    map_frames(ctx, ins[0], out, fix)
    ctx.meta["recolor"] = {"to": p.get("to", "#c01e19"), "region": region, "frames": len(cover),
                           "coverage_mean": round(float(np.mean(cover)), 4) if cover else 0}


STYLES = {
    "bw": "hue=s=0,eq=contrast=1.15",
    "vintage": "curves=preset=vintage,noise=alls=8:allf=t,vignette=PI/4",
    "cinematic": "colorbalance=rs=-0.05:bs=0.08:rh=0.08:bh=-0.06,eq=contrast=1.08:saturation=1.1,vignette=PI/5",
    "sketch": "edgedetect=low=0.08:high=0.25,negate",
    "painterly": "edgedetect=mode=colormix:high=0",
    "noir": "hue=s=0,curves=preset=strong_contrast,vignette=PI/3,noise=alls=6:allf=t",
}


@op("stylize")
def stylize(ctx, ins, p, out):
    """Look/style recipes. params: style = bw | vintage | cinematic | sketch | painterly | noir;
    or backend=onnx with model (.onnx image-to-image style network, same-size output)."""
    if p.get("backend") == "onnx":
        return map_frames(ctx, ins[0], out, lambda f, i: onnx_image(p["model"], f, int(p.get("tile", 512))))
    ctx.encode(["-i", ins[0]], f"[0:v]{STYLES[p.get('style', 'cinematic')]}[v]", "[v]", "0:a?", out)


@op("extend")
def extend(ctx, ins, p, out):
    """Background / shot extension. params: mode = canvas (extend the frame to width x height with a
    blurred fill of itself - aspect conversion without black bars) | hold (freeze the last frame for
    duration s) | push (hold with a slow push-in to zoom, 1.08); width, height (source px), duration, blur (40)."""
    v = info(ins[0])["video"]
    mode, a = p.get("mode", "canvas"), has_audio(ins[0])
    if mode == "canvas":
        W, H = ctx.even(p["width"]), ctx.even(p["height"])
        fc = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"gblur=sigma={ctx.px(p.get('blur', 40))},eq=brightness=-0.06[bg];"
              f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
        return ctx.encode(["-i", ins[0]], fc, "[v]", "0:a?", out)
    d = float(p.get("duration", 2))
    af = f";[0:a]apad=pad_dur={d}[a]" if a else ""
    if mode == "hold":
        return ctx.encode(["-i", ins[0]], f"[0:v]tpad=stop_mode=clone:stop_duration={d}[v]{af}", "[v]",
                          "[a]" if a else None, out)
    W, H, fps = v["width"], v["height"], v["fps_str"]
    n = max(1, round(d * (v["fps"] or 30)))
    z = float(p.get("zoom", 1.08))
    fc = (f"[1:v]reverse,trim=end_frame=1,loop=loop={n - 1}:size=1,setpts=N/FRAME_RATE/TB,"
          f"zoompan=z='1+{z - 1}*on/{n}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={W}x{H}:fps={fps},"
          f"format={v['pix_fmt']},setsar=1[t];[0:v]setsar=1[h];[h][t]concat=n=2:v=1:a=0[v]{af}")
    ctx.encode(["-i", ins[0], "-sseof", "-0.5", "-i", ins[0]], fc, "[v]", "[a]" if a else None, out)


# --- generators ---------------------------------------------------------------------------------

AUDIO_KINDS = {"voice", "music", "sfx"}
EXTS = {"image": {".png", ".jpg", ".jpeg", ".webp"}, "video": {".mp4", ".webm", ".mkv", ".mov", ".gif"},
        "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a"}}


def _family(kind):
    return "audio" if kind in AUDIO_KINDS else kind


def _seed(prompt, seed):
    return int(seed) if seed is not None else int(_h(prompt.encode()).hexdigest()[:8], 16)


def _procedural(req, part):
    """Deterministic gradient plate (placeholder B-roll / backgrounds; no model needed)."""
    kind, w, h, fps = req["kind"], req["width"], req["height"], req["fps"]
    s = _seed(req["prompt"], req["seed"])
    rng = np.random.default_rng(s)
    cols = ":".join(f"c{i}=0x{rng.integers(0, 0xFFFFFF):06x}" for i in range(4))
    src = f"gradients=s={w}x{h}:{cols}:n=4:seed={s % 2**31}:speed={0.01 if kind == 'video' else 0}:r={fps}"
    if kind == "image":
        part = part.with_suffix(".png")
        ffmpeg(["-f", "lavfi", "-i", src, "-frames:v", 1, part])
    else:
        ffmpeg(["-f", "lavfi", "-i", src, "-t", req["duration"], "-c:v", "libx264", "-qp", 0, "-pix_fmt", "yuv444p", part])
    return part, {"device": "cpu", "model": "procedural gradients (placeholder, not neural)"}


def _run(argv, timeout, label):
    """Run a worker process without a shell or console window; a timeout kills only this child."""
    from .workers import NO_WINDOW
    try:
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label} timed out after {timeout} s (process stopped)") from None
    if r.returncode:
        raise RuntimeError(f"{label} failed ({r.returncode}): {(r.stderr or r.stdout)[-1500:]}")
    return r


def _command(req, part, cfg=None):
    if cfg:  # configured worker: request JSON file + output path as separate argv items
        rq = part.with_name(part.stem + ".request.json")
        rq.write_text(json.dumps(req | {"out": str(part)}, indent=1), encoding="utf-8")
        try:
            r = _run([a.replace("{request}", str(rq)).replace("{out}", str(part)) for a in cfg["argv"]],
                     cfg["timeout"], f"worker {cfg['name']}")
        finally:
            rq.unlink(missing_ok=True)
        meta = {}
        for line in reversed(r.stdout.strip().splitlines()):  # workers may print one JSON line of provenance
            try:
                meta = json.loads(line)
            except ValueError:
                continue
            if isinstance(meta, dict):
                break
            meta = {}
        made = Path(meta.get("output") or part)
        if made.resolve().parent != part.resolve().parent or not made.name.startswith(part.stem):
            raise RuntimeError(f"worker {cfg['name']} reported an output outside its temp area: {made}")
        return made, meta
    tpl = os.environ.get("STUDIO_GEN_CMD")
    if not tpl:
        raise RuntimeError("backend=command needs a configured worker (`studio workers`) or STUDIO_GEN_CMD (template "
                           "with {prompt} {out} {width} {height} {duration} {fps} {kind} {seed} {init})")
    vals = {"prompt": req["prompt"], "out": part, "width": req.get("width", ""), "height": req.get("height", ""),
            "duration": req.get("duration") or "", "fps": req.get("fps", ""), "kind": req["kind"],
            "seed": _seed(req["prompt"], req["seed"]),
            "init": (req.get("references") or {}).get("init", "")}
    # JSON list, or a POSIX-quoted string (quote Windows paths: "C:\x\gen.exe" keeps its backslashes)
    parts = json.loads(tpl) if tpl.lstrip().startswith("[") else shlex.split(tpl)
    _run([a.format(**vals) for a in parts], float(os.environ.get("STUDIO_GEN_TIMEOUT", 3600)), "generator command")
    return part, {}


# --- ComfyUI ------------------------------------------------------------------------------------

def _local(url):
    host = urllib.parse.urlparse(url).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"ComfyUI must be local (loopback); refusing {url}")
    return url.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        raise RuntimeError("ComfyUI answered with a redirect; refusing to follow it")


_http = urllib.request.build_opener(_NoRedirect)


def _get(url, timeout=30):
    with _http.open(url, timeout=timeout) as r:
        return json.load(r)


def _post(url, body: bytes, ctype="application/json", timeout=30):
    with _http.open(urllib.request.Request(url, body, {"Content-Type": ctype}), timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


def _upload(url, path):
    """Upload a reference/mask image; returns the filename the server actually stored."""
    data = Path(path).read_bytes()
    name = "studio-" + _h(data).hexdigest()[:20] + Path(path).suffix.lower()  # same bytes -> same name, no clash
    boundary = "studio" + _h(name.encode()).hexdigest()[:16]
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + \
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    r = _post(f"{url}/upload/image", body, f"multipart/form-data; boundary={boundary}")
    return (f"{r['subfolder']}/" if r.get("subfolder") else "") + r["name"]


def select_output(outputs: dict, node: str, kind: str, index: int | None = None) -> dict:
    """The one file of the designated final node; never the first preview that happens to exist."""
    if node not in outputs:
        raise ValueError(f"designated final output node {node!r} produced nothing (nodes with output: {sorted(outputs)})")
    allowed = EXTS[_family(kind)]
    files = [f for coll in ("images", "videos", "gifs", "audio") for f in outputs[node].get(coll, [])
             if Path(f.get("filename", "")).suffix.lower() in allowed]
    for f in files:
        if any(c in f["filename"] for c in ("/", "\\", "..")) or ".." in f.get("subfolder", ""):
            raise ValueError(f"unsafe output filename from ComfyUI: {f}")
    if not files:
        raise ValueError(f"final output node {node!r} has no {kind} file")
    if len(files) > 1 and index is None:
        raise ValueError(f"final output node {node!r} returned {len(files)} files; set an explicit index")
    return files[index or 0]


def _cancel(url, pid):
    """Cancel only our prompt: targeted job cancel if the server has it, else drop it from the queue if still
    pending. Never the global /interrupt (it would stop someone else's running job)."""
    try:
        _post(f"{url}/api/jobs/{pid}/cancel", b"{}")
        return "cancelled (targeted)"
    except Exception:  # noqa: BLE001 - older server without the route
        pass
    try:
        q = _get(f"{url}/queue")
        if any(item[1] == pid for item in q.get("queue_pending", [])):
            _post(f"{url}/queue", json.dumps({"delete": [pid]}).encode())
            return "removed from queue"
        return "running job left alone: this server has no targeted cancel"
    except Exception as e:  # noqa: BLE001
        return f"cancel failed: {e}"


def _comfyui(req, part, cfg=None):
    legacy = cfg is None
    url = _local((cfg or {}).get("url") or os.environ.get("STUDIO_COMFYUI_URL", "http://127.0.0.1:8188"))
    wf_path = (cfg or {}).get("workflow") or os.environ.get("STUDIO_COMFYUI_WORKFLOW")
    if not wf_path:
        raise RuntimeError("backend=comfyui needs a configured worker or STUDIO_COMFYUI_WORKFLOW (API-format workflow "
                           "with {{prompt}} {{width}} {{height}} {{seed}} {{frames}} placeholders)")
    vals = {"prompt": req["prompt"], "width": req.get("width"), "height": req.get("height"),
            "seed": _seed(req["prompt"], req["seed"]), "fps": req.get("fps"), "duration": req.get("duration"),
            "frames": max(1, round((req.get("duration") or 1) * req["fps"])) if req["kind"] == "video" else 1}

    def fill(x):
        if isinstance(x, dict):
            return {k: fill(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fill(v) for v in x]
        if isinstance(x, str):
            for k, v in vals.items():
                if x == "{{%s}}" % k:
                    return v  # whole-string placeholder keeps its number type
                x = x.replace("{{%s}}" % k, str(v))
        return x

    wf = fill(json.loads(Path(wf_path).read_text(encoding="utf-8")))
    if legacy and req.get("references"):
        raise ValueError("references need a configured ComfyUI worker with upload bindings (`studio workers`)")
    if cfg:
        if cfg["node_classes"]:
            have = _get(f"{url}/object_info", 60)
            gone = [c for c in cfg["node_classes"] if c not in have]
            if gone:
                raise RuntimeError(f"ComfyUI is missing node classes {gone}")
        for field, b in cfg["bindings"].items():  # pure data binding, no string substitution into code
            if req.get(field) is not None:
                wf[b["node"]]["inputs"][b["input"]] = vals.get(field, req[field])
        for name, path in (req.get("references") or {}).items():
            b = cfg["uploads"][name]
            wf[b["node"]]["inputs"][b["input"]] = _upload(url, path)
    client = "studio-" + _h(str(part).encode()).hexdigest()[:12]
    r = _post(f"{url}/prompt", json.dumps({"prompt": wf, "client_id": client}).encode())
    if r.get("error") or r.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected the workflow: {r.get('error')} {r.get('node_errors')}")
    pid = r["prompt_id"]
    deadline = time.monotonic() + ((cfg or {}).get("timeout") or float(os.environ.get("STUDIO_COMFYUI_TIMEOUT", 3600)))
    try:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError
            hist = _get(f"{url}/history/{pid}").get(pid)
            st = (hist or {}).get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: {msgs[-1][1] if msgs else st}")
            if hist and (st.get("completed") or hist.get("outputs")):
                break
            time.sleep(1)
    except (TimeoutError, KeyboardInterrupt) as e:
        note = _cancel(url, pid)
        raise RuntimeError(f"ComfyUI generation {'timed out' if isinstance(e, TimeoutError) else 'interrupted'}; "
                           f"prompt {pid}: {note}") from None
    except RuntimeError:
        raise  # an execution error the server already reported: nothing left running
    except Exception as e:  # noqa: BLE001 - connection trouble: don't leave our prompt on the GPU
        raise RuntimeError(f"ComfyUI connection failed while waiting ({e}); prompt {pid}: {_cancel(url, pid)}") from None
    outputs = hist.get("outputs") or {}
    if legacy:  # old single-output workflows: exactly one candidate file, or fail as ambiguous
        nodes = [n for n in outputs if any(Path(f.get("filename", "")).suffix.lower() in EXTS[_family(req["kind"])]
                                           for c in ("images", "videos", "gifs", "audio") for f in outputs[n].get(c, []))]
        if len(nodes) != 1:
            raise RuntimeError(f"ComfyUI returned {req['kind']} output from {len(nodes)} nodes {nodes}; "
                               "configure a worker with output_node")
        f = select_output(outputs, nodes[0], req["kind"])
    else:
        f = select_output(outputs, cfg["output_node"], req["kind"])
    q = urllib.parse.urlencode({k: f.get(k, "") for k in ("filename", "subfolder", "type")})
    part = part.with_suffix(Path(f["filename"]).suffix.lower())
    with _http.open(f"{url}/view?{q}", timeout=300) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):  # streamed: no whole-video read into RAM
            fh.write(chunk)
    return part, {"prompt_id": pid, "device": (cfg or {}).get("device", "unknown"), "workflow": str(wf_path)}


GENERATORS = {"procedural": _procedural, "command": _command, "comfyui": _comfyui}


def generate_media(prompt, kind="image", out=None, width=1920, height=1080, duration=None, fps=30, backend=None,
                   seed=None, init=None, *, references=None, mask=None, worker=None, voice=None, text=None,
                   timeout=None, extra=None) -> str:
    """Generate one image/video/voice/music/sfx file. Returns its path; provenance (kind, worker, device,
    native properties, sha256, seconds) goes beside it as <file>.gen.json. Never replaces an existing file:
    the result is written to a temporary sibling, validated, then published."""
    from . import workers
    from .media import publish, sha256
    from .qc import validate, validate_audio
    if kind not in workers.KINDS:
        raise ValueError(f"kind must be one of {sorted(workers.KINDS)}")
    cfg = workers.worker_config(worker) if worker else None
    if cfg and timeout:
        cfg["timeout"] = float(timeout)
    backend = cfg["backend"] if cfg else (backend or os.environ.get("STUDIO_GEN_BACKEND", "procedural"))
    if backend not in GENERATORS:
        raise ValueError(f"unknown generator {backend}; have {sorted(GENERATORS)}")
    if kind in AUDIO_KINDS and backend == "procedural":
        raise ValueError(f"procedural backend cannot generate {kind}: configure a local worker (`studio workers`)")
    refs = dict(references or {}) | ({"init": str(init)} if init else {}) | ({"mask": str(mask)} if mask else {})
    for k, v in refs.items():
        if not Path(v).is_file():
            raise FileNotFoundError(f"reference {k}: {v} does not exist")
    req = {"kind": kind, "prompt": prompt, "text": text, "voice": voice, "seed": seed, "references": refs or None}
    if kind in AUDIO_KINDS:
        req["duration"] = duration  # None for voice: narration length comes from synthesis
    else:
        req |= {"width": int(width), "height": int(height), "fps": float(eval_fps(fps)), "duration": float(duration or 5)}
    req |= extra or {}
    if cfg:
        workers.request_check(cfg, req)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = {"image": ".png", "video": ".mkv"}.get(kind, ".wav")
    part = out.with_name(out.stem + ".part" + (out.suffix or suffix))
    t0 = time.monotonic()
    produced = None
    try:
        fn = GENERATORS[backend]
        produced, meta = fn(req, part, cfg) if backend != "procedural" else fn(req, part)
        produced = Path(produced)
        if not produced.exists():
            raise RuntimeError(f"generator finished without writing {produced}")
        fam = _family(kind)
        if produced.suffix.lower() not in EXTS[fam] | ({".mkv"} if fam == "video" else set()):
            raise RuntimeError(f"generator wrote {produced.suffix} for a {kind} request")
        try:
            i = info(produced)
        except RuntimeError as e:
            raise RuntimeError(f"generated {kind} failed validation: not decodable ({e})") from None
        if fam == "audio":
            checks = validate_audio(str(produced), duration=req.get("duration") if "duration" in (cfg or {}).get("fields", []) else None)
        else:
            checks = validate(str(produced), decode=True) if fam == "video" else {"ok": bool(i["video"]), "checks": []}
            if fam == "image" and not i["is_image"]:
                checks = {"ok": False, "checks": [{"check": "is_image", "ok": False}]}
        if not checks["ok"]:
            raise RuntimeError(f"generated {kind} failed validation: {[c for c in checks['checks'] if not c['ok']]}")
        final = out.with_suffix(produced.suffix)
        publish(produced, final)
    except BaseException:
        for f in {part, produced, *part.parent.glob(part.stem + ".*")} - {None}:
            Path(f).unlink(missing_ok=True)
        raise
    native = {"video": i["video"] and {k: i["video"][k] for k in ("width", "height", "fps_str", "frames", "pix_fmt")},
              "audio": i["audio"] and {k: i["audio"][k] for k in ("codec", "sample_rate", "channels")},
              "duration": i["duration"]}
    prov = {"kind": kind, "backend": backend, "worker": cfg and cfg["name"], "worker_fingerprint": cfg and workers.fingerprint(cfg),
            "request": {k: v for k, v in req.items() if v is not None}, "native": native, "output": str(final),
            "sha256": sha256(final), "seconds": round(time.monotonic() - t0, 2), "validation": checks,
            "device": meta.get("device") or (cfg or {}).get("device"), "meta": meta}
    Path(str(final) + ".gen.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    return str(final)


def provenance(path) -> dict:
    f = Path(str(path) + ".gen.json")
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _fresh(raw: Path):
    """The op's scratch outputs live in the step cache: clear stale ones from an interrupted attempt."""
    for f in raw.parent.glob(raw.stem + ".*"):
        f.unlink()


@op("generate")
def generate(ctx, ins, p, out):
    """Generated image/video clip (B-roll, plates, backgrounds). inputs: none, or an init image/video
    (image-to-video / restyle, if the backend supports it). params: prompt, kind = video | image,
    backend = procedural | command | comfyui (env STUDIO_GEN_BACKEND), worker (configured local worker,
    see `studio workers`), width, height (source px, default 1920x1080 or the init's size), fps (30),
    duration (s, 5), seed. The result is conformed to exactly width x height @ fps for duration (generated
    audio is kept); upscale/interpolate it afterwards for 4K60 masters."""
    iv = info(ins[0])["video"] if ins else None
    W = ctx.even(p.get("width") or (iv["width"] / ctx.scale if iv else 1920))
    H = ctx.even(p.get("height") or (iv["height"] / ctx.scale if iv else 1080))
    fps, d = p.get("fps") or (iv["fps_str"] if iv else 30), float(p.get("duration", 5))
    raw = ctx.workdir / f"gen-{_h(json.dumps([p, ins], sort_keys=True).encode()).hexdigest()[:16]}.mkv"
    _fresh(raw)
    t0 = time.time()
    res = generate_media(p.get("prompt", ""), p.get("kind", "video"), raw, W, H, d, fps, p.get("backend"), p.get("seed"),
                         ins[0] if ins else None, worker=p.get("worker"))
    img = Path(res).suffix.lower() in IMAGE_EXTS
    nv = info(res)["video"]
    ctx.meta["generate"] = {"backend": p.get("backend") or os.environ.get("STUDIO_GEN_BACKEND", "procedural"),
                            "worker": p.get("worker"), "file": res, "seconds": round(time.time() - t0, 2),
                            "native": [nv["width"], nv["height"], nv["fps_str"]], "conformed": [W, H, str(fps), d],
                            "provenance": provenance(res)}
    fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,"
          f"fps={fps},tpad=stop_mode=clone:stop_duration={d},trim=duration={d},setpts=PTS-STARTPTS[v]")
    a = None
    if not img and has_audio(res):  # keep generated sound, fitted to the clip length
        fc += f";[0:a:0]apad,atrim=duration={d},asetpts=PTS-STARTPTS[a]"
        a = "[a]"
    ctx.encode((["-loop", "1", "-t", d] if img else []) + ["-i", res], fc, "[v]", a, out)


@op("generate_audio")
def generate_audio(ctx, ins, p, out):
    """Generated voice / music / sound effect as an audio asset (a later `audio` step mixes it in).
    inputs: none. params: kind = voice | music | sfx, worker (configured local worker, `studio workers`),
    text (voice) or prompt, voice, seed, duration (s; music/sfx only, when the worker supports it).
    The worker's native WAV is kept; voice length is whatever synthesis produced."""
    kind = p.get("kind", "voice")
    if kind not in AUDIO_KINDS:
        raise ValueError("generate_audio kind must be voice | music | sfx")
    raw = Path(out).with_name(Path(out).stem + ".gen.wav")
    _fresh(raw)
    res = generate_media(p.get("prompt") or p.get("text") or "", kind, raw, duration=p.get("duration"),
                         seed=p.get("seed"), worker=p.get("worker"), backend=p.get("backend"), voice=p.get("voice"),
                         text=p.get("text"))
    ctx.meta["generate_audio"] = {"worker": p.get("worker"), "file": res, "provenance": provenance(res)}
    ctx.commands.append(ffmpeg(["-i", res, "-map", "0:a:0", "-c:a", "pcm_s24le", out]))


def eval_fps(fps) -> float:
    if isinstance(fps, str) and "/" in fps:
        a, b = fps.split("/")
        return float(a) / float(b)
    return float(fps)
