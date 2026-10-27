"""Restoration & motion: stabilize, denoise, sharpen/deblur, deinterlace, lens correction,
super-resolution, frame interpolation, face restoration, shot (color) matching.

Every AI-capable op has a `backend` param, so models swap without changing the contract.
FFmpeg backends are the defaults; `backend=onnx` runs a user-supplied image-to-image model
(Real-ESRGAN-class SR, restoration nets) on CPU, or DirectML if opted in via STUDIO_ORT_PROVIDERS.
"""
import tempfile
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import models
from .media import FrameWriter, ffmpeg, fpath, info, read_frames
from .ops import op


def map_frames(ctx, src, out, fn, size=None, parallel=False):
    """Run fn(rgb_frame, index) -> rgb_frame over every frame at native resolution, keep the audio.
    size = (w, h) of fn's output when it differs from the input. parallel=True runs frames on a thread
    pool (OpenCV/numpy release the GIL), output order kept; only for fns with no frame-to-frame state."""
    v = info(src)["video"]
    W, H = size or (v["width"], v["height"])
    tmp = Path(tempfile.mkdtemp(dir=ctx.workdir)) / "frames.mkv"
    w = FrameWriter(tmp, W, H, v["fps_str"], "rgb24")
    n = 0
    frames = enumerate(read_frames(src, v["width"], v["height"]))
    try:
        if not parallel:
            for i, f in frames:
                w.write(np.ascontiguousarray(fn(f, i), np.uint8))
                n += 1
        else:
            import os
            from collections import deque
            from concurrent.futures import ThreadPoolExecutor
            workers = int(os.environ.get("STUDIO_FRAME_WORKERS", max(1, (os.cpu_count() or 4) // 2)))
            with ThreadPoolExecutor(workers) as pool:
                window = deque()  # bounded look-ahead keeps memory flat (a 4K float frame is ~100 MB of temps)
                for i, f in frames:
                    window.append(pool.submit(fn, f, i))
                    if len(window) >= 2 * workers:
                        w.write(np.ascontiguousarray(window.popleft().result(), np.uint8))
                        n += 1
                while window:
                    w.write(np.ascontiguousarray(window.popleft().result(), np.uint8))
                    n += 1
    finally:
        w.close()
    if not n:
        raise RuntimeError(f"no frames decoded from {src}")
    ctx.encode(["-i", tmp, "-i", src], None, "0:v", "1:a?", out)
    tmp.unlink(missing_ok=True)
    tmp.parent.rmdir()


@lru_cache(3)
def _ort(model):
    import onnxruntime as ort
    from .sched import acquire
    from .vision import ORT_PROVIDERS
    acquire(f"onnx:{Path(str(model)).name}", _ort.cache_clear)
    return ort.InferenceSession(models.path(model), providers=ORT_PROVIDERS)


def onnx_image(model, frame, tile=256, pad=16):
    """Tiled NCHW float(0-1) RGB image-to-image inference; returns uint8 RGB at the model's scale."""
    sess = _ort(model)
    name = sess.get_inputs()[0].name
    x = frame.astype(np.float32) / 255
    H, W = x.shape[:2]
    out, s = None, None
    for y in range(0, H, tile):
        for xx in range(0, W, tile):
            th, tw = min(tile, H - y), min(tile, W - xx)
            y0, x0, y1, x1 = max(0, y - pad), max(0, xx - pad), min(H, y + th + pad), min(W, xx + tw + pad)
            o = sess.run(None, {name: x[y0:y1, x0:x1].transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            if out is None:
                s = round(o.shape[0] / (y1 - y0))
                out = np.zeros((H * s, W * s, 3), np.float32)
            out[y * s:(y + th) * s, xx * s:(xx + tw) * s] = o[(y - y0) * s:(y - y0 + th) * s, (xx - x0) * s:(xx - x0 + tw) * s]
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


@op("stabilize")
def stabilize(ctx, ins, p, out):
    """Two-pass stabilization. params: backend = vidstab | deshake; shakiness (1-10, 5);
    smoothing (frames, 15); zoom (%, 0; negative = show borders); optzoom (0|1|2, 1 = auto zoom
    to hide borders); crop = black | keep."""
    from .devices import ffmpeg_filters
    if p.get("backend") == "deshake" or "vidstabdetect" not in ffmpeg_filters():  # route to what this ffmpeg has
        return ctx.encode(["-i", ins[0]], "[0:v]deshake[v]", "[v]", "0:a?", out)
    trf = ctx.workdir / f"stab-{Path(out).stem}.trf"
    ctx.commands.append(ffmpeg(["-i", ins[0], "-map", "0:v:0", "-vf",
                                f"vidstabdetect=shakiness={p.get('shakiness', 5)}:result={fpath(trf)}", "-f", "null", "-"]))
    f = (f"vidstabtransform=input={fpath(trf)}:smoothing={p.get('smoothing', 15)}:zoom={p.get('zoom', 0)}"
         f":optzoom={p.get('optzoom', 1)}:crop={p.get('crop', 'black')}:interpol=bicubic")
    ctx.encode(["-i", ins[0]], f"[0:v]{f}[v]", "[v]", "0:a?", out)
    trf.unlink(missing_ok=True)


@op("denoise")
def denoise(ctx, ins, p, out):
    """Video denoise. params: backend = hqdn3d (fast, temporal) | nlmeans (slow, best) | onnx;
    strength (hqdn3d luma, 4; nlmeans s, 3); model (.onnx path or registry name for backend=onnx)."""
    b = p.get("backend", "hqdn3d")
    if b == "onnx":
        return map_frames(ctx, ins[0], out, lambda f, i: onnx_image(p["model"], f))
    s = float(p.get("strength", 4 if b == "hqdn3d" else 3))
    f = {"hqdn3d": f"hqdn3d={s}:{s * 0.75}:{s * 1.5}:{s * 1.1}", "nlmeans": f"nlmeans=s={s}:p=7:r=15"}[b]
    ctx.encode(["-i", ins[0]], f"[0:v]{f}[v]", "[v]", "0:a?", out)


@op("sharpen")
def sharpen(ctx, ins, p, out):
    """Detail / mild deblur. params: backend = cas (contrast-adaptive) | unsharp; amount (0-1, 0.5)."""
    a = float(p.get("amount", 0.5))
    f = f"cas=strength={a}" if p.get("backend", "cas") == "cas" else f"unsharp=5:5:{a * 2}:5:5:0"
    ctx.encode(["-i", ins[0]], f"[0:v]{f}[v]", "[v]", "0:a?", out)


@op("deinterlace")
def deinterlace(ctx, ins, p, out):
    """bwdif deinterlace. params: rate = frame (keep fps) | field (double fps, smoother motion)."""
    ctx.encode(["-i", ins[0]], f"[0:v]bwdif=mode={'send_field' if p.get('rate') == 'field' else 'send_frame'}[v]",
               "[v]", "0:a?", out)


@op("lens")
def lens(ctx, ins, p, out):
    """Barrel/pincushion correction. params: k1, k2 (negative = remove barrel, e.g. -0.2, 0)."""
    ctx.encode(["-i", ins[0]], f"[0:v]lenscorrection=k1={p.get('k1', 0)}:k2={p.get('k2', 0)}:i=bilinear[v]",
               "[v]", "0:a?", out)


@op("upscale")
def upscale(ctx, ins, p, out):
    """Super-resolution. params: factor (2) | width+height; backend = lanczos | onnx (model = registry
    name real_esrgan_x2 | real_esrgan_x4 (default by factor) or a .onnx path; output resized to the
    requested size); tile (px, 256). Neural SR is slow on CPU: use it on short or low-res segments."""
    v = info(ins[0])["video"]
    f = float(p.get("factor", 2))
    W, H = ((ctx.even(p["width"]), ctx.even(p["height"])) if p.get("width")  # source px, rescaled for previews
            else (int(v["width"] * f) // 2 * 2, int(v["height"] * f) // 2 * 2))
    if p.get("backend") == "onnx":
        m = p.get("model") or ("real_esrgan_x4" if W / v["width"] > 2.01 else "real_esrgan_x2")
        ctx.meta["upscale"] = {"backend": f"onnx:{Path(str(m)).name}"}
        return map_frames(ctx, ins[0], out, lambda f, i: cv2.resize(
            onnx_image(m, f, int(p.get("tile", 256))), (W, H), interpolation=cv2.INTER_AREA), (W, H))
    ctx.encode(["-i", ins[0]], f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1[v]", "[v]", "0:a?", out)


@op("interpolate")
def interpolate(ctx, ins, p, out):
    """Frame interpolation to a higher fps. params: fps (target, e.g. 60 or "60000/1001"); backend =
    minterpolate (motion-compensated, slow at 4K) | blend (frame blending) | dup (repeat frames) |
    onnx (RIFE-class neural VFI; model = rife_4.9 or a .onnx with inputs img0, img1[, timestep])."""
    fps, b = p["fps"], p.get("backend", "minterpolate")
    if b == "onnx":
        return _vfi(ctx, ins[0], out, Fraction(str(fps)), p.get("model", "rife_4.9"))
    dur = info(ins[0])["video"]["duration"]
    # minterpolate needs a frame past the end to fill the last interval: pad with clones, trim back
    f = {"minterpolate": f"tpad=stop_mode=clone:stop=2,minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc"
                         f":me_mode=bidir:vsbmc=1,trim=duration={dur}",
         "blend": f"framerate=fps={fps}", "dup": f"fps={fps}"}[b]
    ctx.encode(["-i", ins[0]], f"[0:v]{f}[v]", "[v]", "0:a?", out)


def _vfi(ctx, src, out, target, model):
    """Two-frame VFI contract: model(img0, img1[, timestep]) -> frame, NCHW RGB 0-1. Frames are padded
    to a multiple of 64 (what RIFE-class nets need) and cropped back. Models without a timestep input
    can only make exact midpoints, so they need target = 2x source fps."""
    v = info(src)["video"]
    sf, H, W = Fraction(v["fps_str"]), v["height"], v["width"]
    sess = _ort(model)
    names = [i.name for i in sess.get_inputs()]
    if len(names) < 3 and target != 2 * sf:
        raise ValueError(f"{Path(str(model)).name} has no timestep input: it only doubles fps ({float(2 * sf)})")
    ph, pw = -H % 64, -W % 64

    def run(a, b, t):
        x = [np.pad(f.astype(np.float32) / 255, ((0, ph), (0, pw), (0, 0)), mode="edge").transpose(2, 0, 1)[None]
             for f in (a, b)]
        feed = dict(zip(names, x + [np.array([t], np.float32)]))
        y = sess.run(None, feed)[0][0, :3, :H, :W].transpose(1, 2, 0)
        return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)

    tmp = Path(tempfile.mkdtemp(dir=ctx.workdir)) / "vfi.mkv"
    w = FrameWriter(tmp, W, H, str(target), "rgb24")
    frames = read_frames(src, W, H)
    a, b, ia, k, made = next(frames, None), next(frames, None), 0, 0, 0
    total = round(Fraction(v["duration"]).limit_denominator(100000) * target)
    try:
        while a is not None and k < total:
            u = Fraction(k) * sf / target  # output frame k sits at source position u
            while b is not None and u >= ia + 1:
                a, b, ia = b, next(frames, None), ia + 1
            t = float(u - ia) if b is not None else 0.0  # past the last source frame: hold it
            w.write(a if t < 1e-6 else run(a, b, t))
            made += t >= 1e-6
            k += 1
    finally:
        w.close()
    ctx.meta["interpolate"] = {"backend": f"onnx:{Path(str(model)).name}", "frames": k, "synthesized": made}
    ctx.encode(["-i", tmp, "-i", src], None, "0:v", "1:a?", out)
    tmp.unlink(missing_ok=True)
    tmp.parent.rmdir()


# --- look transfer ---------------------------------------------------------------------------------

def _palette(ref_path):
    """A reference image's colour as a function of brightness: (L bin centres, median a*, median b*)."""
    img = cv2.cvtColor(cv2.imread(str(ref_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = img[..., 0] * 100 / 255, img[..., 1] - 128, img[..., 2] - 128
    rows = [(c, np.median(a[m]), np.median(b[m])) for c in np.arange(5, 100, 5.0)
            if (m := (L >= c - 5) & (L < c + 5)).sum() > 300]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), float(np.median(L))


def _tone(L, mid_in, mid_out, gain):
    """Exposure/contrast: linear around the midtone with soft knees, so nothing crushes or clips."""
    x = mid_out + gain * (L - mid_in)
    x = np.where(x < 18, 8 + 10 * np.exp((x - 18) / 10), x)
    return np.where(x > 82, 92 - 10 * np.exp(-(x - 82) / 10), x)


@op("look_match")
def look_match(ctx, ins, p, out):
    """Re-grade a shot into a reference image's look (e.g. "make the sand red" from a mood frame).
    Every pixel takes the reference's colour for its brightness (so shadows, mids and highlights each get
    the reference's tint), texture is kept, and subjects warmer than the scene (insects, skin, wood) keep
    and amplify their warmth instead of being painted over: no masks, no halos.
    inputs: clip, reference image. params: exposure = match (reference brightness) | keep (the clip's own);
    mid_offset (L shift of the midtone, 0); contrast (1.15); saturation (1.0); subject_warmth (1.0);
    clarity (fine-grain contrast, 0.6); vignette (0.12)."""
    bins, pa, pb, ref_mid = _palette(ins[1])
    v = info(ins[0])["video"]
    sample = [f for i, f in enumerate(read_frames(ins[0], 480, max(2, round(480 * v["height"] / v["width"]) // 2 * 2)))
              if i % max(1, int((v["frames"] or 240) // 8)) == 0]
    mid_in = float(np.median([np.median(cv2.cvtColor(f, cv2.COLOR_RGB2LAB)[..., 0]) * 100 / 255 for f in sample]))
    mid_out = (ref_mid if p.get("exposure", "match") == "match" else mid_in) + float(p.get("mid_offset", 0))
    gain, sat = float(p.get("contrast", 1.15)), float(p.get("saturation", 1.0))
    warmth, clarity, vig = float(p.get("subject_warmth", 1.0)), float(p.get("clarity", 0.6)), float(p.get("vignette", 0.12))
    k = v["width"] / 3840  # radii are tuned at 4K; previews scale them down
    H, W = v["height"], v["width"]
    yy, xx = np.ogrid[:H, :W]
    falloff = (1 - vig * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) - 0.5, 0, 1))

    # Per-pixel curves as lookup tables over the 8-bit L channel (float32, no per-frame interp/exp work).
    l8 = np.arange(256, dtype=np.float32) * 100 / 255
    tone_lut = _tone(l8, mid_in, mid_out, gain).astype(np.float32)
    grid = np.linspace(0, 100, 1001, dtype=np.float32)  # palette indexed by the toned L, 0.1 steps
    pal_a = (np.interp(grid, bins, pa) * sat).astype(np.float32)
    pal_b = (np.interp(grid, bins, pb) * sat).astype(np.float32)
    falloff = falloff.astype(np.float32)

    def fix(f, i):
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
        Lc = tone_lut[lab[..., 0]]
        a = lab[..., 1].astype(np.float32) - 128
        b = lab[..., 2].astype(np.float32) - 128
        warm = np.maximum(cv2.GaussianBlur(b, (0, 0), max(0.5, k)) - 3.0, 0) * warmth
        idx = np.clip(Lc * 10, 0, 1000).astype(np.int16)
        A = pal_a[idx] + 0.5 * a - 0.15 * warm
        B = pal_b[idx] + warm
        Lc = Lc + 1.1 * warm  # warm subjects glow against the re-coloured scene
        if clarity:
            Lc = Lc + clarity * (Lc - cv2.GaussianBlur(Lc, (0, 0), max(0.5, 3 * k)))
        out8 = cv2.merge([np.clip(Lc * 2.55 + 0.5, 0, 255), np.clip(A + 128.5, 0, 255), np.clip(B + 128.5, 0, 255)])
        rgb = cv2.cvtColor(out8.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return (rgb * falloff[..., None]).astype(np.uint8) if vig else rgb

    map_frames(ctx, ins[0], out, fix, parallel=True)
    ctx.meta["look_match"] = {"reference": Path(ins[1]).name, "mid_in": round(mid_in, 1), "mid_out": round(mid_out, 1),
                              "palette_mid_ab": [round(float(np.interp(mid_out, bins, pa)), 1),
                                                 round(float(np.interp(mid_out, bins, pb)), 1)]}


# --- face restoration -----------------------------------------------------------------------------

# FFHQ alignment template (eyes, nose, mouth corners; image-left first), as fractions of the crop.
_FFHQ = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                  [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32)


@lru_cache(1)
def _yunet():
    return cv2.FaceDetectorYN.create(models.path("yunet"), "", (320, 320), 0.7, 0.3, 50)


def detect_faces(frame, height=720, score=0.7) -> list:
    """5-point landmarks (5x2, frame px) of each face, detected on a downscaled copy."""
    H, W = frame.shape[:2]
    k = min(1.0, height / H)
    small = cv2.resize(frame, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
    det = _yunet()
    det.setInputSize((small.shape[1], small.shape[0]))
    det.setScoreThreshold(score)
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    return [] if faces is None else [f[4:14].reshape(5, 2) / k for f in faces]


def restore_face(frame, lm, model, strength=0.8, size=512):
    """Align one face to FFHQ, run the restorer (NCHW RGB in [-1, 1]), feather it back into the frame."""
    M, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), _FFHQ * size, method=cv2.LMEDS)
    if M is None:
        return frame
    crop = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sess = _ort(model)
    x = (crop.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0].transpose(1, 2, 0)
    fixed = ((np.clip(y, -1, 1) + 1) * 127.5).astype(np.float32)
    pad = size // 10  # feathered square: blends the crop edges into the original
    m = np.zeros((size, size), np.float32)
    m[pad:-pad, pad:-pad] = 1
    m = cv2.GaussianBlur(m, (0, 0), pad / 2)
    inv = cv2.invertAffineTransform(M)
    H, W = frame.shape[:2]
    back = cv2.warpAffine(fixed, inv, (W, H), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(m, inv, (W, H))[..., None] * strength
    return (frame * (1 - a) + back * a + 0.5).astype(np.uint8)


@op("face_restore")
def face_restore(ctx, ins, p, out):
    """Detect faces (YuNet) and restore each (GFPGAN-class ONNX), pasted back at master resolution.
    params: model (gfpgan_1.4 or a .onnx with a 512x512 NCHW [-1, 1] input), strength (blend 0-1, 0.8),
    min_size (face width px to bother with, source px, 48), score (detector threshold, 0.7).
    CPU cost is roughly 1 s per face per frame: trim to the shots that need it."""
    model, strength = p.get("model", "gfpgan_1.4"), float(p.get("strength", 0.8))
    min_w, score = ctx.px(p.get("min_size", 48)), float(p.get("score", 0.7))
    counts = []

    def fix(f, i):
        # eye-to-mouth-corner span is roughly 40% of the face width
        faces = [lm for lm in detect_faces(f, score=score) if np.ptp(lm[:, 0]) * 2.5 >= min_w]
        counts.append(len(faces))
        f = f.copy()
        for lm in faces:
            f = restore_face(f, lm, model, strength)
        return f

    map_frames(ctx, ins[0], out, fix)
    ctx.meta["face_restore"] = {"model": Path(str(model)).name, "frames": len(counts),
                                "frames_with_faces": sum(c > 0 for c in counts), "faces_max": max(counts, default=0)}


def _stats(path, n=24):
    v = info(path)["video"]
    h = min(256, v["height"])
    w = max(2, int(v["width"] * h / v["height"]) // 2 * 2)
    every = max(1, int((v["frames"] or (v["duration"] or 1) * (v["fps"] or 30)) // n))
    px = np.concatenate([f.reshape(-1, 3) for i, f in enumerate(read_frames(path, w, h)) if i % every == 0])
    px = px.astype(np.float32) / 255
    return px.mean(0), px.std(0) + 1e-4


@op("match_color")
def match_color(ctx, ins, p, out):
    """Shot matching: transfer per-channel mean/contrast of a reference shot onto this clip.
    inputs: clip, reference (video or image). params: strength (0-1, 1)."""
    k = float(p.get("strength", 1))
    (ms, ss), (mr, sr) = _stats(ins[0]), _stats(ins[1])
    exprs = []
    for c, a, b, g, h in zip("rgb", ms, ss, mr, sr):
        gain, off = 1 + k * (h / b - 1), k * (g - a)
        exprs.append(f"{c}='clip((val/maxval-{a:.5f})*{gain:.5f}+{a + off:.5f},0,1)*maxval'")
    ctx.meta["match_color"] = {"source_mean": ms.round(4).tolist(), "reference_mean": mr.round(4).tolist()}
    ctx.encode(["-i", ins[0]], f"[0:v]lutrgb={':'.join(exprs)}[v]", "[v]", "0:a?", out)
