"""AI rotoscope: proxy-resolution masks + tracking, composited back at master resolution.

Masks are inferred at a small `height` (default 512) and stored as lossless gray FFV1 at the
source frame rate; the background op upsamples them (edge-aware, guided by the master luma)
onto the full-resolution frames, so the master is never downscaled for the model.
"""
import os
from functools import lru_cache

import cv2
import numpy as np

from .media import FrameWriter, info, read_frames
from .ops import _fill, media_in, op

ORT_PROVIDERS = os.environ.get("STUDIO_ORT_PROVIDERS", "CPUExecutionProvider").split(",")


@lru_cache(4)
def _session(model):
    from rembg import new_session
    from .sched import acquire
    acquire(f"rembg:{model}", _session.cache_clear)
    return new_session(model, providers=ORT_PROVIDERS)


def _segment(model, frame, **kw):
    from rembg import remove
    return np.asarray(remove(frame, session=_session(model), only_mask=True, **kw))


def _dims(path, height):
    v = info(path)["video"]
    h = min(int(height), v["height"]) // 2 * 2
    return v, int(round(v["width"] * h / v["height"] / 2)) * 2, h


def _flow_box(prev, cur, box):
    """Shift box by the median Lucas-Kanade flow of features inside it."""
    x, y, w, h = [int(v) for v in box]
    m = max(4, int(0.1 * min(w, h)))  # margin: features on the object's own edges sit right at a tight box
    roi = np.zeros_like(prev)
    roi[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m] = 255
    pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 3, mask=roi)
    if pts is None:
        return box
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
    good = st.ravel() == 1
    if good.sum() < 3:
        return box
    dx, dy = np.median((nxt - pts)[good].reshape(-1, 2), axis=0)
    return [box[0] + dx, box[1] + dy, box[2], box[3]]


def track(path, box, height=512, scale=1.0):
    """Track a source-px box [x, y, w, h] through the clip; returns per-frame boxes in source px."""
    v, W, H = _dims(path, height)
    k = H / v["height"] * scale  # source px -> inference px
    b = [c * k for c in box]
    prev, out = None, []
    for f in read_frames(path, W, H):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            b = _flow_box(prev, g, b)
        prev = g
        out.append([round(float(c) / k, 1) for c in b])
    return out


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter or 1)


def _hist(frame, sel):
    """Appearance signature: normalized HSV hue/saturation histogram of the selected pixels."""
    h = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)], [0, 1], sel.astype(np.uint8), [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, None, 1, 0, cv2.NORM_L1)


def _isolate(m, frame, box, tmpl):
    """Identity filter: keep the one subject in mask m that best matches the tracked box (overlap) and
    the target's appearance (histogram), dropping other people/objects. Returns (mask, bbox, hist, score);
    bbox is None when nothing plausible is found (the mask is then left as is)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), connectivity=8)
    min_area = 0.02 * box[2] * box[3]
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < min_area:
            continue
        sel = lab == i
        sim = 1 - cv2.compareHist(tmpl, _hist(frame, sel), cv2.HISTCMP_BHATTACHARYYA) if tmpl is not None else 1.0
        score = 0.6 * _iou([x, y, w, h], box) + 0.4 * sim
        if best is None or score > best[0]:
            best = (score, i, [x, y, w, h], sim)
    if best is None:
        return m, None, None, 0.0
    score, i, (x, y, w, h), sim = best
    # keep fragments (a hand, a split limb) lying inside the chosen subject's box
    keep = [j for j in range(1, n) if j == i or (st[j][0] >= x and st[j][1] >= y and st[j][0] + st[j][2] <= x + w
                                                 and st[j][1] + st[j][3] <= y + h)]
    sel = np.isin(lab, keep)
    return np.where(sel, m, 0).astype(m.dtype), [x, y, w, h], _hist(frame, sel), round(float(score), 3) if sim > 0.4 else 0.0


def _grabcut(frame, box):
    x, y, w, h = [int(round(c)) for c in box]
    H, W = frame.shape[:2]
    x, y = min(max(x, 0), W - 2), min(max(y, 0), H - 2)
    w, h = max(2, min(w, W - x - 1)), max(2, min(h, H - y - 1))
    m = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), m, (x, y, w, h), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    return np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8)


@op("mask")
def mask(ctx, ins, p, out):
    """Segmentation matte -> gray mask video. params:
    target = person | salient | object; model (rembg name: u2net_human_seg, birefnet-portrait,
    isnet-general-use, ...); box [x,y,w,h] / point [x,y] (source px, frame 0): required for
    target=object, and with person/salient it isolates that one subject (identity-tracked);
    backend = sam | grabcut (object only); height (inference px, 512); stride (run the model every
    Nth frame, interpolate between, 1); smooth (temporal EMA weight 0-1, 0.3)."""
    target = p.get("target", "person")
    model = p.get("model") or {"person": "u2net_human_seg", "salient": "isnet-general-use", "object": "sam"}[target]
    if target == "object" and p.get("backend") == "grabcut":
        model = "grabcut"
    stride, smooth = max(1, int(p.get("stride", 1))), float(p.get("smooth", 0.3))
    v, W, H = _dims(ins[0], p.get("height", 512))
    k = H / v["height"] * ctx.scale
    box = [c * k for c in p["box"]] if p.get("box") else None
    point = [c * k for c in p["point"]] if p.get("point") else None
    if target == "object" and not (box or point):
        raise ValueError("target=object needs box [x,y,w,h] or point [x,y]")

    def infer(frame):
        if target != "object":
            return _segment(model, frame)
        if model == "grabcut":
            return _grabcut(frame, box or [point[0] - 40, point[1] - 40, 80, 80])
        prompt = ([{"type": "rectangle", "data": [box[0], box[1], box[0] + box[2], box[1] + box[3]], "label": 1}]
                  if box else [{"type": "point", "data": point, "label": 1}])
        return _segment(model, frame, sam_prompt=prompt)

    writer = FrameWriter(out, W, H, v["fps_str"])
    ema = last_key = tmpl = None
    pending, prev_gray, prev_bin = [], None, None
    cover, ious, n, keys, unsure = [], [], 0, 0, 0
    seen = vel = None  # last confident (frame, box) and its velocity, for coasting through occlusions

    def emit(m):
        nonlocal ema, prev_bin
        m = m.astype(np.float32)
        ema = m if ema is None else (1 - smooth) * m + smooth * ema
        b = ema > 127
        cover.append(float(b.mean()))
        if prev_bin is not None:
            union = (b | prev_bin).sum()
            ious.append(float((b & prev_bin).sum() / union) if union else 1.0)
        prev_bin = b
        writer.write(ema.astype(np.uint8))

    try:
        for i, f in enumerate(read_frames(ins[0], W, H)):
            n += 1
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                if box and vel and seen[0] < i - stride:  # target hidden: coast on its last velocity
                    dt = min(i - seen[0], 2 * float(v["fps"] or 30))  # ponytail: constant velocity, max 2 s
                    box = [seen[1][0] + vel[0] * dt, seen[1][1] + vel[1] * dt, *seen[1][2:]]
                elif box:
                    box = _flow_box(prev_gray, g, box)
                elif point:
                    pb = _flow_box(prev_gray, g, [point[0] - 20, point[1] - 20, 40, 40])
                    point = [pb[0] + 20, pb[1] + 20]
            prev_gray = g
            if i % stride:
                pending.append(i)
                continue
            m = infer(f)
            if box:  # identity: one subject, matched by overlap + appearance, so crossing subjects can't steal it
                keys += 1
                m, bb, hist, score = _isolate(m, f, box, tmpl)
                if bb and score >= 0.3 and 0.4 < (bb[2] * bb[3]) / (box[2] * box[3]) < 2.5:
                    # follow the subject's centre; adapt size slowly so partial occlusions can't shrink the box away
                    w, h = 0.8 * box[2] + 0.2 * bb[2], 0.8 * box[3] + 0.2 * bb[3]
                    bb = [bb[0] + bb[2] / 2 - w / 2, bb[1] + bb[3] / 2 - h / 2, w, h]
                    if seen:
                        new = [(bb[0] - seen[1][0]) / (i - seen[0]), (bb[1] - seen[1][1]) / (i - seen[0])]
                        vel = new if vel is None else [0.5 * a + 0.5 * b for a, b in zip(vel, new)]
                    seen, box = (i, bb), bb
                    tmpl = hist if tmpl is None else cv2.addWeighted(tmpl, 0.9, hist, 0.1, 0)
                else:
                    unsure += 1
            for j, idx in enumerate(pending):  # linear blend between keyframe masks
                t = (j + 1) / (len(pending) + 1)
                emit((1 - t) * last_key + t * m if last_key is not None else m)
            pending, last_key = [], m.astype(np.float32)
            emit(m)
        for _ in pending:
            emit(last_key)
    finally:
        writer.close()
    if not n:
        raise RuntimeError(f"no frames decoded from {ins[0]}")
    empty = sum(c < 0.002 for c in cover) / len(cover)
    iou = float(np.mean(ious)) if ious else 1.0
    import rembg
    ctx.meta["mask"] = {
        "backend": "opencv-grabcut" if model == "grabcut" else f"rembg {rembg.__version__}:{model}",
        "providers": ORT_PROVIDERS, "frames": n, "inference_size": [W, H], "stride": stride,
        "coverage_mean": round(float(np.mean(cover)), 4), "temporal_iou_mean": round(iou, 4),
        "empty_frame_ratio": round(empty, 4),
        **({"identity_unsure_ratio": round(unsure / keys, 4)} if keys else {}),
        "low_confidence": bool(iou < 0.8 or empty > 0.1 or (keys and unsure / keys > 0.1))}


@op("background")
def background(ctx, ins, p, out):
    """Background workflow. inputs: video, mask[, background image/video]. params:
    mode = remove (alpha output) | blur | replace; blur (sigma, source px, 30); color (solid
    replace, e.g. "black", "0x202020"); feather (mask blur sigma px, 2); refine (edge-aware guided
    upsample radius px, 0 = off, default 4); invert (bool)."""
    v = info(ins[0])["video"]
    W, H = v["width"], v["height"]
    deep = any(b in (v["pix_fmt"] or "") for b in ("10", "12"))
    afmt = "yuva444p10le" if deep else "yuva444p"
    refine, feather = ctx.px(p.get("refine", 4)), p.get("feather", 2)
    fc = f"[0:v]setpts=PTS-STARTPTS,split=3[src][g][b];"
    fc += f"[1:v]setpts=PTS-STARTPTS,scale={W}:{H}:flags=bicubic,format=gray"
    if refine:
        # ffmpeg guided: first pad = guide, second = image to filter (verified empirically)
        fc += f"[mu];[g]format=gray[gy];[gy][mu]guided=radius={refine}:eps=0.01:guidance=on"
    else:
        fc += f"[mu];[g]nullsink;[mu]null"
    fc += (",negate" if p.get("invert") else "") + (f",gblur=sigma={ctx.px(feather)}" if feather else "") + "[m];"
    fc += f"[src]format={afmt}[s];[s][m]alphamerge[fg]"
    mode = p.get("mode", "replace")
    if mode == "remove":
        return ctx.encode(["-i", ins[0], "-i", ins[1]], fc + ";[b]nullsink;[fg]null[v]", "[v]", "0:a?", out, alpha=True)
    args = ["-i", ins[0], "-i", ins[1]]
    if mode == "blur":
        fc += f";[b]gblur=sigma={ctx.px(p.get('blur', 30))}[bg]"
        end = ""
    elif mode == "replace" and len(ins) > 2:
        args += media_in(ins[2])
        fc += f";[b]nullsink;{_fill(2, W, H, v['fps_str'])}[bg]"
        end = ":shortest=1"
    elif mode == "replace":
        fc += f";[b]nullsink;color=c={p.get('color', 'black')}:s={W}x{H}:r={v['fps_str']}[bg]"
        end = ":shortest=1"
    else:
        raise ValueError(f"unknown background mode {mode}")
    fc += f";[bg][fg]overlay=format=auto{end}[v]"
    ctx.encode(args, fc, "[v]", "0:a?", out)


@op("mask_edit")
def mask_edit(ctx, ins, p, out):
    """Hand fixes for a matte (the UI's paint tool writes these). inputs: mask video, paint image (any
    size, stretched over the frame: red = force in, blue = force out, else untouched). params:
    start, end (s, default whole clip); feather (blur px of the painted edge, 2)."""
    v = info(ins[0])["video"]
    W, H, fps = v["width"], v["height"], v["fps"] or 30
    paint = cv2.cvtColor(cv2.imread(ins[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    paint = cv2.resize(paint, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int16)
    add = (paint[..., 0] > 127) & (paint[..., 0] - paint[..., 2] > 60)
    sub = (paint[..., 2] > 127) & (paint[..., 2] - paint[..., 0] > 60)
    f = float(p.get("feather", 2)) * H / (v["height"] or H)
    wa, ws = (cv2.GaussianBlur(x.astype(np.float32), (0, 0), f) if f else x.astype(np.float32) for x in (add, sub))
    a, b = float(p.get("start", 0)), float(p.get("end", 1e9))
    writer = FrameWriter(out, W, H, v["fps_str"])
    n = 0
    try:
        for i, m in enumerate(read_frames(ins[0], W, H)):
            g = m[..., 0].astype(np.float32)
            if a <= i / fps < b:
                g = g * (1 - wa) + 255 * wa
                g = g * (1 - ws)
            writer.write(np.clip(g + 0.5, 0, 255).astype(np.uint8))
            n += 1
    finally:
        writer.close()
    ctx.meta["mask_edit"] = {"frames": n, "added_px": int(add.sum()), "removed_px": int(sub.sum()), "range": [a, min(b, n / fps)]}
