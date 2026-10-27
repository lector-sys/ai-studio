"""Deterministic edit operations. Each op renders inputs -> one lossless intermediate.

Pixel parameters are in *source* pixels; ctx.px/ctx.even rescale them for proxy previews.
"""
import json
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .media import audio_maps, ffmpeg, fpath, info, tool

OPS: dict = {}


def op(name):
    def deco(fn):
        OPS[name] = fn
        return fn
    return deco


@dataclass
class Ctx:
    workdir: Path
    scale: float = 1.0
    commands: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # backend/model versions, quality stats

    def px(self, v):
        return v if isinstance(v, str) else int(round(v * self.scale))

    def even(self, v):
        return v if isinstance(v, str) else max(2, int(round(v * self.scale / 2)) * 2)

    def encode(self, inputs, fc, v, a, out, alpha=False):
        args = [*inputs]
        if fc:
            args += ["-filter_complex", fc]
        if a == "0:a?":  # every decodable audio track of input 0 (iPhone spatial-audio APAC has no decoder)
            amap = audio_maps(args[args.index("-i") + 1])
        else:
            amap = ["-map", a] if a else []
        args += ["-map", v] + amap
        # Lossless intermediates so chained steps don't stack generation loss.
        args += (["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuva444p10le"] if alpha
                 else ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"])
        args += (["-c:a", "flac"] if a else []) + [out]
        self.commands.append(ffmpeg(args))


def vdur(path) -> float:
    d = info(path)
    return (d["video"] or {}).get("duration") or d["duration"]


def has_audio(path) -> bool:
    return info(path)["audio"] is not None


def media_in(path) -> list:
    """Input args for a background/overlay layer: images loop forever, videos loop too."""
    return ["-loop", "1", "-i", path] if info(path)["is_image"] else ["-stream_loop", "-1", "-i", path]


def _fill(i, w, h, fps):
    """Background layer conformed to the main clip (looped images default to 25 fps otherwise)."""
    return f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps={fps}"


def _between(p):
    s, e = p.get("start"), p.get("end")
    if s is None and e is None:
        return ""
    return f":enable='between(t,{s or 0},{e if e is not None else 1e9})'"


@op("trim")
def trim(ctx, ins, p, out):
    """Frame-accurate cut. params: start (s), end (s) | duration (s)."""
    start = float(p.get("start", 0))
    dur = p.get("duration", (float(p["end"]) - start) if "end" in p else None)
    args = ["-ss", start, "-i", ins[0]] + (["-t", dur] if dur is not None else [])
    ctx.encode(args, None, "0:v", "0:a?", out)


@op("crop")
def crop(ctx, ins, p, out):
    """params: x, y, w, h (source px)."""
    fc = f"[0:v]crop={ctx.even(p['w'])}:{ctx.even(p['h'])}:{ctx.px(p.get('x', 0))}:{ctx.px(p.get('y', 0))}[v]"
    ctx.encode(["-i", ins[0]], fc, "[v]", "0:a?", out)


@op("resize")
def resize(ctx, ins, p, out):
    """params: width, height, mode = fit (letterbox/pillarbox) | fill (crop) | stretch."""
    w, h, mode = ctx.even(p["width"]), ctx.even(p["height"]), p.get("mode", "fit")
    f = {"fit": f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
         "fill": f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}",
         "stretch": f"scale={w}:{h}:flags=lanczos"}[mode]
    ctx.encode(["-i", ins[0]], f"[0:v]{f},setsar=1[v]", "[v]", "0:a?", out)


@op("rotate")
def rotate(ctx, ins, p, out):
    """params: degrees (any; 90/180/270 are lossless transposes), flip = h | v."""
    deg = float(p.get("degrees", 0)) % 360
    f = {0: "null", 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}.get(
        deg, f"rotate={deg}*PI/180:fillcolor=black")
    f += {"h": ",hflip", "v": ",vflip"}.get(p.get("flip"), "")
    ctx.encode(["-i", ins[0]], f"[0:v]{f}[v]", "[v]", "0:a?", out)


def _atempo(f):
    out = []
    while f > 2:
        out.append("atempo=2")
        f /= 2
    while f < 0.5:
        out.append("atempo=0.5")
        f /= 0.5
    return ",".join(out + [f"atempo={f}"])


@op("speed")
def speed(ctx, ins, p, out):
    """params: factor (2 = twice as fast, 0.5 = half speed; pitch preserved)."""
    f = float(p["factor"])
    fc = f"[0:v]setpts=PTS/{f},fps={info(ins[0])['video']['fps_str']}[v]"  # keep source frame rate
    a = None
    if has_audio(ins[0]):
        fc += f";[0:a]{_atempo(f)}[a]"
        a = "[a]"
    ctx.encode(["-i", ins[0]], fc, "[v]", a, out)


@op("reverse")
def reverse(ctx, ins, p, out):
    """Play backwards. ponytail: buffers the whole clip in RAM; trim first for long clips."""
    a = "[a]" if has_audio(ins[0]) else None
    fc = "[0:v]reverse[v]" + (";[0:a]areverse[a]" if a else "")
    ctx.encode(["-i", ins[0]], fc, "[v]", a, out)


@op("fps")
def fps(ctx, ins, p, out):
    """Frame-rate conversion by drop/duplicate. params: fps (e.g. 30, "60000/1001")."""
    ctx.encode(["-i", ins[0]], f"[0:v]fps={p['fps']}[v]", "[v]", "0:a?", out)


@op("freeze")
def freeze(ctx, ins, p, out):
    """Hold the frame at `at` (s) for `duration` (s); audio gets matching silence."""
    at, d = float(p["at"]), float(p["duration"])
    fc = (f"[0:v]split[x][y];[x]trim=end={at},setpts=PTS-STARTPTS[p];"
          f"[y]trim=start={at},setpts=PTS-STARTPTS,tpad=start_mode=clone:start_duration={d}[q];"
          f"[p][q]concat=n=2:v=1:a=0[v]")
    a = None
    if has_audio(ins[0]):
        fc += (f";[0:a]asplit[ax][ay];[ax]atrim=end={at},asetpts=PTS-STARTPTS[pa];"
               f"[ay]atrim=start={at},asetpts=PTS-STARTPTS,adelay={int(d * 1000)}:all=1[qa];"
               f"[pa][qa]concat=n=2:v=0:a=1[a]")
        a = "[a]"
    ctx.encode(["-i", ins[0]], fc, "[v]", a, out)


def _normalized(ins):
    """Conform every input to the first one's size/fps/pix_fmt; returns (args, fc parts, has_audio)."""
    r = info(ins[0])["video"]
    w, h, f, pf = r["width"], r["height"], r["fps_str"], r["pix_fmt"]
    audio = any(has_audio(x) for x in ins)
    args, parts = [], []
    for i, path in enumerate(ins):
        args += ["-i", path]
        parts.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                     f"setsar=1,fps={f},format={pf},setpts=PTS-STARTPTS[v{i}]")
        if audio:
            parts.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                         f"asetpts=PTS-STARTPTS[a{i}]" if has_audio(path)
                         else f"anullsrc=r=48000:cl=stereo,atrim=duration={vdur(path)}[a{i}]")
    return args, parts, audio


@op("concat")
def concat(ctx, ins, p, out):
    """Join clips in order; later clips are conformed to the first clip's format."""
    args, parts, audio = _normalized(ins)
    n = len(ins)
    streams = "".join(f"[v{i}]" + (f"[a{i}]" if audio else "") for i in range(n))
    # concat starts each clip at the end of its longest stream: a clip whose video ends a frame before its audio
    # leaves a timestamp hole, so re-grid the joined video (fills holes with the previous frame, keeps sync).
    fc = (";".join(parts) + f";{streams}concat=n={n}:v=1:a={int(audio)}[vc]" + ("[a]" if audio else "")
          + f";[vc]fps={info(ins[0])['video']['fps_str']}[v]")
    ctx.encode(args, fc, "[v]", "[a]" if audio else None, out)


@op("transition")
def transition(ctx, ins, p, out):
    """Join two clips with an overlap. params: type (xfade name: fade, dissolve, wipeleft,
    slideright, circleopen, ...), duration (s)."""
    kind = str(p.get("type", "fade"))
    if not kind.isalnum():
        raise ValueError(f"transition type must be an xfade name (letters/digits), got {kind!r}")
    args, parts, audio = _normalized(ins[:2])
    d = float(p.get("duration", 1))
    off = vdur(ins[0]) - d
    fc = ";".join(parts) + f";[v0][v1]xfade=transition={kind}:duration={d}:offset={off}[v]"
    if audio:
        fc += f";[a0][a1]acrossfade=d={d}[a]"
    ctx.encode(args, fc, "[v]", "[a]" if audio else None, out)


@op("fade")
def fade(ctx, ins, p, out):
    """Fade from/to black (and audio). params: in (s), out (s)."""
    D, fi, fo = vdur(ins[0]), float(p.get("in", 0)), float(p.get("out", 0))
    v = ["null"] + ([f"fade=t=in:d={fi}"] if fi else []) + ([f"fade=t=out:st={D - fo}:d={fo}"] if fo else [])
    fc = f"[0:v]{','.join(v)}[v]"
    a = None
    if has_audio(ins[0]):
        af = ["anull"] + ([f"afade=t=in:d={fi}"] if fi else []) + ([f"afade=t=out:st={D - fo}:d={fo}"] if fo else [])
        fc += f";[0:a]{','.join(af)}[a]"
        a = "[a]"
    ctx.encode(["-i", ins[0]], fc, "[v]", a, out)


@op("overlay")
def overlay(ctx, ins, p, out):
    """Picture-in-picture / logo / watermark. inputs: main, layer (image or video).
    params: x, y (px or ffmpeg expr like "W-w-40"), width (px), opacity (0-1), start, end (s)."""
    img = info(ins[1])["is_image"]
    args = ["-i", ins[0]] + (["-loop", "1", "-i", ins[1]] if img else ["-i", ins[1]])
    o = "[1:v]format=rgba"
    w = p.get("width") or (info(ins[1])["video"]["width"] if ctx.scale != 1 else None)
    if w:
        o += f",scale={ctx.even(w)}:-2"
    if float(p.get("opacity", 1)) < 1:
        o += f",colorchannelmixer=aa={p['opacity']}"
    if not img:
        o += f",setpts=PTS-STARTPTS+{float(p.get('start', 0))}/TB"
    end = ":shortest=1" if img else ":eof_action=pass"
    fc = (f"{o}[ov];[0:v][ov]overlay=x={ctx.px(p.get('x', 0))}:y={ctx.px(p.get('y', 0))}"
          f":format=auto{end}{_between(p)}[v]")
    ctx.encode(args, fc, "[v]", "0:a?", out)


@op("text")
def text(ctx, ins, p, out):
    """Titles / lower thirds. params: text, x, y (px or expr), size (px), color, box (bool),
    boxcolor, font (path), start, end."""
    tf = ctx.workdir / f"text-{abs(hash(p['text']))}.txt"
    tf.write_text(p["text"], encoding="utf-8")
    font = p.get("font") or next((f for f in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                                  if Path(f).exists()), None)
    f = (f"drawtext=textfile={fpath(tf)}:fontsize={ctx.px(p.get('size', 64))}:fontcolor={p.get('color', 'white')}"
         f":x={ctx.px(p.get('x', '(w-text_w)/2'))}:y={ctx.px(p.get('y', 'h-text_h-80'))}")
    if font:
        f += f":fontfile={fpath(font)}"
    if p.get("box"):
        f += f":box=1:boxcolor={p.get('boxcolor', 'black@0.5')}:boxborderw={ctx.px(16)}"
    ctx.encode(["-i", ins[0]], f"[0:v]{f}{_between(p)}[v]", "[v]", "0:a?", out)


@op("subtitles")
def subtitles(ctx, ins, p, out):
    """Burn in an .srt/.ass file. params: file, style (ASS force_style, e.g. "Fontsize=28")."""
    f = f"subtitles={fpath(p['file'])}" + (f":force_style='{p['style']}'" if p.get("style") else "")
    ctx.encode(["-i", ins[0]], f"[0:v]{f}[v]", "[v]", "0:a?", out)


@op("audio")
def audio(ctx, ins, p, out):
    """Audio replace/mix. inputs: video[, audio]. params: mode = replace | mix | duck | mute,
    volume (original, default 1), music_volume (new, default 1), delay (s the new audio starts at, 0).
    Duck = sidechain-compress the new audio under the original (music under dialogue). Output keeps
    the video's duration."""
    mode = p.get("mode", "replace")
    if mode == "mute":
        return ctx.encode(["-i", ins[0]], None, "0:v", None, out)
    D, v0, v1 = vdur(ins[0]), p.get("volume", 1), p.get("music_volume", 1)
    delay = f"adelay={round(float(p['delay']) * 1000)}:all=1," if p.get("delay") else ""
    new = f"[1:a]{delay}volume={v1},apad,atrim=duration={D}"
    if mode == "replace" or not has_audio(ins[0]):
        fc = f"{new}[a]"
    elif mode == "mix":
        fc = f"[0:a]volume={v0}[a0];{new}[a1];[a0][a1]amix=inputs=2:duration=first:normalize=0[a]"
    elif mode == "duck":
        fc = (f"{new}[m];[0:a]volume={v0},asplit[d][sc];"
              f"[m][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[md];"
              f"[d][md]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        raise ValueError(f"unknown audio mode {mode}")
    ctx.encode(["-i", ins[0], "-i", ins[1]], fc, "0:v", "[a]", out)


@op("loudnorm")
def loudnorm(ctx, ins, p, out):
    """EBU R128 loudness normalization. params: I (LUFS, default -16), TP (-1.5), LRA (11),
    linear (bool): one measured constant gain + delay-compensated peak limiter at TP instead of loudnorm's
    dynamic mode, which lifts the noise floor between phrases."""
    I, TP = float(p.get("I", -16)), float(p.get("TP", -1.5))
    f = f"loudnorm=I={I}:TP={TP}:LRA={p.get('LRA', 11)}"
    if p.get("linear"):
        r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", ins[0], "-vn", "-af",
                            f + ":print_format=json", "-f", "null", "-"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        m = json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
        f = f"volume={I - float(m['input_i']):.2f}dB,alimiter=limit={10 ** (TP / 20):.4f}:level=false:latency=1"
        ctx.meta["loudnorm"] = {"input_i": m["input_i"], "gain_db": round(I - float(m["input_i"]), 2)}
    f += ",aresample=48000,asetnsamples=n=4096:p=0"  # p=0: no padding, sample count kept
    ctx.encode(["-i", ins[0]], f"[0:a]{f}[a]", "0:v", "[a]", out)


@lru_cache(64)
def filter_delay(filters: str, rate: int) -> int:
    """Measured latency (samples) a filter chain adds, from a tone burst through the real chain."""
    import numpy as np

    def pcm(chain):
        cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-f", "lavfi", "-i",
               f"aevalsrc='if(between(t,0.5,0.52),sin(2*PI*1000*t)*sin(PI*(t-0.5)/0.02),0)':s={rate}:d=1.5",
               "-af", chain, "-f", "f32le", "-ac", "1", "-"]
        return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)
    a, b = pcm("anull"), pcm(filters)
    n = min(len(a), len(b))
    lag = int(np.argmax(np.correlate(b[:n], a[:n], "full"))) - (n - 1)
    return max(0, lag)


def clean_audio(p, rate) -> tuple[str, dict]:
    """Shared dialogue cleanup (high-pass + FFT denoise) with the chain's measured delay removed,
    so speech stays in sync and the sample count is unchanged. params: highpass (Hz, 80),
    denoise (dB noise floor, -25), reduction (dB, afftdn default 12), track_noise (bool),
    smoothing (afftdn gain smoothing radius 0-50)."""
    hp, nf = float(p.get("highpass", 80)), float(p.get("denoise", -25))
    if not (20 <= hp <= 1000 and -80 <= nf <= -20):
        raise ValueError("audio cleanup: highpass must be 20-1000 Hz and denoise -80..-20 dB")
    if p.get("noise_print"):  # the spectral gate replaces afftdn (audio_clean applies it after this chain)
        filters = f"highpass=f={hp}"
    else:
        filters = (f"highpass=f={hp},afftdn=nf={nf}" + (f":nr={float(p['reduction'])}" if p.get("reduction") else "")
                   + (":tn=1" if p.get("track_noise") else "") + (f":gs={int(p['smoothing'])}" if p.get("smoothing") else ""))
    delay = filter_delay(filters, rate)
    chain = (f"apad=pad_dur={delay / rate},{filters},atrim=start_sample={delay},asetpts=PTS-STARTPTS,asetnsamples=n=4096:p=0"
             if delay else filters)
    return chain, {"filters": filters, "delay_samples": delay, "sample_rate": rate}


def _detap(x, r, jump_db, broad):
    """Hand/nail taps on a mic: STFT frames (~5 ms hop) where >= `broad` of the 1-12 kHz bins jump `jump_db`
    above their own 120 ms temporal median are pulled down to that median (phase kept). Steady speech never
    jumps like that; a sharp consonant can, so keep jump_db >= 6."""
    import numpy as np
    from scipy.ndimage import maximum_filter1d, median_filter
    from scipy.signal import istft, stft
    n, hop = 1024, 256
    f, _, Z = stft(x, r, nperseg=n, noverlap=n - hop)
    mag = np.abs(Z) + 1e-12
    med = median_filter(mag, size=(1, int(0.12 * r / hop) | 1), mode="nearest")
    hot = mag > med * 10 ** (jump_db / 20)
    tap = maximum_filter1d((hot[(f >= 1000) & (f <= 12000)].mean(0) >= broad).astype(int), 3).astype(bool)
    over = hot & tap[None, :]
    scale = np.ones_like(mag)
    scale[over] = med[over] / mag[over]
    _, y = istft(Z * scale, r, nperseg=n, noverlap=n - hop)
    return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)], int(tap.sum())


def _gate(x, r, ranges, reduction):
    """Stationary-noise gate: noise print from `ranges` (s, pure background); STFT bins near it are
    attenuated by `reduction` dB with a smoothed soft mask (no musical noise)."""
    import numpy as np
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft
    _, t, Z = stft(x, r, nperseg=2048, noverlap=1536)
    mag = np.abs(Z)
    sel = np.zeros(len(t), bool)
    for a, b in ranges:
        sel |= (t >= a) & (t <= b)
    if sel.sum() < 8:
        raise ValueError("noise_print: ranges hold too little audio; give >= 0.1 s of pure background")
    noise = mag[:, sel]
    thresh = noise.mean(1, keepdims=True) + 1.5 * noise.std(1, keepdims=True)
    mask = uniform_filter(np.clip(20 * np.log10(mag / thresh + 1e-12) / 6, 0, 1), size=(5, 7))
    floor = 10 ** (-reduction / 20)
    _, y = istft(Z * (floor + (1 - floor) * mask), r, nperseg=2048, noverlap=1536)
    return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]


def spectral_clean(src, dst, ranges=None, reduction=20, taps=None) -> dict:
    """Tap suppression, then the noise-print gate, per channel. STFT/ISTFT is zero-phase: timing and
    sample count are unchanged. ponytail: whole clip in RAM; chunk it for hour-long audio."""
    import numpy as np
    from scipy.io import wavfile
    r, x = wavfile.read(src)
    x = x[:, None] if x.ndim == 1 else x
    out, frames = np.empty_like(x), 0
    for ch in range(x.shape[1]):
        y = x[:, ch].astype(np.float64)
        if taps is not None:
            y, n = _detap(y, r, float(taps.get("jump_db", 6)), float(taps.get("broad", 0.3)))
            frames = max(frames, n)
        if ranges:
            y = _gate(y, r, ranges, reduction)
        out[:, ch] = y
    wavfile.write(dst, r, out.astype(np.float32))
    return {"tap_frames": frames} if taps is not None else {}


DF_WORKER = Path(__file__).resolve().parents[1] / "workers" / "deepfilter"


@op("speech_enhance")
def speech_enhance(ctx, ins, p, out):
    """AI speech enhancement (DeepFilterNet3, CPU, ~10 s per minute): keeps voices, removes background
    noise, hum and mic-handling knocks. params: mono (false; true mixes the mics first, keeps more voice
    detail on phone recordings), limit (max dB removed; unset = full; 12-24 sounds less processed),
    highpass (Hz, 70; after enhancement). Output is stereo, same length and timing.
    Worker: workers/deepfilter (env STUDIO_DF_PYTHON overrides its python)."""
    import os
    py = os.environ.get("STUDIO_DF_PYTHON", str(DF_WORKER / ".venv" / "Scripts" / "python.exe"))
    if not Path(py).exists():
        raise RuntimeError(f"speech_enhance worker missing: {py} (see docs/LOCAL_WORKERS.md, deepfilter)")
    raw, clean, fin = (Path(out).with_suffix(s) for s in (".in.wav", ".df.wav", ".fin.wav"))
    try:
        ctx.commands.append(ffmpeg(["-i", ins[0], "-vn", *(["-ac", 1] if p.get("mono") else []), "-ar", 48000,
                                    "-c:a", "pcm_f32le", raw]))
        cmd = [py, "-I", str(DF_WORKER / "worker.py"), str(raw), str(clean)]
        cmd += ["--limit", str(float(p["limit"]))] if p.get("limit") else []
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            raise RuntimeError(f"speech_enhance worker failed: {r.stderr[-2000:]}")
        ctx.meta["speech_enhance"] = json.loads(r.stdout.strip().splitlines()[-1])
        ctx.commands.append(ffmpeg(["-i", clean, "-af", f"highpass=f={float(p.get('highpass', 70))}", "-ac", 2,
                                    "-c:a", "pcm_f32le", fin]))
        ctx.encode(["-i", ins[0], "-i", fin], None, "0:v", "1:a", out)
    finally:
        for f in (raw, clean, fin):
            f.unlink(missing_ok=True)


@op("gain")
def gain(ctx, ins, p, out):
    """Volume automation, e.g. matching an interviewer's level to the guest's. params: ranges
    [[start, end, dB], ...] (s; constant gain inside, 0 dB outside), ramp (s, 0.08: linear in dB,
    centred on each edge). Sample count and timing unchanged."""
    R = float(p.get("ramp", 0.08))
    terms = []
    for a, b, db in p["ranges"]:
        a, b, db = float(a), float(b), float(db)
        if not (0 <= a < b and -60 <= db <= 30 and R > 0):
            raise ValueError(f"gain: bad range {[a, b, db]} (need 0 <= start < end, -60..30 dB, ramp > 0)")
        terms.append(f"{db}*min(1,max(0,(t-{a})/{R}+0.5))*min(1,max(0,({b}-t)/{R}+0.5))")
    # 64-sample frames: eval=frame steps every 1.3 ms, inaudible inside an 80 ms ramp
    f = f"asetnsamples=n=64:p=0,volume='pow(10,({'+'.join(terms)})/20)':eval=frame,asetnsamples=n=4096:p=0"
    ctx.encode(["-i", ins[0]], f"[0:a]{f}[a]", "0:v", "[a]", out)


@op("audio_clean")
def audio_clean(ctx, ins, p, out):
    """Dialogue cleanup: high-pass + FFT denoise, filter delay measured and compensated.
    params: highpass (Hz, 80), denoise (dB floor, -25), reduction (dB), track_noise, smoothing,
    noise_print [[start, end], ...] (s of pure background, e.g. pauses): learn that noise and gate it
    instead of afftdn (much stronger on steady fan/AC hum); reduction then defaults to 20 dB.
    taps (true or {jump_db 6, broad 0.3}): suppress hand/nail taps on the mic (short broadband hits)."""
    taps = {} if p.get("taps") is True else (p.get("taps") or None)
    chain, ctx.meta["audio_clean"] = clean_audio(p, info(ins[0])["audio"]["sample_rate"])
    if not p.get("noise_print") and taps is None:
        return ctx.encode(["-i", ins[0]], f"[0:a]{chain}[a]", "0:v", "[a]", out)
    hp, gated = Path(out).with_suffix(".hp.wav"), Path(out).with_suffix(".gated.wav")
    try:
        ctx.commands.append(ffmpeg(["-i", ins[0], "-vn", "-af", chain, "-c:a", "pcm_f32le", hp]))
        ctx.meta["audio_clean"] |= spectral_clean(hp, gated, p.get("noise_print"), float(p.get("reduction", 20)), taps)
        ctx.meta["audio_clean"]["noise_print"] = p.get("noise_print")
        ctx.encode(["-i", ins[0], "-i", gated], None, "0:v", "1:a", out)
    finally:
        hp.unlink(missing_ok=True)
        gated.unlink(missing_ok=True)


@op("color")
def color(ctx, ins, p, out):
    """Basic grade. params: brightness (-1..1, 0), contrast (1), saturation (1), gamma (1),
    temperature (Kelvin, 6500 = neutral), lut (.cube path). HDR/10-bit safe (never drop to 8-bit):
    tone (luma power, >1 darkens mid-tones, black and peak white kept), luma (gain above black, <1 darkens all).
    brightness/contrast/saturation/gamma use `eq`, which is 8-bit only: avoid them on HDR."""
    ctx.encode(["-i", ins[0]], f"[0:v]{grade(p)}[v]", "[v]", "0:a?", out)


def grade(p) -> str:
    f = []
    if any(k in p for k in ("brightness", "contrast", "saturation", "gamma")):
        f.append(f"eq=brightness={p.get('brightness', 0)}:contrast={p.get('contrast', 1)}"
                 f":saturation={p.get('saturation', 1)}:gamma={p.get('gamma', 1)}")
    if p.get("tone", 1) != 1 or p.get("luma", 1) != 1:
        n = "clip((val-minval)/(maxval-minval),0,1)"  # 0..1 over the legal luma range at any bit depth
        f.append(f"lutyuv=y='clip(minval+(maxval-minval)*{float(p.get('luma', 1))}*pow({n},{float(p.get('tone', 1))})"
                 f",minval,maxval)'")
    if p.get("temperature"):
        f.append(f"colortemperature=temperature={p['temperature']}")
    if p.get("lut"):
        f.append(f"lut3d=file={fpath(p['lut'])}")
    return ",".join(f) or "null"


@op("chromakey")
def chromakey(ctx, ins, p, out):
    """Green/blue screen key. inputs: video[, background]. params: color ("0x00FF00"),
    similarity (0.15), blend (0.05). Without a background the result keeps alpha."""
    key = f"[0:v]chromakey=color={p.get('color', '0x00FF00')}:similarity={p.get('similarity', 0.15)}:blend={p.get('blend', 0.05)}"
    if len(ins) < 2:
        return ctx.encode(["-i", ins[0]], key + "[v]", "[v]", "0:a?", out, alpha=True)
    v = info(ins[0])["video"]
    fc = f"{_fill(1, v['width'], v['height'], v['fps_str'])}[bg];{key}[fg];[bg][fg]overlay=format=auto:shortest=1[v]"
    ctx.encode(["-i", ins[0], *media_in(ins[1])], fc, "[v]", "0:a?", out)
