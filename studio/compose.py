"""`compose`: timed image/text-card layers (+ optional dialogue cleanup) over one source in a single pass.
The base picture is never scaled, graded or re-timed: cards are converted into the source's colour space
(HLG/PQ through an explicit zscale transfer) and overlaid; frame timestamps pass through untouched."""
import math
from fractions import Fraction

from .media import info
from .ops import clean_audio, op

ANIMATIONS = {"none", "fade", "slide-scale"}
# base pix_fmt -> (overlay format, card pix_fmt, alpha pix_fmt)
FORMATS = {"yuv420p": ("yuv420", "yuva420p", "gray"), "yuv420p10le": ("yuv420p10", "yuva420p10le", "gray10le"),
           "yuv422p": ("yuv422", "yuva422p", "gray"), "yuv422p10le": ("yuv422p10", "yuva422p10le", "gray10le"),
           "yuv444p": ("yuv444", "yuva444p", "gray"), "yuv444p10le": ("yuv444p10", "yuva444p10le", "gray10le")}
SDR_MATRIX = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "unknown": "bt709", None: "bt709"}
HDR_TRC = {"arib-std-b67": "arib-std-b67", "smpte2084": "smpte2084"}
LAYER_KEYS = {"input", "start", "end", "x", "y", "width", "height", "animation", "fade", "entrance", "slide",
              "initial_scale", "opacity"}


def _num(layer, k, default=None, lo=-math.inf, hi=math.inf):
    v = layer.get(k, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"compose layer {k} must be a finite number, got {v!r}")
    if not lo <= v <= hi:
        raise ValueError(f"compose layer {k}={v} outside [{lo}, {hi}]")
    return float(v)


def validate_compose(ins, p, scale=1.0) -> dict:
    """Check params against the actual inputs; returns the normalized layer list and colour plan."""
    if len(ins) < 2:
        raise ValueError("compose needs a base clip and at least one layer image")
    v = info(ins[0])["video"]
    if not v:
        raise ValueError("compose base has no video")
    if v["pix_fmt"] not in FORMATS:
        raise ValueError(f"compose: base pix_fmt {v['pix_fmt']} unsupported; convert to one of {sorted(FORMATS)}")
    if p.get("color_policy", "preserve") != "preserve":
        raise ValueError("compose color_policy must be 'preserve' (conversions are separate, explicit steps)")
    trc = v["color_transfer"]
    if v["hdr"] and v["color_primaries"] != "bt2020":
        raise ValueError(f"compose: HDR with {v['color_primaries']} primaries is unsupported; needs bt2020")
    if not v["hdr"] and v["color_space"] not in SDR_MATRIX:
        raise ValueError(f"compose: SDR matrix {v['color_space']} unsupported")
    unknown = set(p) - {"layers", "audio", "color_policy", "graphics_white_nits"}
    if unknown:
        raise ValueError(f"compose: unknown params {sorted(unknown)}")
    dur = v["duration"] or math.inf
    layers = []
    for n, raw in enumerate(p.get("layers") or []):
        if set(raw) - LAYER_KEYS:
            raise ValueError(f"compose layer {n}: unknown keys {sorted(set(raw) - LAYER_KEYS)}")
        i = raw.get("input")
        if not isinstance(i, int) or isinstance(i, bool) or not 1 <= i < len(ins):
            raise ValueError(f"compose layer {n}: input must be an index 1..{len(ins) - 1}")
        if not info(ins[i])["is_image"]:
            raise ValueError(f"compose layer {n}: input {i} must be an image (render text/cards to PNG first)")
        L = {"input": i, "start": _num(raw, "start", lo=0), "end": _num(raw, "end", lo=0),
             "x": _num(raw, "x"), "y": _num(raw, "y"), "width": _num(raw, "width", lo=2), "height": _num(raw, "height", lo=2),
             "animation": raw.get("animation", "slide-scale"), "fade": _num(raw, "fade", 0.14, 0, 5),
             "entrance": _num(raw, "entrance", 0.22, 0.01, 5), "slide": _num(raw, "slide", 46),
             "initial_scale": _num(raw, "initial_scale", 0.955, 0.1, 1), "opacity": _num(raw, "opacity", 1, 0, 1)}
        if L["animation"] not in ANIMATIONS:
            raise ValueError(f"compose layer {n}: animation must be one of {sorted(ANIMATIONS)}")
        if L["end"] <= L["start"] or L["start"] >= dur:
            raise ValueError(f"compose layer {n}: needs start < end and start inside the clip ({dur:.3f} s)")
        if 2 * L["fade"] > L["end"] - L["start"]:
            raise ValueError(f"compose layer {n}: fades longer than the layer")
        layers.append(L)
    if not layers:
        raise ValueError("compose needs at least one layer")
    nits = _num(p, "graphics_white_nits", 203, 50, 1000)
    return {"layers": layers, "base": v, "hdr": HDR_TRC.get(trc) if v["hdr"] else None,
            "formats": FORMATS[v["pix_fmt"]], "nits": nits, "scale": scale}


def _card(i, L, plan, t0, rate, ctx):
    """Filter chain turning input i into an animated card in the base's colour space."""
    ofmt, cfmt, afmt = plan["formats"]
    a, b = t0 + L["start"], t0 + L["end"]
    w0, h0 = ctx.even(L["width"]), ctx.even(L["height"])
    f = f"[{i}:v]format=rgba"
    if L["opacity"] < 1:
        f += f",colorchannelmixer=aa={L['opacity']}"
    f += f",scale={w0}:{h0}:flags=lanczos,format=rgba"
    # Colour + alpha conversion happens once at full card size; per-frame animation scaling comes after,
    # so every branch sees the same frame size (a size change mid-branch corrupts the alpha edge).
    if plan["hdr"]:  # sRGB card -> BT.2020 HLG/PQ; alpha carried separately so it stays straight
        f += (f",split[r{i}][m{i}];[m{i}]alphaextract,format={afmt}[a{i}];"
              f"[r{i}]format=gbrp16le,zscale=pin=709:tin=iec61966-2-1:min=gbr:rin=full:p=2020:t={plan['hdr']}"
              f":m=2020_ncl:r=limited:npl={plan['nits']},format={cfmt.replace('yuva', 'yuv')}[h{i}];"
              f"[h{i}][a{i}]alphamerge,format={cfmt}")
    else:
        f += f",scale=out_color_matrix={SDR_MATRIX[plan['base']['color_space']]}:out_range=tv,format={cfmt}"
    # Repeat the one converted still inside the graph (deterministic: overlay pulls frames on demand; separate
    # looping image inputs raced the base decoder, giving 1-2 frame lag and run-to-run differences).
    # Frame 0 is a transparent copy one frame before the cue: `fade` zeroes the first frame it sees, and the
    # fade-in starts there, so the first enabled frame is visible (mirroring the last fade-out frame).
    r = Fraction(rate)
    n = math.ceil((b - a) * r) + 1
    f += (f",split[s{i}][z{i}];[z{i}]lutyuv=a=0[t{i}];[s{i}]loop=loop={n - 1}:size=1[l{i}];"
          f"[t{i}][l{i}]concat=n=2:v=1:a=0,settb=1/{r.numerator * 1000},"
          f"setpts=({a}+(N-1)*{r.denominator}/{r.numerator})/TB"
          ",trim=start=0")  # a cue at 0 puts the lead-in at -1 frame; `fade` then never shows the card
    dy = ""
    if L["animation"] == "slide-scale":
        ease = f"min(1,max(0,(t-{a})/{L['entrance']}))"
        k = f"({L['initial_scale']}+(1-{L['initial_scale']})*(1-pow(1-{ease},3)))"
        # scale about the centre, then pad back to the full card box: constant frame size downstream
        f += (f",scale=w='2*trunc({w0}*{k}/2)':h='2*trunc({h0}*{k}/2)':eval=frame:flags=lanczos"
              f",pad=w={w0}:h={h0}:x='(ow-iw)/2':y='(oh-ih)/2':color=black@0:eval=frame")
        dy = f"+{ctx.px(L['slide'])}*pow(1-{ease},3)"
    if L["animation"] != "none" and L["fade"]:
        f += (f",fade=t=in:st={max(0.0, a - 1 / float(r))}:d={L['fade']}:alpha=1"
              f",fade=t=out:st={b - L['fade']}:d={L['fade']}:alpha=1")
    pos = f"x={ctx.px(L['x'])}:y='{ctx.px(L['y'])}{dy}'"
    over = (f"overlay={pos}:format={ofmt}:alpha=straight:eof_action=pass:repeatlast=0"
            f":enable='gte(t,{a})*lt(t,{b})'")
    return f, over


@op("compose")
def compose(ctx, ins, p, out):
    """Timed image/text-card layers over the base clip in one pass, base picture/clock/colour preserved.
    inputs: base clip, then layer images (rounded cards, labels, logos; render text to PNG first).
    params: layers [{input (index into inputs, >=1), start, end (s, half-open [start, end)), x, y,
    width, height (source px, card box top-left + size), animation = slide-scale | fade | none,
    fade (0.14 s), entrance (0.22 s), slide (46 px), initial_scale (0.955), opacity (1)}];
    audio {highpass, denoise, reduction} (optional dialogue cleanup, delay-compensated);
    graphics_white_nits (HDR card white, 203 per BT.2408); color_policy = preserve."""
    plan = validate_compose(ins, p, ctx.scale)
    v = plan["base"]
    t0, rate = v["start_time"] or 0, v["r_frame_rate"] or v["fps_str"]
    args, parts, cur = ["-i", ins[0]], [], "0:v"
    for n, L in enumerate(plan["layers"]):
        i = n + 1
        args += ["-framerate", rate, "-i", ins[L["input"]]]
        chain, over = _card(i, L, plan, t0, rate, ctx)
        parts += [f"{chain}[c{i}]", f"[{cur}][c{i}]{over}[v{i}]"]
        cur = f"v{i}"
    tags = ":".join(f"{k}={v[s]}" for k, s in (("color_primaries", "color_primaries"), ("color_trc", "color_transfer"),
                                                ("colorspace", "color_space"), ("range", "color_range"))
                    if v.get(s) and v[s] not in ("unknown", "reserved"))
    parts.append(f"[{cur}]" + (f"setparams={tags}" if tags else "null") + "[v]")
    a = "0:a?"
    if p.get("audio") is not None:
        au = info(ins[0])["audio"]
        if not au:
            raise ValueError("compose audio cleanup requested but the base has no audio")
        if p["audio"].get("noise_print"):
            raise ValueError("compose audio cleanup has no noise_print; use an audio_clean step")
        chain, ctx.meta["audio"] = clean_audio(p["audio"], au["sample_rate"])
        parts.append(f"[0:a:0]{chain}[a]")
        a = "[a]"
    args += ["-fps_mode:v", "passthrough", "-enc_time_base:v", v["time_base"]]
    ctx.encode(args, ";".join(parts), "[v]", a, out)
    ctx.meta["compose"] = {"layers": len(plan["layers"]), "hdr_transfer": plan["hdr"],
                           "graphics_white_nits": plan["nits"] if plan["hdr"] else None,
                           "picture": "base not scaled, graded or re-timed"}
