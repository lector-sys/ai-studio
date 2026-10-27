"""Reusable creative recipes: a validated request expands into an ordinary version-1 job manifest
(inspect/check/preview/render it without the recipe system). Text is rasterized here with Pillow from
literal strings (never passed into FFmpeg expressions); cards are cached by content."""
import json
import math
from hashlib import sha256 as _h
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import jobs, media

CATALOG = json.loads(files("studio").joinpath("recipes.json").read_text(encoding="utf-8"))
_CODE = _h(Path(__file__).read_bytes()).hexdigest()[:12]  # editing the card renderer invalidates cached cards
PALETTE = {"card": "#f6efe5", "edge": "#b69964", "title": "#3e3025", "subtitle": "#806a51",
           "label": "#1d1a17", "label_text": "#f6efe5"}
_FONTS = {"title": ["C:/Windows/Fonts/georgiai.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"],
          "body": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
          "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]}


def default_font(kind) -> str:
    f = next((f for f in _FONTS[kind] if Path(f).exists()), None)
    if not f:
        raise FileNotFoundError(f"no default {kind} font found; pass a font path")
    return f


def listing() -> dict:
    return CATALOG


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(Strict):
    start: float = Field(ge=0)
    end: float
    x: float
    y: float
    width: float = Field(gt=1)
    height: float = Field(gt=1)
    animation: Literal["slide-scale", "fade", "none"] = "slide-scale"
    # compose motion overrides (None = compose defaults)
    entrance: float | None = Field(None, gt=0, le=5)
    slide: float | None = None
    initial_scale: float | None = Field(None, gt=0, le=1)
    fade: float | None = Field(None, ge=0, le=5)

    @model_validator(mode="after")
    def _order(self):
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class PopupItem(Box):
    asset: str
    title: str = Field(min_length=1)
    subtitle: str = ""


class LabelItem(Box):
    text: str = Field(min_length=1)
    style: Literal["pill", "shadow"] = "pill"  # shadow = no box, text with a soft drop shadow
    font: str | None = None  # per-item override of the request font
    palette: dict[str, str] = Field(default_factory=dict)  # per-item override; shadow style adds "shadow"


class Base(Strict):
    source: str
    output: str
    profile: str | None = None
    name: str | None = None


class Popup(Base):
    items: list[PopupItem] = Field(min_length=1)
    title_font: str | None = None
    body_font: str | None = None
    palette: dict[str, str] = Field(default_factory=dict)
    graphics_white_nits: float = Field(203, ge=50, le=1000)
    audio: dict | None = None


class Lower(Base):
    items: list[LabelItem] = Field(min_length=1)
    font: str | None = None
    palette: dict[str, str] = Field(default_factory=dict)
    graphics_white_nits: float = Field(203, ge=50, le=1000)


class Word(Strict):
    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float


class Cue(Strict):
    start: float = Field(ge=0)
    end: float
    words: list[Word] = Field(min_length=1)


class Captions(Base):
    reviewed: Literal[True]
    cues: list[Cue] = Field(min_length=1)
    font: str = "Segoe UI"
    size: float = Field(64, gt=4)
    margin_v: float = Field(120, ge=0)
    color: str = "#ffffff"
    highlight: str = "#ffd166"


class Comparison(Base):
    inset: str
    x: float
    y: float
    width: float = Field(gt=1)
    height: float | None = Field(None, gt=1)
    start: float = Field(0, ge=0)
    end: float
    label: str = ""
    font: str | None = None


class Transitions(Strict):
    clips: list[str] = Field(min_length=2)
    output: str
    type: str = Field("fade", pattern=r"^[a-z0-9]+$")
    duration: float = Field(1, gt=0, le=10)
    profile: str | None = None
    name: str | None = None


class Look(Base):
    look: Literal["editorial", "vintage", "monochrome"]
    approved: Literal[True]


class Narration(Base):
    narration: str
    narration_start: float = Field(0, ge=0)
    narration_volume: float = Field(1, gt=0, le=4)
    music: str | None = None
    music_volume: float = Field(0.35, gt=0, le=4)
    loudness: float | None = Field(None, ge=-40, le=-5)


# --- card rendering ---------------------------------------------------------------------------------

def _font(path, size):
    from PIL import ImageFont
    return ImageFont.truetype(path, max(8, round(size)))


def _fit(draw, text, path, size, max_w) -> object:
    """Largest font <= size (down to 60%) whose rendering fits max_w; otherwise refuse (no silent cut)."""
    for k in [1 - 0.05 * i for i in range(9)]:
        f = _font(path, size * k)
        if draw.textlength(text, font=f) <= max_w:
            return f
    raise ValueError(f"text {text!r} does not fit a {round(max_w)} px card line even at 60% size; "
                     "shorten it or widen the card")


def _cached(asset_dir, recipe, parts, render) -> str:
    """Render once per content: name = hash of recipe version + every input (file bytes by hash)."""
    key = _h(json.dumps([recipe, CATALOG[recipe]["version"], _CODE, parts], sort_keys=True).encode()).hexdigest()[:20]
    out = Path(asset_dir) / f"{recipe}-{key}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.png")
        render().save(tmp)
        tmp.replace(out)
    return str(out)


def popup_card(asset, title, subtitle, W, H, title_font, body_font, pal):
    from PIL import Image, ImageDraw, ImageFilter, ImageOps
    W, H = round(W), round(H)
    card = Image.new("RGBA", (W, H))
    m = Image.new("L", card.size)
    ImageDraw.Draw(m).rounded_rectangle((W * .023, H * .017, W * .977, H * .972), radius=W * .045, fill=180)
    shadow = Image.new("RGBA", card.size, (20, 15, 10, 0))
    shadow.putalpha(m.filter(ImageFilter.GaussianBlur(W * .014)))
    card.alpha_composite(shadow)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((W * .019, H * .010, W * .981, H * .959), radius=W * .040, fill=pal["card"],
                        outline=pal["edge"], width=max(1, round(W * .0032)))
    photo = ImageOps.exif_transpose(Image.open(asset)).convert("RGB")
    photo.thumbnail((round(W * .885), round(H * (.746 if subtitle else .78))), Image.Resampling.LANCZOS)
    card.paste(photo, ((W - photo.width) // 2, round(H * .045) + (round(H * (.746 if subtitle else .78)) - photo.height) // 2))
    line_y = H * (.818 if subtitle else .85)
    d.line((W * .379, line_y, W * .621, line_y), fill=pal["edge"], width=max(1, round(W * .002)))
    tf = _fit(d, title, title_font, H * .0586, W * .88)
    d.text((W / 2, H * (.86 if subtitle else .905)), title, anchor="mm", font=tf, fill=pal["title"])
    if subtitle:
        d.text((W / 2, H * .915), subtitle, anchor="mm", font=_fit(d, subtitle, body_font, H * .0293, W * .88),
               fill=pal["subtitle"])
    return card


def label_card(text, W, H, font, pal, style="pill"):
    from PIL import Image, ImageDraw, ImageFilter
    W, H = round(W), round(H)
    card = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(card)
    if style == "shadow":  # margins inside the card keep the blur from clipping at its edges
        f = _fit(d, text, font, H * .6, W * .9)
        sr, sg, sb = (int(pal.get("shadow", "#000000").lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        d.text((W / 2, H / 2 + H * .03), text, anchor="mm", font=f, fill=(sr, sg, sb, 180))
        card = card.filter(ImageFilter.GaussianBlur(H * .05))
        ImageDraw.Draw(card).text((W / 2, H / 2), text, anchor="mm", font=f, fill=pal["label_text"])
        return card
    r, g, b = (int(pal["label"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=H * .28, fill=(r, g, b, 225))
    d.text((W / 2, H / 2), text, anchor="mm", font=_fit(d, text, font, H * .46, W * .9), fill=pal["label_text"])
    return card


# --- expansion --------------------------------------------------------------------------------------

def _src(path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    return str(p.resolve())


def _geometry(boxes, path):
    v = media.info(path)["video"]
    if not v:
        raise ValueError(f"{path} has no video")
    for n, b in enumerate(boxes):
        h = b.height if b.height is not None else 2
        if b.x < 0 or b.y < 0 or b.x + b.width > v["width"] or b.y + h > v["height"]:
            raise ValueError(f"item {n}: box {b.x},{b.y} {b.width}x{h} is outside the "
                             f"{v['width']}x{v['height']} frame (source px)")
        if b.start >= (v["duration"] or math.inf):
            raise ValueError(f"item {n}: starts after the clip ends")
    return v


def _profile(req, v):
    return req.profile or ("native-hdr-master" if v["hdr"] else "4k60-master")


def _job(name, req, sources, steps, v, **kw):
    return jobs.Job(name=req.name or name, sources={k: {"path": p} for k, p in sources.items()}, steps=steps,
                    output=str(Path(req.output).resolve()), profile=_profile(req, v) if v else (req.profile or "4k60-master"),
                    **kw)


def _compose_job(name, req, cards, v, extra=None):
    sources = {"base": _src(req.source)}
    layers = []
    for n, (png, it) in enumerate(cards):
        sid = f"card{n}"
        sources[sid] = png
        layers.append({"input": n + 1, "start": it.start, "end": it.end, "x": it.x, "y": it.y,
                       "width": it.width, "height": it.height, "animation": it.animation}
                      | {k: getattr(it, k) for k in ("entrance", "slide", "initial_scale", "fade")
                         if getattr(it, k, None) is not None})
    params = {"layers": layers, "color_policy": "preserve",
              "graphics_white_nits": getattr(req, "graphics_white_nits", 203)} | (extra or {})
    step = {"id": "compose", "op": "compose", "inputs": list(sources), "params": params}
    return _job(name, req, sources, [step], v, validate={"against": "base"})


def _popup(req: Popup, asset_dir):
    v = _geometry(req.items, req.source)
    pal = PALETTE | req.palette
    tfont, bfont = req.title_font or default_font("title"), req.body_font or default_font("body")
    cards = []
    for it in req.items:
        a = _src(it.asset)
        parts = [media.sha256(a), it.title, it.subtitle, it.width, it.height, media.sha256(tfont),
                 media.sha256(bfont), pal]
        cards.append((_cached(asset_dir, "image-popup", parts, lambda a=a, it=it: popup_card(
            a, it.title, it.subtitle, it.width, it.height, tfont, bfont, pal)), it))
    return _compose_job("image-popup", req, cards, v, {"audio": req.audio} if req.audio is not None else None)


def _lower(req: Lower, asset_dir):
    v = _geometry(req.items, req.source)
    cards = []
    for it in req.items:
        pal, font = PALETTE | req.palette | it.palette, it.font or req.font or default_font("bold")
        parts = [it.text, it.width, it.height, media.sha256(font), pal] + ([it.style] if it.style != "pill" else [])
        cards.append((_cached(asset_dir, "lower-third", parts, lambda it=it, pal=pal, font=font: label_card(
            it.text, it.width, it.height, font, pal, it.style)), it))
    return _compose_job("lower-third", req, cards, v)


def _ass_color(c):
    r, g, b = (int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"&H00{b:02X}{g:02X}{r:02X}"


def ass_text(s) -> str:
    """Literal text for an ASS event: braces escaped, backslashes can't start \\N/\\h/override tags."""
    return s.replace("\\", "\\\u2060").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def _t(s):
    cs = round(s * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def _captions(req: Captions, asset_dir):
    v = media.info(_src(req.source))["video"]
    if v["hdr"]:
        raise ValueError("kinetic captions are burned by libass in SDR colour; for HDR sources use lower-third "
                         "cards (compose converts them into HLG/PQ)")
    prev = 0.0
    for n, c in enumerate(req.cues):
        if c.end <= c.start or c.start < prev:
            raise ValueError(f"cue {n}: needs start < end and no overlap with the previous cue")
        prev = c.end
        w0 = c.start
        for w in c.words:
            if w.start < w0 or w.end < w.start or w.end > c.end:
                raise ValueError(f"cue {n}: word {w.text!r} timing is outside the cue or out of order")
            w0 = w.end
    lines = []
    for c in req.cues:
        t, txt = c.start, ""
        for w in c.words:
            if w.start > t:
                txt += f"{{\\k{round((w.start - t) * 100)}}}"
            txt += f"{{\\k{max(1, round((w.end - w.start) * 100))}}}{ass_text(w.text)} "
            t = w.end
        lines.append(f"Dialogue: 0,{_t(c.start)},{_t(c.end)},K,,0,0,0,,{txt.rstrip()}")
    ass = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {v['width']}\nPlayResY: {v['height']}\n"
           "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
           "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
           "MarginL, MarginR, MarginV, Encoding\n"
           f"Style: K,{req.font},{round(req.size)},{_ass_color(req.highlight)},{_ass_color(req.color)},&H00000000,"
           f"&H80000000,-1,0,0,0,100,100,0,0,1,{max(1, round(req.size / 16))},0,2,40,40,{round(req.margin_v)},1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           + "\n".join(lines) + "\n")
    key = _h(ass.encode()).hexdigest()[:20]
    f = Path(asset_dir) / f"kinetic-captions-{key}.ass"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(ass, encoding="utf-8")
    step = {"id": "captions", "op": "subtitles", "inputs": ["base"], "params": {"file": str(f)}}
    return _job("kinetic-captions", req, {"base": _src(req.source)}, [step], v, validate={"against": "base"})


def _comparison(req: Comparison, asset_dir):
    v = _geometry([req], req.source)
    inset = _src(req.inset)
    if media.info(inset)["is_image"]:
        if req.height is None:
            iv = media.info(inset)["video"]
            req.height = round(req.width * iv["height"] / iv["width"])
        cards = [(inset, Box(start=req.start, end=req.end, x=req.x, y=req.y, width=req.width, height=req.height,
                             animation="fade"))]
        if req.label:
            lh = max(24, req.width * .12)
            font = req.font or default_font("bold")
            png = _cached(asset_dir, "comparison", [req.label, req.width, lh, media.sha256(font)],
                          lambda: label_card(req.label, req.width, lh, font, PALETTE))
            cards.append((png, Box(start=req.start, end=req.end, x=req.x, y=min(req.y + req.height + lh * .2,
                                   v["height"] - lh), width=req.width, height=lh, animation="fade")))
        return _compose_job("comparison", req, cards, v)
    if v["hdr"]:
        raise ValueError("comparison: a clip inset over HDR needs a colour-managed composite; "
                         "use an image inset, or conform the clip first")
    if req.label:
        raise ValueError("comparison: labels are supported for image insets")
    step = {"id": "pip", "op": "overlay", "inputs": ["base", "inset"],
            "params": {"x": req.x, "y": req.y, "width": req.width, "start": req.start, "end": req.end}}
    return _job("comparison", req, {"base": _src(req.source), "inset": inset}, [step], v, validate={"against": "base"})


def _transitions(req: Transitions, asset_dir):
    clips = [_src(c) for c in req.clips]
    v = media.info(clips[0])["video"]
    for c in clips:
        d = media.info(c)["video"]["duration"] or 0
        if d <= req.duration:
            raise ValueError(f"clip {c} is not longer than the {req.duration} s transition")
    steps, cur = [], "c0"
    for i in range(1, len(clips)):
        steps.append({"id": f"t{i}", "op": "transition", "inputs": [cur, f"c{i}"],
                      "params": {"type": req.type, "duration": req.duration}})
        cur = f"t{i}"
    return _job("clip-transitions", req, {f"c{i}": c for i, c in enumerate(clips)}, steps, v)


def _look(req: Look, asset_dir):
    v = media.info(_src(req.source))["video"]
    if v["hdr"]:
        raise ValueError("look recipes are SDR grades; for HDR use an approved look_match/LUT step")
    style = {"editorial": "cinematic", "vintage": "vintage", "monochrome": "bw"}[req.look]
    step = {"id": "look", "op": "stylize", "inputs": ["base"], "params": {"style": style}}
    return _job("look", req, {"base": _src(req.source)}, [step], v, validate={"against": "base"})


def _narration(req: Narration, asset_dir):
    src = _src(req.source)
    v = media.info(src)["video"]
    sources = {"base": src, "narration": _src(req.narration)}
    steps = [{"id": "voice", "op": "audio", "inputs": ["base", "narration"],
              "params": {"mode": "mix", "delay": req.narration_start, "music_volume": req.narration_volume}}]
    if req.music:
        sources["music"] = _src(req.music)
        steps.append({"id": "bed", "op": "audio", "inputs": ["voice", "music"],
                      "params": {"mode": "duck", "music_volume": req.music_volume}})
    if req.loudness is not None:
        steps.append({"id": "level", "op": "loudnorm", "inputs": [steps[-1]["id"]], "params": {"I": req.loudness}})
    return _job("narration-mix", req, sources, steps, v, validate={"against": "base"})


RECIPES = {"image-popup": (Popup, _popup), "lower-third": (Lower, _lower), "kinetic-captions": (Captions, _captions),
           "comparison": (Comparison, _comparison), "clip-transitions": (Transitions, _transitions),
           "look": (Look, _look), "narration-mix": (Narration, _narration)}


def expand(name, request: dict, asset_dir) -> jobs.Job:
    """Validate a request and expand it into a job manifest (renders only static card assets)."""
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name}; have {sorted(RECIPES)}")
    model, fn = RECIPES[name]
    job = fn(model.model_validate(request), Path(asset_dir))
    if job.profile not in jobs.PROFILES:
        raise ValueError(f"unknown profile {job.profile}")
    return job
