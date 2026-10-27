"""FFmpeg/FFprobe wrappers: the media I/O backbone."""
import hashlib
import json
import os
import shutil
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@lru_cache(None)
def tool(name: str) -> str:
    p = os.environ.get(f"STUDIO_{name.upper()}") or shutil.which(name)
    if not p:  # winget adds a PATH entry that open shells don't see yet
        link = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links" / f"{name}.exe"
        p = str(link) if link.exists() else None
    if not p:
        raise RuntimeError(f"{name} not found: install FFmpeg or set STUDIO_{name.upper()}")
    return p


def ffmpeg(args, cwd=None) -> list[str]:
    cmd = [tool("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr[-3000:]}\ncmd: {subprocess.list2cmdline(cmd)}")
    return cmd


def ffprobe_json(args) -> dict:
    r = subprocess.run([tool("ffprobe"), "-v", "error", "-of", "json", *map(str, args)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()}")
    return json.loads(r.stdout or "{}")


def _stat_key(path) -> tuple:
    p = Path(path).resolve()
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


def info(path) -> dict:
    return _info(*_stat_key(path))


def _dur(s: dict, fallback):
    if s.get("duration") not in (None, "N/A"):
        return float(s["duration"])
    tag = (s.get("tags") or {}).get("DURATION")  # matroska
    if tag:
        h, m, sec = tag.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return fallback


def _rate(s: dict):
    for k in ("avg_frame_rate", "r_frame_rate"):
        v = s.get(k, "0/0")
        if v not in ("0/0", "0/1"):
            return v
    return None


@lru_cache(256)
def _info(path: str, _mtime: int, _size: int) -> dict:
    raw = ffprobe_json(["-show_streams", "-show_format", path])
    fmt = raw.get("format", {})
    fdur = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
    streams = raw.get("streams", [])
    v = next((s for s in streams if s["codec_type"] == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s["codec_type"] == "audio"), None)
    out = {"path": path, "container": fmt.get("format_name"), "duration": fdur,
           "size_bytes": int(fmt.get("size", 0) or 0), "bit_rate": int(fmt.get("bit_rate", 0) or 0),
           "start_time": float(fmt.get("start_time", 0) or 0), "video": None, "audio": None,
           "is_image": Path(path).suffix.lower() in IMAGE_EXTS,
           "streams": [{"index": s["index"], "type": s["codec_type"], "codec": s.get("codec_name")} for s in streams]}
    if v:
        w, h = v["width"], v["height"]
        rot = 0
        for sd in v.get("side_data_list", []):
            rot = int(sd.get("rotation", rot) or rot)
        rot = int((v.get("tags") or {}).get("rotate", rot))
        if abs(rot) % 180 == 90:  # ffmpeg autorotates on decode, so report display dims
            w, h = h, w
        rate = _rate(v)
        trc = v.get("color_transfer")
        out["video"] = {
            "codec": v.get("codec_name"), "profile": v.get("profile"), "width": w, "height": h,
            "rotation": rot, "fps": float(Fraction(rate)) if rate else None, "fps_str": rate,
            "r_frame_rate": v.get("r_frame_rate"), "time_base": v.get("time_base"),
            "coded": [v["width"], v["height"]],
            "vfr": v.get("avg_frame_rate") != v.get("r_frame_rate"),
            "pix_fmt": v.get("pix_fmt"), "bit_depth": int(v.get("bits_per_raw_sample") or 0) or None,
            "color_space": v.get("color_space"), "color_primaries": v.get("color_primaries"),
            "color_transfer": trc, "color_range": v.get("color_range"),
            "hdr": trc in ("smpte2084", "arib-std-b67"),
            "duration": _dur(v, fdur), "frames": int(v["nb_frames"]) if v.get("nb_frames") else None,
            "start_time": float(v.get("start_time", 0) or 0)}
    if a:
        out["audio"] = {"codec": a.get("codec_name"), "sample_rate": int(a.get("sample_rate", 0)),
                        "channels": a.get("channels"), "layout": a.get("channel_layout"),
                        "duration": _dur(a, fdur), "start_time": float(a.get("start_time", 0) or 0)}
    return out


def keyframes(path) -> list[float]:
    r = ffprobe_json(["-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries",
                      "frame=pts_time", path])
    return [float(f["pts_time"]) for f in r.get("frames", []) if f.get("pts_time") not in (None, "N/A")]


def hdr10(path) -> dict | None:
    """HDR10 static metadata from the first frame: {"display": [gx, gy, bx, by, rx, ry, wpx, wpy,
    max_nits, min_nits], "cll": [max_cll, max_fall]} (either may be missing), or None."""
    fr = ffprobe_json(["-select_streams", "v:0", "-read_intervals", "%+#1", "-show_frames", path]).get("frames") or [{}]
    out = {}
    for sd in fr[0].get("side_data_list", []):
        if sd.get("side_data_type") == "Mastering display metadata" and "red_x" in sd:
            out["display"] = [float(Fraction(sd[k])) for k in (
                "green_x", "green_y", "blue_x", "blue_y", "red_x", "red_y", "white_point_x", "white_point_y",
                "max_luminance", "min_luminance")]
        if sd.get("side_data_type") == "Content light level metadata":
            out["cll"] = [int(sd.get("max_content", 0)), int(sd.get("max_average", 0))]
    return out or None


def sha256(path) -> str:
    key = _stat_key(path)
    # ponytail: small files (fonts, LUTs, cues, cards) are always re-hashed, since a same-size rewrite can
    # keep its mtime; big media trust (path, mtime, size). Hash on every call if that ever bites.
    return _sha.__wrapped__(*key) if key[2] < 64 << 20 else _sha(*key)


@lru_cache(512)
def _sha(path: str, _mtime: int, _size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(1)
def _decoders() -> frozenset:
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-decoders"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return frozenset(x.split()[1] for x in r.stdout.splitlines() if len(x.split()) > 1 and x[:1] == " " and x[1:2] == "A")


def audio_maps(path, input_index=0) -> list:
    """-map args for every audio track FFmpeg can decode (iPhone spatial-audio APAC tracks have no decoder)."""
    out = []
    for s in info(path)["streams"]:
        if s["type"] == "audio" and s["codec"] in _decoders():
            out += ["-map", f"{input_index}:{s['index']}"]
    return out


@lru_cache(1)
def ffmpeg_version() -> str:
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-version"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout.splitlines()[0] if r.stdout else "unknown"


def publish(tmp, destination) -> None:
    """Move a finished temp file to `destination` without ever replacing an existing file.
    A same-directory hard link is atomic and fails if the destination appeared meanwhile."""
    tmp, destination = Path(tmp), Path(destination)
    if tmp.resolve() == destination.resolve():
        raise ValueError("temporary file equals destination")
    try:
        os.link(tmp, destination)
    except FileExistsError:
        raise FileExistsError(f"output already exists: {destination}") from None
    except OSError:  # no hard links here: exclusive create + copy still never replaces a file
        try:
            fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
        except FileExistsError:
            raise FileExistsError(f"output already exists: {destination}") from None
        try:
            with os.fdopen(fd, "wb") as dst, open(tmp, "rb") as src:
                shutil.copyfileobj(src, dst, 1 << 24)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    tmp.unlink()


def frame_clock(path) -> dict:
    """Video stream time base and every frame's integer pts (presentation order)."""
    r = ffprobe_json(["-select_streams", "v:0", "-show_entries", "stream=time_base:packet=pts", path])
    pts = sorted(int(p["pts"]) for p in r.get("packets", []) if p.get("pts") not in (None, "N/A"))
    return {"time_base": r["streams"][0]["time_base"], "pts": pts}


def fpath(p) -> str:
    """Escape a file path for use inside an ffmpeg filter argument (Windows drive colons)."""
    return "'" + str(Path(p).resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "") + "'"


def read_frames(path, width: int, height: int):
    """Yield RGB uint8 frames scaled to width x height (constant-rate decode of the video stream)."""
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", f"scale={width}:{height}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    import numpy as np
    n = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        while (buf := p.stdout.read(n)) and len(buf) == n:
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)


class FrameWriter:
    """Pipe uint8 frames (gray masks, or rgb24 pictures) into a lossless FFV1 video."""

    def __init__(self, out, width, height, fps_str, pix_fmt="gray"):
        self.p = subprocess.Popen(
            [tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}",
             "-r", fps_str, "-i", "-", "-c:v", "ffv1", "-level", "3", "-slices", "16", "-threads", "8", str(out)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode(errors="replace")
        if self.p.wait():
            raise RuntimeError(f"frame encode failed: {err}")
