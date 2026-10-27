"""End-to-end checks against real ffmpeg on synthetic clips (MVP acceptance tests from the spec)."""
import json

import numpy as np
import pytest

from studio import jobs, media
from studio.jobs import Job
from studio.qc import validate


def lavfi(out, video, audio=None, dur=2):
    args = ["-f", "lavfi", "-i", video]
    if audio:
        args += ["-f", "lavfi", "-i", f"{audio}:d={dur}"]
    media.ffmpeg(args + ["-t", dur, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", out])
    return str(out)


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    t = tmp_path_factory.mktemp("media")
    ws = t / "ws"
    clip = lavfi(t / "clip.mp4", "testsrc2=s=640x360:r=30", "sine=f=440:r=48000")
    clip2 = lavfi(t / "clip2.mp4", "smptebars=s=320x240:r=25", None)  # different size/fps, no audio
    media.ffmpeg(["-f", "lavfi", "-i", "sine=f=220:d=5:r=48000", "-c:a", "aac", t / "music.m4a"])
    media.ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=200x100", "-frames:v", "1", t / "logo.png"])
    (t / "subs.srt").write_text("1\n00:00:00,000 --> 00:00:01,500\nHello studio\n", encoding="utf-8")
    # bright background + dark subject: a mask that just copied luma would fail the pixel asserts
    square = lavfi(t / "square.mp4", "color=c=yellow:s=640x360:r=30[b];color=c=blue:s=100x100:r=30[s];"
                   "[b][s]overlay=x='60+t*80':y=130:shortest=1", None)
    green = lavfi(t / "green.mp4", "color=c=0x00FF00:s=640x360:r=30[b];color=c=white:s=100x100:r=30[s];"
                  "[b][s]overlay=x=270:y=130:shortest=1", None)
    return {"t": t, "ws": ws, "clip": clip, "clip2": clip2, "music": str(t / "music.m4a"),
            "logo": str(t / "logo.png"), "srt": str(t / "subs.srt"), "square": square, "green": green}


def run(d, srcs, steps, name, **kw):
    job = Job(name=name, sources={f"s{i}": {"path": p} for i, p in enumerate(srcs)}, steps=steps,
              output=str(d["t"] / f"{name}.mp4"), profile=kw.pop("profile", "h264-master"), **kw)
    return jobs.run(job, ws=d["ws"], log=lambda m: None)


def test_inspect(d):
    i = media.info(d["clip"])
    assert (i["video"]["width"], i["video"]["height"], i["video"]["fps"]) == (640, 360, 30)
    assert abs(i["video"]["duration"] - 2) < 0.05 and i["audio"]["sample_rate"] == 48000


CASES = [
    ("trim", [], {"start": 0.5, "end": 1.5}, 1.0),
    ("crop", [], {"x": 10, "y": 10, "w": 320, "h": 180}, 2),
    ("resize", [], {"width": 1280, "height": 720, "mode": "fit"}, 2),
    ("rotate", [], {"degrees": 90}, 2),
    ("speed", [], {"factor": 2}, 1.0),
    ("reverse", [], {}, 2),
    ("fps", [], {"fps": 60}, 2),
    ("freeze", [], {"at": 1, "duration": 0.5}, 2.5),
    ("concat", ["clip2"], {}, 4),
    ("transition", ["clip2"], {"type": "dissolve", "duration": 0.5}, 3.5),
    ("fade", [], {"in": 0.5, "out": 0.5}, 2),
    ("overlay", ["logo"], {"x": "W-w-20", "y": 20, "opacity": 0.8, "start": 0.5, "end": 1.5}, 2),
    ("text", [], {"text": "Lower third", "box": True, "size": 32}, 2),
    ("subtitles", [], None, 2),
    ("audio", ["music"], {"mode": "mix", "music_volume": 0.3}, 2),
    ("audio", ["music"], {"mode": "duck"}, 2),
    ("audio", ["music"], {"mode": "replace"}, 2),
    ("loudnorm", [], {}, 2),
    ("audio_clean", [], {}, 2),
    ("color", [], {"brightness": 0.05, "contrast": 1.1, "saturation": 1.2, "temperature": 5600}, 2),
]


@pytest.mark.parametrize("op,extra,params,dur", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_ops(d, op, extra, params, dur):
    params = {"file": d["srt"]} if params is None else params
    r = run(d, [d["clip"]] + [d[e] for e in extra],
            [{"id": "x", "op": op, "inputs": [f"s{i}" for i in range(1 + len(extra))], "params": params}],
            f"op-{op}-{abs(hash(json.dumps(params))) % 10000}", validate={"duration": dur})
    assert r["ok"], json.dumps(r["validation"], indent=1)


def test_concat_fills_short_video_holes(d):
    """iPhone trims can end a frame before their audio; concat must not leave a timestamp hole at each cut."""
    short = d["t"] / "short.mp4"  # 29 video frames (0.967 s) + 1.0 s audio
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=1", "-f", "lavfi", "-i", "sine=f=440:r=48000:d=1",
                  "-filter_complex", "[0:v]trim=end_frame=29[v]", "-map", "[v]", "-map", "1:a",
                  "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", short])
    r = run(d, [str(short)] * 3, [{"id": "x", "op": "concat", "inputs": ["s0", "s1", "s2"], "params": {}}],
            "concat-holes", validate={"duration": 3})
    pk = media.ffprobe_json(["-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                             r["output"]])
    assert int(pk["streams"][0]["nb_read_packets"]) >= 89  # 87 if the two inner holes stay


def test_4k60_trim_overlay_audio(d):
    """Acceptance: 4K60 source, trim + overlay + audio change, export 4K60 with right duration and sync."""
    src = lavfi(d["t"] / "src4k.mp4", "testsrc2=s=3840x2160:r=60", "sine=f=440:r=48000", 2)
    steps = [{"id": "cut", "op": "trim", "inputs": ["src"], "params": {"start": 0.5, "duration": 1}},
             {"id": "ov", "op": "overlay", "inputs": ["cut", "logo"], "params": {"x": 100, "y": 100, "width": 600}},
             {"id": "mix", "op": "audio", "inputs": ["ov", "music"], "params": {"mode": "mix"}}]
    job = Job(name="acc4k", sources={"src": {"path": src}, "logo": {"path": d["logo"]}, "music": {"path": d["music"]}},
              steps=steps, output=str(d["t"] / "acc4k.mp4"), profile="4k60-master", validate={"duration": 1.0})
    r = jobs.run(job, ws=d["ws"], log=lambda m: None)
    assert r["ok"], json.dumps(r["validation"], indent=1)
    s = r["validation"]["summary"]
    assert s["size"] == [3840, 2160] and s["fps"] == 60 and s["vcodec"] == "hevc"
    pv = jobs.run(job, mode="preview", ws=d["ws"], log=lambda m: None)
    assert pv["ok"] and pv["validation"]["summary"]["size"][1] == 540


def test_object_mask_and_background(d):
    """Acceptance (model-free backend): select by box, track, replace background at source res."""
    steps = [{"id": "m", "op": "mask", "inputs": ["s0"],
              "params": {"target": "object", "backend": "grabcut", "box": [55, 125, 110, 110], "height": 360}},
             {"id": "bg", "op": "background", "inputs": ["s0", "m"], "params": {"mode": "replace", "color": "black"}}]
    r = run(d, [d["square"]], steps, "bgreplace", validate={"against": "s0", "duration": 2})
    assert r["ok"], json.dumps(r["validation"], indent=1)
    stats = r["steps"][0]["meta"]["mask"]
    assert 0.03 < stats["coverage_mean"] < 0.07 and not stats["low_confidence"], stats
    last = list(media.read_frames(r["output"], 640, 360))[-1].astype(int)
    # square moved to x~60+80*2=220 by the end: inside stays blue, old position/background black
    assert last[180, 260, 2] > 150 and last[180, 260, 0] < 80, last[180, 260]
    assert last[180, 30].max() < 30 and last[20, 600].max() < 30, (last[180, 30], last[20, 600])
    rm = run(d, [d["square"]], [steps[0], {"id": "rm", "op": "background", "inputs": ["s0", "m"],
                                           "params": {"mode": "remove"}}], "bgremove", profile="alpha-master")
    assert rm["ok"] and rm["steps"][0]["cached"] and rm["validation"]["summary"]["pix_fmt"].startswith("yuva")


def test_chromakey(d):
    r = run(d, [d["green"], d["logo"]], [{"id": "k", "op": "chromakey", "inputs": ["s0", "s1"], "params": {}}], "key")
    assert r["ok"], json.dumps(r["validation"], indent=1)
    f = list(media.read_frames(r["output"], 640, 360))[10].astype(int)
    assert f[20, 20, 0] > 180 and f[20, 20, 1] < 80  # green replaced by the red background


def test_resume_after_failure(d):
    """An interrupted/failed job resumes without repeating completed analysis."""
    good = {"id": "cut", "op": "trim", "inputs": ["s0"], "params": {"start": 0, "end": 1.2}}
    with pytest.raises(RuntimeError):
        run(d, [d["clip"]], [good, {"id": "bad", "op": "crop", "inputs": ["cut"], "params": {"w": 99999, "h": 10}}],
            "resume", retries=0)
    r = run(d, [d["clip"]], [good, {"id": "ok", "op": "crop", "inputs": ["cut"], "params": {"w": 320, "h": 180}}],
            "resume", retries=0)
    assert r["ok"] and r["steps"][0]["cached"] and not r["steps"][1]["cached"]


def test_never_overwrites_source(d):
    job = Job(sources={"s": {"path": d["clip"]}}, steps=[{"id": "a", "op": "fade", "inputs": ["s"]}],
              output=d["clip"])
    with pytest.raises(ValueError, match="overwrite"):
        jobs.run(job, ws=d["ws"], log=lambda m: None)


def test_validate_detects_mismatch(d):
    r = validate(d["clip2"], against=d["clip"])
    assert not r["ok"] and {c["check"] for c in r["checks"] if not c["ok"]} >= {"resolution", "fps", "audio_present"}
    assert np.isclose(r["summary"]["fps"], 25)
