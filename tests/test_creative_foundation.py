"""Slice 1: cache/publication safety, compose, recipes, audio timing, review evidence."""
import json
import subprocess
from pathlib import Path

import pytest

from studio import jobs, media
from test_phases import talk  # noqa: F401
from test_studio import d, lavfi, run  # noqa: F401  (shared module fixture)


def test_resource_content_changes_cache_key(tmp_path):
    font = tmp_path / "typeface.ttf"
    font.write_bytes(b"first")
    before = jobs.resource_hashes({"layers": [{"font": str(font)}]})
    font.write_bytes(b"other")
    assert before != jobs.resource_hashes({"layers": [{"font": str(font)}]})
    with pytest.raises(FileNotFoundError):
        jobs.resource_hashes({"lut": str(tmp_path / "missing.cube")})


def test_publish_preserves_existing_destination(tmp_path):
    tmp, dst = tmp_path / "out.part.wav", tmp_path / "out.wav"
    tmp.write_bytes(b"new")
    dst.write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        media.publish(tmp, dst)
    assert dst.read_bytes() == b"user file" and tmp.exists()
    media.publish(tmp, tmp_path / "fresh.wav")
    assert (tmp_path / "fresh.wav").read_bytes() == b"new" and not tmp.exists()


def test_same_path_subtitle_change_and_damaged_cache_rerender(d, tmp_path):
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nfirst\n", encoding="utf-8")
    step = [{"id": "sub", "op": "subtitles", "inputs": ["s0"], "params": {"file": str(srt)}}]
    first = run(d, [d["clip"]], step, "subcache-a")
    assert not first["steps"][0]["cached"]
    assert run(d, [d["clip"]], step, "subcache-b")["steps"][0]["cached"]
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nsecond\n", encoding="utf-8")
    changed = run(d, [d["clip"]], step, "subcache-c")
    assert not changed["steps"][0]["cached"]
    cached = Path(changed["steps"][0]["output"])
    cached.write_bytes(cached.read_bytes()[:1000])  # truncated intermediate
    assert not run(d, [d["clip"]], step, "subcache-d")["steps"][0]["cached"]


def test_delivery_never_replaced(d):
    step = [{"id": "f", "op": "fade", "inputs": ["s0"], "params": {"in": 0.5}}]
    first = run(d, [d["clip"]], step, "deliver-once")
    digest = media.sha256(first["output"])
    again = run(d, [d["clip"]], step, "deliver-once")  # identical request: verified reuse
    assert again["reused"] and again["sha256"] == digest
    with pytest.raises(FileExistsError):
        run(d, [d["clip"]], [{"id": "f", "op": "fade", "inputs": ["s0"], "params": {"in": 1}}], "deliver-once")
    assert media.sha256(first["output"]) == digest


# --- compose -------------------------------------------------------------------------------------

from studio.compose import validate_compose  # noqa: E402
from studio.qc import clock_match  # noqa: E402


def layer(**kw):
    return {"input": 1, "start": 0.5, "end": 1.0, "x": 20, "y": 20, "width": 80, "height": 40,
            "animation": "none"} | kw


@pytest.mark.parametrize("bad, match", [
    ({"start": float("nan")}, "finite"), ({"end": 0.5}, "start < end"), ({"input": 2}, "index"),
    ({"width": -5}, "outside"), ({"bogus": 1}, "unknown"), ({"animation": "spin"}, "animation"),
    ({"start": True}, "finite")])
def test_compose_rejects_bad_layers(d, bad, match):
    with pytest.raises(ValueError, match=match):
        validate_compose([d["clip"], d["logo"]], {"layers": [layer(**bad)]})


def test_compose_is_half_open_and_preserves_base(d):
    r = run(d, [d["clip"], d["logo"]], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"],
                                         "params": {"layers": [layer()]}}], "compose-bounds", validate={"against": "s0"})
    assert r["ok"], json.dumps(r["validation"], indent=1)
    out = list(media.read_frames(r["steps"][0]["output"], 640, 360))
    base = list(media.read_frames(d["clip"], 640, 360))
    assert len(out) == len(base)
    assert out[15][30, 30, 0] > 200 and out[29][30, 30, 0] > 200 and out[15][30, 30, 1] < 60
    for i in (0, 14, 30, 59):
        assert (out[i] == base[i]).all(), i  # whole frame untouched outside [0.5, 1.0)
    assert (out[20][200:, 200:] == base[20][200:, 200:]).all()  # untouched region while the layer shows
    assert media.info(r["steps"][0]["output"])["audio"]


@pytest.fixture(scope="module")
def hlg(d):
    """10-bit HLG BT.2020 VFR clip at a 30000/1001-ish rate, no audio."""
    out = d["t"] / "hlg.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30000/1001", "-t", 1.5,
                  "-vf", "setpts='PTS+if(gte(N,20),0.004/TB,0)',setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc", "-fps_mode", "passthrough",
                  "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
                  "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", out])
    return str(out)


def _y10(path, w=640, h=360):
    import numpy as np
    import subprocess
    raw = subprocess.run([media.tool("ffmpeg"), "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "yuv420p10le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint16).reshape(-1, w * h * 3 // 2)[:, :w * h].reshape(-1, h, w)


def test_compose_hdr_keeps_yuv10_clock_and_tags(d, hlg):
    p = {"layers": [layer(animation="slide-scale", start=0.2, end=1.2, x=40, y=40, width=200, height=100)],
         "graphics_white_nits": 203}
    r = run(d, [hlg, d["logo"]], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"], "params": p}],
            "compose-hlg", validate={"against": "s0"}, profile="4k60-master")
    assert r["ok"], json.dumps(r["validation"], indent=1)
    mid = r["steps"][0]["output"]
    v = media.info(mid)["video"]
    assert (v["pix_fmt"], v["color_transfer"], v["color_primaries"]) == ("yuv420p10le", "arib-std-b67", "bt2020")
    a, b = _y10(hlg), _y10(mid)
    assert a.shape == b.shape
    assert (a[:, 200:, 300:] == b[:, 200:, 300:]).all()  # untouched YUV10 region bit-identical, every frame
    assert (a[:, 60:120, 80:200] != b[:, 60:120, 80:200]).any()  # card present
    for f in (mid, r["output"]):
        c = clock_match(hlg, f)
        assert c["ok"], c
    fv = media.info(r["output"])["video"]
    assert fv["color_transfer"] == "arib-std-b67" and fv["pix_fmt"] == "yuv420p10le"


def test_compose_audio_cleanup_rejects_silent_base(d, hlg):
    with pytest.raises(RuntimeError, match="no audio"):
        run(d, [hlg, d["logo"]], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"],
                                   "params": {"layers": [layer()], "audio": {}}}], "compose-noaudio", retries=0)


# --- recipes -------------------------------------------------------------------------------------

from typer.testing import CliRunner  # noqa: E402

from studio.recipes import ass_text, expand, listing  # noqa: E402


def _render(d, job):
    return jobs.run(job, ws=d["ws"], log=lambda m: None)


def test_popup_expands_deterministically_and_renders(d, tmp_path):
    req = {"source": d["clip"], "output": str(tmp_path / "pop.mp4"), "profile": "h264-master",
           "items": [{"asset": d["logo"], "title": "Soft {shape}, C:\\nails", "subtitle": "ünïcode 'quotes'",
                      "start": .5, "end": 1.5, "x": 20, "y": 20, "width": 200, "height": 250}]}
    job = expand("image-popup", req, tmp_path / "assets")
    again = expand("image-popup", req, tmp_path / "assets")
    assert job.model_dump() == again.model_dump() and [s.op for s in job.steps] == ["compose"]
    assert job.sources["base"].path == str(Path(d["clip"]).resolve())
    assert job.steps[0].params["layers"][0]["start"] == .5
    assert "image-popup" in listing() and len(listing()) == 7
    r = _render(d, job)
    assert r["ok"], json.dumps(r["validation"], indent=1)
    f = list(media.read_frames(r["steps"][0]["output"], 640, 360))
    base = list(media.read_frames(d["clip"], 640, 360))[30]
    assert f[30][150, 120, 0] > 200 and f[30][150, 120, 1] < 60 and (f[30][300:, 300:] == base[300:, 300:]).all()


@pytest.mark.parametrize("bad, match", [
    ({"x": 600}, "outside"), ({"end": 0.2}, "after start"), ({"start": float("inf")}, "finite"),
    ({"extra": 1}, "Extra inputs"), ({"title": "x" * 400}, "does not fit")])
def test_popup_rejects_bad_requests(d, tmp_path, bad, match):
    item = {"asset": d["logo"], "title": "ok", "start": .5, "end": 1.5, "x": 20, "y": 20, "width": 200,
            "height": 250} | bad
    with pytest.raises(Exception, match=match):
        expand("image-popup", {"source": d["clip"], "output": str(tmp_path / "x.mp4"), "items": [item]},
               tmp_path / "assets")


def test_kinetic_captions_literal_and_reviewed(d, tmp_path):
    cue = {"start": 0.2, "end": 1.6, "words": [{"text": "{\\b1}literal", "start": 0.2, "end": 0.7},
                                                {"text": "C:\\new", "start": 0.8, "end": 1.4}]}
    req = {"source": d["clip"], "output": str(tmp_path / "k.mp4"), "profile": "h264-master", "reviewed": True,
           "cues": [cue], "size": 40, "margin_v": 30}
    job = expand("kinetic-captions", req, tmp_path / "assets")
    ass = Path(job.steps[0].params["file"]).read_text(encoding="utf-8")
    assert "\\{\\\u2060b1\\}literal" in ass and "{\\k10}" in ass  # gap before the 2nd word
    assert ass_text("a{b}\\N") == "a\\{b\\}\\\u2060N"
    assert _render(d, job)["ok"]
    with pytest.raises(Exception, match="reviewed"):
        expand("kinetic-captions", req | {"reviewed": False}, tmp_path / "assets")
    with pytest.raises(ValueError, match="outside the cue"):
        expand("kinetic-captions", req | {"cues": [cue | {"end": 1.0}]}, tmp_path / "assets")


def test_other_recipes_expand_and_render(d, tmp_path):
    out = lambda n: str(tmp_path / f"{n}.mp4")  # noqa: E731
    cases = [
        ("lower-third", {"source": d["clip"], "output": out("lt"), "profile": "h264-master",
                         "items": [{"text": "Dr. Ünal — host", "start": 0, "end": 1.5, "x": 20, "y": 280,
                                    "width": 300, "height": 60, "animation": "fade"}]}, ["compose"]),
        ("comparison", {"source": d["clip"], "output": out("cmp"), "profile": "h264-master", "inset": d["logo"],
                        "x": 400, "y": 20, "width": 200, "start": 0, "end": 2, "label": "Before"}, ["compose"]),
        ("comparison", {"source": d["clip"], "output": out("pip"), "profile": "h264-master", "inset": d["clip2"],
                        "x": 400, "y": 20, "width": 200, "start": 0, "end": 2}, ["overlay"]),
        ("clip-transitions", {"clips": [d["clip"], d["clip2"]], "output": out("tr"), "type": "dissolve",
                              "duration": 0.5, "profile": "h264-master"}, ["transition"]),
        ("look", {"source": d["clip"], "output": out("look"), "profile": "h264-master", "look": "monochrome",
                  "approved": True}, ["stylize"]),
        ("narration-mix", {"source": d["clip"], "output": out("nar"), "profile": "h264-master",
                           "narration": d["music"], "narration_start": 0.5, "music": d["music"], "loudness": -16},
         ["audio", "audio", "loudnorm"]),
    ]
    for name, req, ops in cases:
        job = expand(name, req, tmp_path / "assets")
        assert [s.op for s in job.steps] == ops, name
        r = _render(d, job)
        assert r["ok"], (name, json.dumps(r["validation"], indent=1))
    with pytest.raises(Exception, match="approved"):
        expand("look", cases[4][1] | {"approved": False}, tmp_path / "assets")


def test_recipe_cli_writes_checked_manifest(d, tmp_path):
    from studio.cli import app
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"source": d["clip"], "output": str(tmp_path / "cli.mp4"),
                               "items": [{"text": "Hi", "start": 0, "end": 1, "x": 10, "y": 10, "width": 120,
                                          "height": 40}]}), encoding="utf-8")
    runner = CliRunner()
    assert "image-popup" in runner.invoke(app, ["recipes"]).output
    r = runner.invoke(app, ["recipe", "lower-third", "--request", str(req), "-o", str(tmp_path / "job.json")])
    assert r.exit_code == 0, r.output
    assert jobs.load(tmp_path / "job.json").steps[0].op == "compose"


# --- audio timing and metrics --------------------------------------------------------------------

from studio.qc import audio_alignment, audio_metrics  # noqa: E402


@pytest.fixture(scope="module")
def speechy(d):
    """Pink-noise bursts (decorrelated like speech) on a 4 s clip, 48 kHz stereo."""
    out = d["t"] / "speechy.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i",
                  "anoisesrc=d=4:c=pink:r=48000:a=0.3,volume='0.15+0.85*gt(sin(2*PI*1.7*t),0)':eval=frame",
                  "-t", 4, "-ac", 2, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "pcm_s16le",
                  str(out).replace(".mp4", ".mov")])
    return str(out).replace(".mp4", ".mov")


def test_silence_is_not_evidence_of_alignment(tmp_path):
    p = tmp_path / "silent.wav"
    media.ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", 1, "-c:a", "pcm_s16le", p])
    r = audio_alignment(str(p), str(p))
    assert r["conclusive"] is False and r["lag_samples"] is None
    m = audio_metrics(str(p))
    assert m["silent"] and m["samples"] == 48000 and m["clipped_samples"] == 0


def test_alignment_measures_known_delay_and_clipping(speechy, tmp_path):
    late = tmp_path / "late.wav"
    media.ffmpeg(["-i", speechy, "-af", "adelay=100S:all=1", "-c:a", "pcm_s16le", late])
    r = audio_alignment(speechy, str(late))
    assert r["conclusive"] and r["lag_samples"] == 100, r
    hot = tmp_path / "hot.wav"
    media.ffmpeg(["-i", speechy, "-af", "volume=8", "-c:a", "pcm_f32le", hot])
    assert audio_metrics(str(hot))["clipped_samples"] > 0


def test_cleanup_is_delay_compensated_everywhere(d, talk):
    speechy = talk  # real (SAPI) speech: the denoiser keeps it, so correlation is meaningful
    r = run(d, [speechy], [{"id": "c", "op": "audio_clean", "inputs": ["s0"], "params": {}}], "clean-sync")
    mid = r["steps"][0]["output"]
    a = audio_alignment(speechy, mid)
    assert a["conclusive"] and abs(a["lag_samples"]) <= 48, a  # spec: within 1 ms (high-pass phase lead ~0.5 ms)
    assert audio_metrics(mid)["samples"] == audio_metrics(speechy)["samples"]
    c = run(d, [speechy, d["logo"]], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"],
                                       "params": {"layers": [layer()], "audio": {}}}], "compose-clean")
    cm = c["steps"][0]["output"]
    assert audio_metrics(cm)["samples"] == audio_metrics(speechy)["samples"]
    b = audio_alignment(speechy, cm)
    assert b["conclusive"] and abs(b["lag_samples"]) <= 48, b


def test_noise_print_gate_and_linear_loudness(d, tmp_path):
    """Fan hum + hiss under bursts: the noise-print gate removes it in pauses, keeps sync and length;
    linear loudness reaches target with one constant gain (pauses not lifted), still in sync."""
    import numpy as np
    src = tmp_path / "hum.mov"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i",
                  "anoisesrc=d=4:c=pink:r=48000:a=0.02[n];sine=f=100:r=48000:d=4,volume=0.03[h];"
                  "anoisesrc=d=4:c=white:r=48000:a=0.3:seed=7,volume='gt(t,1)*gt(sin(2*PI*1.7*t),0)':eval=frame[s];"
                  "[n][h][s]amix=inputs=3:normalize=0", "-t", 4, "-ac", 2, "-c:v", "libx264", "-preset", "ultrafast",
                  "-c:a", "pcm_s16le", src])
    steps = [{"id": "c", "op": "audio_clean", "inputs": ["s0"], "params": {"noise_print": [[0.1, 0.9]]}},
             {"id": "l", "op": "loudnorm", "inputs": ["c"], "params": {"I": -16, "TP": -1, "linear": True}}]
    r = run(d, [str(src)], steps, "noise-print")
    clean, level = r["steps"][0]["output"], r["steps"][1]["output"]

    def pause_db(path):  # a later pause (sin < 0) the print never saw
        x = np.frombuffer(subprocess.run([media.tool("ffmpeg"), "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                                          "-"], capture_output=True, check=True).stdout, np.float32)
        return 20 * np.log10(np.sqrt((x[int(1.5 * 48000):int(1.72 * 48000)] ** 2).mean()))
    assert pause_db(clean) < pause_db(str(src)) - 12, (pause_db(str(src)), pause_db(clean))
    for out in (clean, level):
        a = audio_alignment(str(src), out)
        assert a["conclusive"] and abs(a["lag_samples"]) <= 48, a
        assert audio_metrics(out)["samples"] == audio_metrics(str(src))["samples"]
    assert abs(audio_metrics(level)["loudness_lufs"] + 16) <= 1
    gain = r["steps"][1]["meta"]["loudnorm"]["gain_db"]
    assert abs((pause_db(level) - pause_db(clean)) - gain) <= 1.5  # pauses got the same gain as speech


# --- review bundle -------------------------------------------------------------------------------

def test_review_bundle_reuses_and_flags_pending(d, tmp_path):
    from studio.review import bundle, record
    job = jobs.Job(name="rv", sources={"s0": {"path": d["clip"]}, "s1": {"path": d["logo"]}},
                   steps=[{"id": "c", "op": "compose", "inputs": ["s0", "s1"], "params": {"layers": [layer()]}}],
                   output=str(tmp_path / "rv.mp4"), profile="h264-master")
    r = jobs.run(job, ws=d["ws"], log=lambda m: None)
    b = bundle(job, r, tmp_path / "bundle")
    assert b["technical_ok"] and not b["reused"], json.dumps(b["checks"], indent=1, default=str)
    assert b["checks"]["preserved_outside_layers"]["ok"] and b["checks"]["clock"]["ok"]
    first = next(c for c in b["checkpoints"] if c["checkpoint"] == "L0-first")
    last = next(c for c in b["checkpoints"] if c["checkpoint"] == "L0-last")
    assert (first["frame"], last["frame"]) == (15, 29) and first["edit"] and first["source"]
    assert bundle(job, r, tmp_path / "bundle")["reused"]
    reviews = json.loads(Path(b["reviews"]).read_text(encoding="utf-8"))["reviews"]
    assert all(v["status"] == "pending" for v in reviews.values())
    record(tmp_path / "bundle", "background-detail", "tester", "pass", [])
    with pytest.raises(ValueError):
        record(tmp_path / "bundle", "made-up", "tester", "pass", [])


def test_compose_hdr_transparent_margin_stays_base_while_animating(d, hlg, tmp_path):
    """Regression: per-frame scaling before the HDR alpha split left opaque bands at card edges."""
    from PIL import Image, ImageDraw
    card = tmp_path / "margin.png"
    im = Image.new("RGBA", (200, 120))
    ImageDraw.Draw(im).rectangle((10, 10, 170, 90), fill=(240, 230, 210, 255))  # bottom/right margin transparent
    im.save(card)
    p = {"layers": [layer(animation="slide-scale", start=0.1, end=1.4, x=40, y=40, width=200, height=120)]}
    r = run(d, [hlg, str(card)], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"], "params": p}],
            "compose-margin", validate={"against": "s0"}, profile="4k60-master")
    import numpy as np
    a, b = _y10(hlg), _y10(r["steps"][0]["output"])
    ys, xs = np.nonzero(a[30] != b[30])  # settled: only the opaque card area (x 50-210, y 50-130) changes
    assert abs(ys.min() - 50) <= 1 and abs(ys.max() - 130) <= 1 and abs(xs.min() - 50) <= 1 and abs(xs.max() - 210) <= 1, (
        ys.min(), ys.max(), xs.min(), xs.max())
    for n in (5, 8):  # mid-entrance: transparent right margin stays base
        assert (a[n, 60:130, 225:240] == b[n, 60:130, 225:240]).all(), n


def test_compose_first_and_last_enabled_frames_are_visible_and_symmetric(d, tmp_path):
    """Regression: `fade` zeroed the first card frame, and looping image inputs lagged the base."""
    from PIL import Image
    card = tmp_path / "white.png"
    Image.new("RGBA", (100, 100), (255, 255, 255, 255)).save(card)
    base = lavfi(tmp_path / "black.mp4", "color=c=black:s=320x240:r=30", dur=3)
    p = {"layers": [{"input": 1, "start": 1.0, "end": 2.0, "x": 10, "y": 10, "width": 100, "height": 100,
                     "animation": "fade", "fade": 0.3}]}
    r = run(d, [base, str(card)], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"], "params": p}], "compose-fade")
    a = [round(float(f[60, 60, 0]) / 255, 2) for f in media.read_frames(r["steps"][0]["output"], 320, 240)]
    assert a[29] == 0 and a[60] == 0  # outside [1.0, 2.0)
    assert a[30:34] == [0.11, 0.22, 0.33, 0.44] and a[56:60] == [0.44, 0.33, 0.22, 0.11], (a[28:36], a[54:62])


def test_compose_cue_at_zero_shows(d, tmp_path):
    """Regression: a card cued at 0 put its lead-in frame at -1 frame and `fade` never showed the card."""
    from PIL import Image
    card = tmp_path / "white0.png"
    Image.new("RGBA", (100, 100), (255, 255, 255, 255)).save(card)
    base = lavfi(tmp_path / "black0.mp4", "color=c=black:s=320x240:r=30", dur=2)
    p = {"layers": [{"input": 1, "start": 0, "end": 1.5, "x": 10, "y": 10, "width": 100, "height": 100,
                     "animation": "slide-scale"}]}
    r = run(d, [base, str(card)], [{"id": "c", "op": "compose", "inputs": ["s0", "s1"], "params": p}], "compose-zero")
    a = [f[60, 60, 0] for f in media.read_frames(r["steps"][0]["output"], 320, 240)]
    assert a[15] > 200 and a[50] == 0, (a[15], a[50])


def test_review_extract_many_frames(tmp_path):
    """Regression: 37 caption cards gave >100 select terms and ffmpeg refused the expression."""
    from studio.review import _extract
    clip = lavfi(tmp_path / "long.mp4", "testsrc2=s=64x64:r=30", dur=6)
    _extract(clip, list(range(150)), tmp_path, "x", False, [])
    assert len(list(tmp_path.glob("x-*.png"))) == 150


def test_review_bundle_with_silent_source_and_audio_output(d, tmp_path):
    """Regression: a generated (silent) first source with a mixed-in soundtrack broke the waveform step."""
    from studio.review import bundle
    job = jobs.Job(name="rv-silent", sources={"s0": {"path": d["clip2"]}, "s1": {"path": d["music"]}},
                   steps=[{"id": "a", "op": "audio", "inputs": ["s0", "s1"], "params": {"mode": "replace"}}],
                   output=str(tmp_path / "rv-silent.mp4"), profile="h264-master")
    r = jobs.run(job, ws=d["ws"], log=lambda m: None)
    b = bundle(job, r, tmp_path / "bundle-silent")
    assert (Path(b["evidence_dir"]) / "waveform-edit.png").exists()


def test_rgb_source_encodes_to_yuv_master(d, tmp_path):
    """Regression: RGB (gbr) sources passed their colour tags to x264, which rejected them."""
    src = tmp_path / "rgb.mkv"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=0.5", "-pix_fmt", "gbrp", "-c:v", "ffv1", src])
    r = run(d, [str(src)], [{"id": "f", "op": "fade", "inputs": ["s0"], "params": {"in": 0.1}}], "rgb-master")
    assert r["ok"] and media.info(r["output"])["video"]["color_space"] == "bt709"


def test_lower_third_per_item_style_font_and_motion(d, tmp_path):
    from studio.recipes import expand
    req = {"source": d["clip"], "output": str(tmp_path / "lt2.mp4"), "profile": "h264-master", "items": [
        {"text": "Nail Revolution", "start": 0, "end": 1.5, "x": 20, "y": 20, "width": 400, "height": 120,
         "style": "shadow", "font": "C:/Windows/Fonts/georgiaz.ttf", "palette": {"label_text": "#ffffff"},
         "entrance": 0.5, "slide": 80, "initial_scale": 0.8},
        {"text": "caption", "start": 0.5, "end": 1.5, "x": 20, "y": 280, "width": 300, "height": 60}]}
    job = expand("lower-third", req, tmp_path / "assets")
    L = job.steps[0].params["layers"]
    assert (L[0]["entrance"], L[0]["slide"], L[0]["initial_scale"]) == (0.5, 80, 0.8) and "entrance" not in L[1]
    from PIL import Image
    shadow = Image.open(job.sources["card0"].path)
    assert shadow.getpixel((0, 0))[3] == 0 and shadow.getchannel("A").getextrema()[1] == 255  # no box, opaque text
    assert _render(d, job)["ok"]


def test_taps_suppressed_steady_sound_kept(d, tmp_path):
    """Nail taps on a handheld mic: short broadband hits drop; a steady voice-like tone is untouched."""
    import numpy as np
    src = tmp_path / "taps.mov"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i",
                  "sine=f=220:r=48000:d=3,volume=0.2[v];"
                  "aevalsrc='0.6*(2*random(0)-1)*lt(mod(t-0.25\,0.5)\,0.006)':s=48000:d=3[t];"
                  "[v][t]amix=inputs=2:normalize=0", "-t", 3, "-ac", 2, "-c:v", "libx264", "-preset", "ultrafast",
                  "-c:a", "pcm_s16le", src])
    r = run(d, [str(src)], [{"id": "c", "op": "audio_clean", "inputs": ["s0"],
                            "params": {"highpass": 60, "taps": True}}], "taps")
    out = r["steps"][0]["output"]

    def pcm(p):
        return np.frombuffer(subprocess.run([media.tool("ffmpeg"), "-v", "error", "-i", p, "-f", "f32le", "-ac", "1",
                                             "-af", "highpass=f=2000", "-"], capture_output=True, check=True).stdout,
                             np.float32)
    a, b = pcm(str(src)), pcm(out)
    taps = [int((0.25 + 0.5 * k) * 48000) for k in range(1, 5)]
    peak = lambda x: np.mean([np.abs(x[i - 240:i + 720]).max() for i in taps])  # noqa: E731
    assert 20 * np.log10(peak(b) / peak(a)) < -8, (peak(a), peak(b))
    assert r["steps"][0]["meta"]["audio_clean"]["tap_frames"] > 0
    assert audio_metrics(out)["samples"] == audio_metrics(str(src))["samples"]
    plain = run(d, [str(src)], [{"id": "c", "op": "audio_clean", "inputs": ["s0"], "params": {"highpass": 60}}],
                "no-taps")["steps"][0]["output"]
    def between(p):  # level between hits (0.30-0.70 s etc.): the steady sound must be untouched
        x = np.frombuffer(subprocess.run([media.tool("ffmpeg"), "-v", "error", "-i", p, "-f", "f32le", "-ac", "1", "-"],
                                         capture_output=True, check=True).stdout, np.float32)
        return [20 * np.log10(np.sqrt((x[int(a * 48000):int((a + 0.4) * 48000)] ** 2).mean())) for a in (0.3, 1.3, 2.3)]
    assert np.allclose(between(out), between(plain), atol=0.5), (between(out), between(plain))


def test_gain_ranges_level_one_speaker(d, tmp_path):
    import numpy as np
    src = tmp_path / "tone.mov"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i", "sine=f=300:r=48000:d=3",
                  "-t", 3, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "pcm_s16le", src])
    r = run(d, [str(src)], [{"id": "g", "op": "gain", "inputs": ["s0"], "params": {"ranges": [[1, 2, -12]]}}], "gain")
    out = r["steps"][0]["output"]
    x = np.frombuffer(subprocess.run([media.tool("ffmpeg"), "-v", "error", "-i", out, "-f", "f32le", "-ac", "1", "-"],
                                     capture_output=True, check=True).stdout, np.float32)
    db = lambda a, b: 20 * np.log10(np.sqrt((x[int(a * 48000):int(b * 48000)] ** 2).mean()))  # noqa: E731
    assert abs(db(1.2, 1.8) - db(0.2, 0.8) + 12) < 0.3 and abs(db(2.2, 2.8) - db(0.2, 0.8)) < 0.3
    assert audio_metrics(out)["samples"] == audio_metrics(str(src))["samples"]


@pytest.mark.skipif(not (Path(__file__).parents[1] / "workers/deepfilter/.venv/Scripts/python.exe").exists(),
                    reason="deepfilter worker not installed")
def test_speech_enhance_removes_noise_keeps_timing(d, tmp_path):
    import numpy as np
    src = tmp_path / "noisy.mov"  # hum + hiss, then a voice-like harmonic burst train from 1 s
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i",
                  "anoisesrc=d=3:c=pink:r=48000:a=0.03[n];sine=f=100:r=48000:d=3,volume=0.05[h];"
                  "aevalsrc='gte(t\,1)*0.2*(sin(2*PI*180*t)+0.6*sin(2*PI*360*t)+0.4*sin(2*PI*540*t))*(0.6+0.4*sin(2*PI*4*t))':s=48000:d=3[v];"
                  "[n][h][v]amix=inputs=3:normalize=0", "-t", 3, "-ac", 2, "-c:v", "libx264", "-preset", "ultrafast",
                  "-c:a", "pcm_s16le", src])
    r = run(d, [str(src)], [{"id": "e", "op": "speech_enhance", "inputs": ["s0"], "params": {"mono": True}}], "df")
    out = r["steps"][0]["output"]
    pcm = lambda p: np.frombuffer(subprocess.run([media.tool("ffmpeg"), "-v", "error", "-i", p, "-f", "f32le", "-ac", "1",  # noqa: E731
                                                  "-"], capture_output=True, check=True).stdout, np.float32)
    a, b = pcm(str(src)), pcm(out)
    db = lambda x: 20 * np.log10(np.sqrt((x[int(0.2 * 48000):int(0.9 * 48000)] ** 2).mean()) + 1e-12)  # noqa: E731
    assert db(b) < db(a) - 15, (db(a), db(b))  # noise-only first second
    assert audio_metrics(out)["samples"] == audio_metrics(str(src))["samples"]
    assert media.info(out)["audio"]["channels"] == 2
    assert r["steps"][0]["meta"]["speech_enhance"]["model"] == "DeepFilterNet3"


def test_color_tone_darkens_hdr_without_8bit(d, hlg):
    r = run(d, [hlg], [{"id": "c", "op": "color", "inputs": ["s0"], "params": {"tone": 1.3, "luma": 0.95}}],
            "tone-hlg", profile="4k60-master")
    mid = r["steps"][0]["output"]
    assert media.info(mid)["video"]["pix_fmt"] == "yuv420p10le"
    a, b = _y10(hlg).astype(int), _y10(mid).astype(int)
    assert b.mean() < a.mean() - 10 and b.min() >= 64 - 1  # darker, legal black kept
    levels = len(set(b[0].ravel().tolist()))
    assert levels > 256, levels  # still 10-bit precision, not 8-bit steps
