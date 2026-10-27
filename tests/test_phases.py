"""Phases 3-6: editorial (STT/cuts/captions/index), restoration, generative ops, agent reliability."""
import json
import urllib.parse
import shutil
import subprocess
import sys

import numpy as np
import pytest

from studio import jobs, media, speech
from studio.jobs import Job
from test_studio import d, lavfi, run  # noqa: F401  (shared module fixture)


@pytest.fixture(scope="module")
def talk(d):
    """Speech clip: sentence, 1.5 s silence, sentence with a filler (Windows SAPI TTS)."""
    if sys.platform != "win32":
        pytest.skip("needs Windows SAPI for synthetic speech")
    wav = [d["t"] / f"say{i}.wav" for i in (1, 2)]
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{wav[0]}'); $s.Speak('Hello and welcome to the studio.'); "
          f"$s.SetOutputToWaveFile('{wav[1]}'); $s.Speak('Um, today we test the captions.'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    out = d["t"] / "talk.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30", "-i", wav[0], "-i", wav[1], "-filter_complex",
                  "[1:a]aresample=48000,apad=pad_dur=1.5[a1];[2:a]aresample=48000[a2];[a1][a2]concat=n=2:v=0:a=1[a]",
                  "-map", "0:v", "-map", "[a]", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                  "-pix_fmt", "yuv420p", "-c:a", "aac", out])
    return str(out)


def test_transcribe_captions_search(d, talk):
    t = speech.transcribe(talk, "base", cache_dir=d["ws"] / "cache")
    text = " ".join(s["text"] for s in t["segments"]).lower()
    assert t["language"] == "en" and "welcome" in text and "captions" in text
    srt = speech.to_srt(t)
    assert srt.startswith("1\n00:00:0") and "-->" in srt and speech.to_vtt(t).startswith("WEBVTT")
    assert speech.phrase_ranges(t, "welcome to the studio")
    r = speech.index(d["t"], d["ws"], log=lambda m: None)
    assert r["items"] >= 1
    hits = [h for h in speech.search("welcome studio", d["ws"]) if h["path"].endswith("talk.mp4")]
    assert hits and "welcome" in hits[0]["text"].lower(), hits
    # semantic: no word in common with "today we test the captions", same meaning
    sem = [h for h in speech.search("checking subtitles", d["ws"]) if h["path"].endswith("talk.mp4")]
    assert sem and "caption" in sem[0]["text"].lower() and sem[0]["match"] == "semantic", sem


def test_cut_silence_and_phrase(d, talk):
    total = media.info(talk)["duration"]
    r = run(d, [talk], [{"id": "c", "op": "cut", "inputs": ["s0"], "params": {"silence": True, "fillers": True}}], "cutsil")
    assert r["ok"], json.dumps(r["validation"], indent=1)
    m = r["steps"][0]["meta"]["cut"]
    assert m["removed_seconds"] > 1.0 and abs(r["validation"]["summary"]["duration"] - m["kept_seconds"]) < 0.1
    r2 = run(d, [talk], [{"id": "c", "op": "cut", "inputs": ["s0"], "params": {"phrases": ["welcome to the studio"]}}], "cutph")
    assert r2["ok"] and 0.5 < total - r2["validation"]["summary"]["duration"] < 2.5
    r3 = run(d, [talk], [{"id": "cap", "op": "captions", "inputs": ["s0"], "params": {}}], "cap")
    assert r3["ok"] and r3["steps"][0]["meta"]["captions"]["cues"] >= 2


def test_cut_ranges(d):
    r = run(d, [d["clip"]], [{"id": "c", "op": "cut", "inputs": ["s0"], "params": {"remove": [[0.2, 0.7], [1.5, 1.8]]}}],
            "cutr", validate={"duration": 1.2})
    assert r["ok"], json.dumps(r["validation"], indent=1)
    assert speech.merge([[1, 2], [1.5, 3], [5, 6]]) == [[1, 3], [5, 6]]
    assert speech.keep_from_remove([[1, 3], [5, 6]], 7) == [[0, 1], [3, 5], [6, 7]]


def _sr_model(path):
    """Tiny ONNX 'super-resolution' net: nearest 2x resize (exercises the tiled onnx backend)."""
    import onnx
    from onnx import TensorProto, helper
    g = helper.make_graph(
        [helper.make_node("Resize", ["x", "", "s"], ["y"], mode="nearest")], "sr",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, None, None])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, None, None])],
        [helper.make_tensor("s", TensorProto.FLOAT, [4], [1, 1, 2, 2])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8  # loadable by older onnxruntime builds
    onnx.save(m, path)
    return str(path)


RESTORE = [
    ("stabilize", [], {}, 2, None),
    ("stabilize", [], {"backend": "deshake"}, 2, None),
    ("denoise", [], {}, 2, None),
    ("denoise", [], {"backend": "nlmeans", "strength": 2}, 2, None),
    ("sharpen", [], {"amount": 0.6}, 2, None),
    ("deinterlace", [], {}, 2, None),
    ("lens", [], {"k1": -0.1}, 2, None),
    ("upscale", [], {"factor": 2}, 2, [1280, 720]),
    ("upscale", [], {"backend": "onnx", "factor": 2}, 2, [1280, 720]),
    ("interpolate", [], {"fps": 60, "backend": "blend"}, 2, None),
    ("interpolate", [], {"fps": 60}, 2, None),
    ("match_color", ["clip2"], {}, 2, None),
    ("stylize", [], {"style": "vintage"}, 2, None),
    ("stylize", [], {"style": "sketch"}, 2, None),
    ("extend", [], {"mode": "canvas", "width": 1080, "height": 1080}, 2, [1080, 1080]),
    ("extend", [], {"mode": "hold", "duration": 1}, 3, None),
    ("extend", [], {"mode": "push", "duration": 1}, 3, None),
]


@pytest.mark.parametrize("op,extra,params,dur,size", RESTORE, ids=[f"{c[0]}-{i}" for i, c in enumerate(RESTORE)])
def test_restore_and_style_ops(d, op, extra, params, dur, size):
    if params.get("backend") == "onnx":
        params = params | {"model": _sr_model(d["t"] / "sr.onnx")}
    r = run(d, [d["clip"]] + [d[e] for e in extra],
            [{"id": "x", "op": op, "inputs": [f"s{i}" for i in range(1 + len(extra))], "params": params}],
            f"r-{op}-{abs(hash(json.dumps(params))) % 10000}", validate={"duration": dur})
    assert r["ok"], json.dumps(r["validation"], indent=1)
    if size:
        assert r["validation"]["summary"]["size"] == size


GRAB = {"id": "m", "op": "mask", "inputs": ["s0"],
        "params": {"target": "object", "backend": "grabcut", "box": [55, 125, 110, 110], "height": 360}}


def test_object_remove_inpaint(d):
    """Remove the moving blue square: plate fill should leave yellow everywhere."""
    r = run(d, [d["square"]], [GRAB, {"id": "rm", "op": "inpaint", "inputs": ["s0", "m"], "params": {}}], "objrm",
            validate={"against": "s0", "duration": 2})
    assert r["ok"], json.dumps(r["validation"], indent=1)
    for f in list(media.read_frames(r["output"], 640, 360))[::15]:
        f = f.astype(int)
        assert (f[..., 2] > 120).mean() < 0.01, "blue pixels left"  # yellow has blue~0
    t = run(d, [d["square"]], [GRAB, {"id": "rm", "op": "inpaint", "inputs": ["s0", "m"], "params": {"method": "telea"}}],
            "objrm2")
    assert t["ok"]


def test_tracked_insert_and_relight(d):
    r = run(d, [d["square"], d["logo"]], [{"id": "i", "op": "insert", "inputs": ["s0", "s1"],
                                            "params": {"box": [60, 130, 100, 100]}}], "ins")
    assert r["ok"], json.dumps(r["validation"], indent=1)
    last = list(media.read_frames(r["output"], 640, 360))[-1].astype(int)
    assert last[180, 265, 0] > 180 and last[180, 265, 2] < 80, last[180, 265]  # red logo followed the square
    rl = run(d, [d["square"]], [GRAB, {"id": "l", "op": "relight", "inputs": ["s0", "m"],
                                       "params": {"background": {"brightness": -0.3}}}], "relight")
    assert rl["ok"]
    f0 = list(media.read_frames(rl["output"], 640, 360))[0].astype(int)
    assert f0[20, 600].sum() < 400 and f0[180, 110, 2] > 150  # background darkened, subject kept


def test_generate_without_sources(d, tmp_path):
    job = Job(name="gen", sources={}, steps=[{"id": "g", "op": "generate", "inputs": [],
                                              "params": {"prompt": "sunset", "width": 640, "height": 360,
                                                         "duration": 1.5, "fps": 30}}],
              output=str(tmp_path / "gen.mp4"), profile="h264-master", validate={"duration": 1.5})
    r = jobs.run(job, ws=d["ws"], log=lambda m: None)
    assert r["ok"] and r["validation"]["summary"]["size"] == [640, 360], json.dumps(r["validation"], indent=1)
    from studio.generate import generate_media
    img = generate_media("plate", "image", tmp_path / "p", 320, 180)
    assert media.info(img)["video"]["width"] == 320
    if shutil.which("python") or sys.executable:
        import os
        os.environ["STUDIO_GEN_CMD"] = (f'"{sys.executable}" -c "import shutil,sys; shutil.copy(sys.argv[1], sys.argv[2])" '
                                        f'"{img}" {{out}}')
        try:
            out = generate_media("x", "image", tmp_path / "c.png", backend="command")
            assert media.info(out)["video"]["width"] == 320
        finally:
            del os.environ["STUDIO_GEN_CMD"]


def test_check_and_auto(d, tmp_path):
    from studio import agent
    from test_phases import GRAB
    job = Job(name="autojob", sources={"s0": {"path": d["square"]}},
              steps=[GRAB, {"id": "bg", "op": "background", "inputs": ["s0", "m"], "params": {"mode": "blur", "bogus": 1}}],
              output=str(tmp_path / "auto.mp4"), profile="h264-master")
    c = agent.check(job)
    assert c["ok"] and c["ai_steps"] == ["m"] and any("bogus" in w for w in c["warnings"]), c
    bad = job.model_copy(deep=True)
    bad.steps[1].inputs = ["s0", "nope"]
    assert not agent.check(bad)["ok"]
    r = agent.auto(job, ws=d["ws"], log=lambda m: None)
    assert r["ok"] and r["stage"] == "final" and len(r["rounds"]) == 1, r
    # refine ladder: low-confidence person mask -> stride 1, then next model
    j2 = Job(sources={"s": {"path": d["clip"]}}, steps=[{"id": "m", "op": "mask", "inputs": ["s"],
                                                         "params": {"target": "person", "stride": 2}}], output="x.mp4")
    low = {"steps": [{"step": "m", "meta": {"mask": {"low_confidence": True}}}]}
    assert agent._refine(j2, low) == ["m: stride -> 1"]
    assert agent._refine(j2, low) == ["m: model u2net_human_seg -> birefnet-portrait"]


def test_scheduler_selfcheck():
    r = subprocess.run([sys.executable, "-m", "studio.sched"], capture_output=True, text=True)
    assert r.returncode == 0 and "sched ok" in r.stdout, r.stderr


def test_ui_endpoints(d):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from studio import ui
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path, token=True, host=None):
        h = ({"X-Studio-Token": ui.TOKEN} if token else {}) | ({"Host": host} if host else {})
        return urllib.request.urlopen(urllib.request.Request(base + path, headers=h), timeout=30)

    try:
        assert "__TOKEN__" not in get("/", token=False).read().decode()
        for path, kw in (("/api/list", {"token": False}), ("/api/list", {"host": "evil.example"})):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(path, **kw)
            assert e.value.code == 403
        assert "manifests" in json.load(get("/api/list"))
        with pytest.raises(urllib.error.HTTPError):  # not a known file
            get("/api/file?path=" + urllib.parse.quote(str(d["clip"])))
        m = {"sources": {"s0": {"path": d["clip"]}}, "steps": [{"id": "f", "op": "fade", "inputs": ["s0"]}],
             "output": str(d["t"] / "ui.mp4")}
        req = urllib.request.Request(base + "/api/run", json.dumps({"mode": "check", "manifest": m}).encode(),
                                     {"X-Studio-Token": ui.TOKEN, "Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req))["ok"]
        q = urllib.parse.quote(str(d["clip"]))
        assert get(f"/api/frame?path={q}&time=0.5").read(2) == b"\xff\xd8"  # JPEG, now that the job named it
        r = urllib.request.Request(base + f"/api/file?path={q}", headers={"X-Studio-Token": ui.TOKEN, "Range": "bytes=0-99"})
        resp = urllib.request.urlopen(r)
        assert resp.status == 206 and len(resp.read()) == 100

        def post(path, body):
            return urllib.request.Request(base + path, json.dumps(body).encode(),
                                          {"X-Studio-Token": ui.TOKEN, "Content-Type": "application/json"})
        import base64
        png = d["logo"]  # a real PNG from the fixture
        saved = json.load(urllib.request.urlopen(post("/api/paint", {"png": base64.b64encode(open(png, "rb").read()).decode()})))
        assert saved["path"].endswith(".png") and "paint" in saved["path"]
        with pytest.raises(urllib.error.HTTPError):  # only PNGs are accepted
            urllib.request.urlopen(post("/api/paint", {"png": base64.b64encode(b"<script>").decode()}))
    finally:
        srv.shutdown()
