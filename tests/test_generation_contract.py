"""Slice 2: local generation contract (fake command worker + fake ComfyUI; no model downloads)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from studio import jobs, media, workers
from studio.generate import generate_media, provenance, select_output
from studio.qc import validate_audio
from test_studio import d, run  # noqa: F401  (shared module fixture)

FAKE = str(Path(__file__).with_name("fake_generator.py"))


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Isolated workspace with a fake command worker configured (workspace/workers.json)."""
    w = tmp_path / "ws"
    monkeypatch.setenv("STUDIO_WORKSPACE", str(w))
    jobs.workspace(w)
    cfg = {"fake": {"backend": "command", "kinds": ["voice", "music", "video"], "fields": ["prompt", "text", "voice",
                    "duration", "seed", "width", "height", "fps"],
                    "argv": [sys.executable, FAKE, "--request", "{request}", "--out", "{out}"], "timeout": 20},
           "slow": {"backend": "command", "kinds": ["voice"], "fields": ["prompt", "text"],
                    "argv": [sys.executable, FAKE, "--request", "{request}", "--out", "{out}"], "timeout": 2}}
    (w / "workers.json").write_text(json.dumps(cfg), encoding="utf-8")
    return w


def test_audio_only_validation(tmp_path):
    p = tmp_path / "tone.wav"
    media.ffmpeg(["-f", "lavfi", "-i", "sine=f=440:r=48000:d=0.5", "-c:a", "pcm_s24le", p])
    report = validate_audio(str(p), duration=.5)
    assert report["ok"], report
    assert media.info(p)["video"] is None and media.info(p)["audio"]["sample_rate"] == 48000
    assert not validate_audio(str(p), duration=2)["ok"]


def test_procedural_cannot_fake_voice(tmp_path):
    with pytest.raises(ValueError, match="voice"):
        generate_media("Speak this", "voice", tmp_path / "voice.wav", backend="procedural")


def test_command_worker_voice_provenance_and_literal_text(ws, tmp_path):
    text = 'Say {braces}, "quotes" & C:\\paths; $(rm -rf) `x`'
    out = generate_media("tone", "voice", tmp_path / "v.wav", worker="fake", text=text, duration=None)
    p = provenance(out)
    assert out.endswith("v.wav") and p["kind"] == "voice" and p["worker"] == "fake"
    assert p["meta"]["echo_text"] == text  # passed as data in the request file, never through a shell
    assert p["native"]["audio"]["sample_rate"] == 24000 and p["sha256"] == media.sha256(out)
    assert p["request"].get("duration") is None  # voice length comes from synthesis, not the 5 s visual default
    assert not list(tmp_path.glob("*.part*")) and not list(tmp_path.glob("*.request.json"))


@pytest.mark.parametrize("mode, match", [("corrupt", "validation"), ("nothing", "without writing"),
                                         ("fail", "model exploded")])
def test_command_worker_failures_publish_nothing(ws, tmp_path, mode, match):
    o = tmp_path / "o"
    with pytest.raises(RuntimeError, match=match):
        generate_media(mode, "voice", o / "v.wav", worker="fake")
    assert list(o.iterdir()) == []


def test_worker_timeout_and_unsupported_requests(ws, tmp_path):
    o = tmp_path / "o"
    with pytest.raises(RuntimeError, match="timed out"):
        generate_media("sleep", "voice", o / "v.wav", worker="slow")
    with pytest.raises(ValueError, match="does not generate image"):
        generate_media("x", "image", o / "i.png", worker="fake")
    with pytest.raises(ValueError, match="does not support"):
        generate_media("x", "voice", o / "v.wav", worker="slow", voice="af_heart")
    assert list(o.iterdir()) == []


def test_direct_generation_never_replaces(ws, tmp_path):
    (tmp_path / "o").mkdir()
    tmp_path = tmp_path / "o"
    keep = tmp_path / "v.wav"
    keep.write_bytes(b"user audio")
    with pytest.raises(FileExistsError):
        generate_media("tone", "voice", keep, worker="fake", text="hello")
    assert keep.read_bytes() == b"user audio"
    assert sorted(f.name for f in tmp_path.iterdir()) == ["v.wav"]


def test_audio_job_alone_and_mixed_into_video(ws, d, tmp_path):
    gen = {"id": "say", "op": "generate_audio", "inputs": [], "params": {"kind": "voice", "worker": "fake",
                                                                        "text": "hello studio", "prompt": "tone"}}
    alone = jobs.run(jobs.Job(name="voice", sources={}, steps=[gen], output=str(tmp_path / "voice.mp4")),
                     ws=ws, log=lambda m: None)
    assert alone["ok"] and alone["output"].endswith(".wav") and alone["profile"] == "wav", alone["validation"]
    mixed = jobs.Job(name="mix", sources={"clip": {"path": d["clip"]}},
                     steps=[gen, {"id": "mix", "op": "audio", "inputs": ["clip", "say"],
                                  "params": {"mode": "mix", "delay": 0.2}}],
                     output=str(tmp_path / "mixed.mp4"), profile="h264-master")
    r = jobs.run(mixed, ws=ws, log=lambda m: None)
    assert r["ok"] and media.info(r["output"])["audio"], r["validation"]
    from studio.agent import check
    bad = jobs.Job(name="bad", sources={}, steps=[gen, {"id": "f", "op": "fade", "inputs": ["say"]}],
                   output=str(tmp_path / "bad.mp4"))
    c = check(bad)
    assert not c["ok"] and any("audio-only" in e for e in c["errors"])


def test_generate_op_keeps_generated_audio(ws, d, tmp_path):
    job = jobs.Job(name="gv", sources={}, steps=[{"id": "g", "op": "generate", "inputs": [], "params": {
        "kind": "video", "worker": "fake", "prompt": "video", "width": 320, "height": 240, "fps": 24, "duration": 1}}],
        output=str(tmp_path / "gv.mp4"), profile="h264-master")
    r = jobs.run(job, ws=ws, log=lambda m: None)
    assert r["ok"] and media.info(r["output"])["audio"], r["validation"]
    assert r["steps"][0]["meta"]["generate"]["native"][:2] == [320, 240]


def test_readiness_needs_reviewed_evidence_and_tracks_changes(ws, tmp_path):
    model = tmp_path / "model.bin"
    raw = json.loads((ws / "workers.json").read_text(encoding="utf-8"))
    raw["fake"]["models"] = [str(model)]
    (ws / "workers.json").write_text(json.dumps(raw), encoding="utf-8")
    cfg = workers.worker_config("fake")
    assert workers.readiness(cfg)["state"] == "unavailable"
    model.write_bytes(b"weights v1")
    assert workers.readiness(cfg)["kinds"]["voice"]["state"] == "configured"
    out = generate_media("tone", "voice", tmp_path / "smoke.wav", worker="fake", text="smoke test")
    workers.record_smoke(cfg, provenance(out) | {"kind": "voice"})
    assert workers.readiness(cfg)["kinds"]["voice"]["state"] == "configured"  # not reviewed yet
    workers.record_review(cfg, "voice", "tester", "pass", ["clear tone"])
    assert workers.readiness(cfg)["kinds"]["voice"]["state"] == "ready"
    model.write_bytes(b"weights v2")  # same path, new model: evidence no longer applies
    assert workers.readiness(cfg)["kinds"]["voice"]["state"] == "configured"
    assert "workers" in __import__("studio.devices", fromlist=["x"]).capabilities()


# --- ComfyUI contract (fake local server) -------------------------------------------------------

class Fake(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        s = self.state
        if self.path.startswith("/system_stats"):
            return self._json({})
        if self.path.startswith("/object_info"):
            return self._json({"KSampler": {}, "SaveImage": {}, "LoadImage": {}})
        if self.path.startswith("/queue"):
            return self._json({"queue_running": [[0, "someone-else"]], "queue_pending": []})
        if self.path.startswith("/history/"):
            mode = s["mode"]
            if mode == "hang":
                return self._json({})
            if mode == "error":
                return self._json({"p1": {"status": {"status_str": "error", "messages": [
                    ["execution_error", {"exception_message": "out of memory"}]]}, "outputs": {}}})
            outs = {"7": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]},
                    "9": {"images": [{"filename": "final.png", "subfolder": "", "type": "output"}]}}
            if mode == "unsafe":
                outs["9"]["images"][0]["filename"] = "../../evil.png"
            return self._json({"p1": {"status": {"status_str": "success", "completed": True}, "outputs": outs}})
        if self.path.startswith("/view"):
            s.setdefault("viewed", []).append(self.path)
            body = Path(s["png"]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json({}, 404)

    def do_POST(self):
        s = self.state
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/upload/image":
            s["uploaded"] = True
            return self._json({"name": "stored_ref_1.png", "subfolder": "", "type": "input"})
        if self.path == "/prompt":
            s["workflow"] = json.loads(body)["prompt"]
            return self._json({"prompt_id": "p1", "number": 1, "node_errors": {}})
        if self.path.startswith("/api/jobs/"):
            s.setdefault("cancelled", []).append(self.path)
            return self._json({})
        if self.path == "/interrupt":
            s["global_interrupt"] = True
        return self._json({})


@pytest.fixture
def comfy(ws, tmp_path):
    png = tmp_path / "out.png"
    media.ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=64x48", "-frames:v", 1, png])
    Fake.state = {"mode": "ok", "png": str(png)}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 0, "text": "{{prompt}}"}},
                              "5": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
                              "7": {"class_type": "PreviewImage", "inputs": {}},
                              "9": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
    ckpt = tmp_path / "ckpt.safetensors"
    ckpt.write_bytes(b"weights")
    raw = json.loads((ws / "workers.json").read_text(encoding="utf-8"))
    raw["cf"] = {"models": [str(ckpt)], "backend": "comfyui", "kinds": ["image"], "url": f"http://127.0.0.1:{srv.server_port}",
                 "workflow": str(wf), "output_node": "9", "output_type": "images",
                 "bindings": {"seed": {"node": "3", "input": "seed"}}, "uploads": {"init": {"node": "5", "input": "image"}},
                 "node_classes": ["KSampler"], "fields": ["prompt", "seed", "width", "height", "references"], "timeout": 3}
    (ws / "workers.json").write_text(json.dumps(raw), encoding="utf-8")
    yield Fake.state
    srv.shutdown()


def test_comfy_binds_uploaded_reference_and_takes_designated_output(comfy, d, tmp_path):
    out = generate_media("a {weird} 'prompt'", "image", tmp_path / "img", 64, 48, seed=42, init=d["logo"], worker="cf")
    wf = comfy["workflow"]
    assert wf["5"]["inputs"]["image"] == "stored_ref_1.png"  # the name the server returned, not ours
    assert wf["3"]["inputs"]["seed"] == 42 and wf["3"]["inputs"]["text"] == "a {weird} 'prompt'"
    assert "final.png" in comfy["viewed"][0] and media.info(out)["video"]["width"] == 64


@pytest.mark.parametrize("mode, match", [("error", "out of memory"), ("unsafe", "unsafe"), ("hang", "timed out")])
def test_comfy_failures_are_prompt_and_owned(comfy, tmp_path, mode, match):
    comfy["mode"] = mode
    with pytest.raises((RuntimeError, ValueError), match=match):
        generate_media("x", "image", tmp_path / "img", 64, 48, worker="cf")
    assert not list(tmp_path.glob("img*")) and not list(tmp_path.glob("*.part*"))
    if mode == "hang":  # only our prompt is cancelled; never the global interrupt
        assert comfy["cancelled"] == ["/api/jobs/p1/cancel"] and "global_interrupt" not in comfy


def test_select_output_is_explicit():
    outputs = {"preview": {"images": [{"filename": "draft.png"}]}, "final": {"videos": [{"filename": "done.mp4"}]}}
    assert select_output(outputs, node="final", kind="video")["filename"] == "done.mp4"
    with pytest.raises(ValueError, match="missing"):
        select_output(outputs, node="missing", kind="video")
    with pytest.raises(ValueError, match="no image"):
        select_output(outputs, node="final", kind="image")
    two = {"f": {"images": [{"filename": "a.png"}, {"filename": "b.png"}]}}
    with pytest.raises(ValueError, match="explicit index"):
        select_output(two, node="f", kind="image")
    with pytest.raises(ValueError, match="loopback"):
        workers.Worker  # noqa: B018
        from studio.generate import _local
        _local("http://example.com:8188")


def test_review_regressions_model_dirs_stdout_and_audio_jobs(ws, tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "sub").mkdir(parents=True)
    (bundle / "sub" / "w.bin").write_bytes(b"a")
    raw = json.loads((ws / "workers.json").read_text(encoding="utf-8"))
    raw["fake"]["models"] = [str(bundle)]
    (ws / "workers.json").write_text(json.dumps(raw), encoding="utf-8")
    cfg = workers.worker_config("fake")
    before = workers.fingerprint(cfg)  # a model folder fingerprints its files (no PermissionError)
    (bundle / "sub" / "w.bin").write_bytes(b"b")
    assert workers.fingerprint(cfg) != before
    gen = {"id": "say", "op": "generate_audio", "inputs": [], "params": {"kind": "voice", "worker": "fake",
                                                                        "text": "again", "prompt": "tone"}}
    job = lambda: jobs.Job(name="again", sources={}, steps=[gen], output=str(tmp_path / "again.mp4"))  # noqa: E731
    first = jobs.run(job(), ws=ws, log=lambda m: None)
    assert first["output"].endswith(".wav")
    assert jobs.run(job(), ws=ws, log=lambda m: None).get("reused")  # identical audio job: verified reuse


@pytest.mark.parametrize("fps,duration,match", [(64, 2.0625, "fps"), (16, 2, "4n\\+1"), (16, 4.0625, "frames"),
                                                   (0, 1, "positive"), (float("nan"), 1, "positive"),
                                                   (16, float("inf"), "positive"), (16, -1, "positive")])
def test_video_limits_reject_untested_rates_and_latent_counts(ws, fps, duration, match):
    cfg = workers.worker_config("fake")
    cfg["limits"] = {"max_fps": 16, "max_frames": 33, "frame_step": 4}
    with pytest.raises(ValueError, match=match):
        workers.request_check(cfg, {"kind": "video", "width": 832, "height": 480,
                                    "fps": fps, "duration": duration})
    workers.request_check(cfg, {"kind": "video", "width": 832, "height": 480,
                                "fps": 16, "duration": 2.0625})


def test_video_limits_accept_seventeen_frames(ws):
    cfg = workers.worker_config("fake")
    cfg["limits"] = {"max_fps": 16, "max_frames": 33, "frame_step": 4}
    workers.request_check(cfg, {"kind": "video", "fps": 16, "duration": 17/16})
