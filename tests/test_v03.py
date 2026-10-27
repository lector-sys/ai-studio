"""v0.3 checks: HDR10 passthrough, chunked long-footage steps, identity tracking, cache prune."""
import numpy as np
import pytest

from studio import jobs, media
from studio.jobs import Job


@pytest.fixture(scope="module")
def t(tmp_path_factory):
    return tmp_path_factory.mktemp("v03")


def run(t, src, steps, name, profile="archive"):
    job = Job(name=name, sources={"s": {"path": str(src)}}, steps=steps, output=str(t / f"{name}.mkv"), profile=profile)
    return jobs.run(job, ws=t / "ws", log=lambda m: None)


def test_hdr10_metadata_survives_master(t):
    src = t / "hdr.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-t", 1, "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                  "-x265-params", "hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                  "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400",
                  "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", src])
    want = media.hdr10(str(src))
    assert want["cll"] == [1000, 400] and want["display"][8] == 1000.0
    for prof in ("4k60-master", "av1-master"):
        job = Job(name=f"hdr-{prof}", sources={"s": {"path": str(src)}}, output=str(t / f"hdr-{prof}.mp4"), profile=prof,
                  steps=[{"id": "t", "op": "trim", "inputs": ["s"], "params": {"start": 0.2}}])
        r = jobs.run(job, ws=t / "ws", log=lambda m: None)
        got = media.hdr10(r["output"])
        assert r["ok"] and got["cll"] == want["cll"], prof
        assert np.allclose(got["display"], want["display"], atol=1e-3), prof
        assert media.info(r["output"])["video"]["color_transfer"] == "smpte2084"


def test_chunked_step_matches_whole(t):
    src = t / "long.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i", "sine=d=3.2", "-t", 3.2,
                  "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", src])
    whole = run(t, src, [{"id": "d", "op": "denoise", "inputs": ["s"], "params": {}, "chunk": 0}], "whole")
    parts = run(t, src, [{"id": "d", "op": "denoise", "inputs": ["s"], "params": {"strength": 4}, "chunk": 1}], "parts")
    assert parts["ok"] and parts["steps"][0]["meta"]["chunks"] == 4
    a, b = list(media.read_frames(whole["output"], 320, 180)), list(media.read_frames(parts["output"], 320, 180))
    assert len(a) == len(b) == 96
    assert max(np.abs(x.astype(int) - y).mean() for x, y in zip(a, b)) < 1  # 1 s lead-in hides the seams
    assert media.info(parts["output"])["audio"]
    assert not list((t / "ws" / "cache").glob("*-c0*"))  # chunk files cleaned up


def test_identity_tracking_through_occlusion(t):
    """A red square passes behind a blue one; the mask must stay on red, before and after."""
    src = t / "cross.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "color=c=0xe0e0e0:s=320x180:r=30[b];color=c=red:s=40x40:r=30[r];"
                  "color=c=blue:s=40x40:r=30[u];[b][r]overlay=x='20+t*80':y=60[br];[br][u]overlay=x='260-t*80':y=70"
                  ":shortest=1", "-t", 3, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", src])
    r = run(t, src, [{"id": "m", "op": "mask", "inputs": ["s"], "params": {
        "target": "object", "backend": "grabcut", "box": [15, 55, 50, 50], "height": 180}}], "cross")
    frames = list(media.read_frames(r["output"], 320, 180))
    for k in (0, 30, 60, 89):
        xs = np.nonzero(frames[k][..., 0] > 127)[1]
        assert len(xs) and abs(xs.mean() - (40 + k / 30 * 80)) < 10, k
    assert not r["steps"][0]["meta"]["mask"]["low_confidence"]


def test_cache_prune(t):
    ws = jobs.workspace(t / "prune")
    (ws / "cache" / "x.mkv").write_bytes(b"0" * 1000)
    (ws / "proxies" / "y.mp4").write_bytes(b"0" * 1000)
    assert jobs.prune(7, t / "prune")["files"] == 0  # fresh files are kept
    assert jobs.prune(None, t / "prune") == {"files": 2, "freed_gb": 0.0}


def _model_or_skip(name):
    from studio import models
    try:
        return models.path(name)
    except Exception as e:  # noqa: BLE001 - offline with an empty HF cache
        pytest.skip(f"model {name} unavailable: {e}")


def test_rife_interpolation_follows_motion(t):
    _model_or_skip("rife_4.9")
    src = t / "sq.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "color=c=gray:s=256x144:r=15[b];color=c=white:s=32x32:r=15[s];"
                  "[b][s]overlay=x='20+t*120':y=56:shortest=1", "-f", "lavfi", "-i", "sine=d=1", "-t", 1,
                  "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", src])
    r = run(t, src, [{"id": "i", "op": "interpolate", "inputs": ["s"], "params": {"fps": 30, "backend": "onnx"}}], "rife")
    assert r["ok"] and r["steps"][0]["meta"]["interpolate"]["frames"] == 30
    xs = [np.nonzero(f[..., 0] > 200)[1].mean() for f in list(media.read_frames(r["output"], 256, 144))[:6]]
    assert np.allclose(np.diff(xs), 4, atol=1)  # 8 px per source frame -> 4 px per output frame, synthesized ones too


def test_face_restore_on_portrait(t):
    _model_or_skip("gfpgan_1.4")
    import cv2
    from skimage import data
    cv2.imwrite(str(t / "astronaut.png"), cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))  # public-domain NASA photo
    src = t / "face.mp4"
    media.ffmpeg(["-loop", 1, "-i", t / "astronaut.png", "-t", 0.2, "-r", 10, "-c:v", "libx264", "-preset", "ultrafast",
                  "-pix_fmt", "yuv420p", src])
    r = run(t, src, [{"id": "f", "op": "face_restore", "inputs": ["s"], "params": {"strength": 1}}], "faces")
    assert r["ok"] and r["steps"][0]["meta"]["face_restore"]["frames_with_faces"] == 2
    a, b = next(media.read_frames(str(src), 512, 512)), next(media.read_frames(r["output"], 512, 512))
    diff = np.abs(a.astype(int) - b).mean(2)
    assert diff[80:170, 180:260].mean() > 2 * diff[400:, :].mean() + 1  # the face changed, the suit did not


def test_recolor_ground_spares_colourful_subjects(t):
    """Grey textured ground at the bottom, flat grey sky on top, an orange 'ant' on the ground:
    region=ground + sat_max turns only the ground red."""
    src = t / "ground.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "color=c=0x909090:s=320x180:r=30[s];color=c=0x808080:s=320x100:r=30,"
                  "noise=c0s=60:c0f=t[g];[s][g]overlay=0:80[b];color=c=0xd07020:s=20x12:r=30[a];[b][a]overlay=150:130:shortest=1",
                  "-t", 0.5, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", src])
    r = run(t, src, [{"id": "r", "op": "recolor", "inputs": ["s"],
                      "params": {"to": "#c01e19", "region": "ground", "sat_max": 0.12}}], "recolor")
    f = list(media.read_frames(r["output"], 320, 180))[-1].astype(int)
    assert r["ok"]
    assert (f[150:175, 20:120, 0] - f[150:175, 20:120, 1]).mean() > 40  # ground is red now
    assert abs(f[10:60, :, 0] - f[10:60, :, 1]).mean() < 8  # sky stays grey
    ant = f[133:139, 154:166]
    assert ant[..., 0].mean() > ant[..., 2].mean() + 60 and ant[..., 1].mean() > 80  # still orange, not red


def test_look_match_regrades_scene_and_keeps_warm_subjects(t):
    """Grey scene + a warm (amber) subject, regraded to a terracotta reference: the scene turns red-earth,
    the subject stays warmer and brighter than its surroundings (no mask)."""
    import cv2
    ref = np.zeros((200, 300, 3), np.uint8)
    for y in range(200):  # terracotta gradient, dark to light
        ref[y] = cv2.cvtColor(np.uint8([[[60 + y // 2, 160, 155]]]), cv2.COLOR_LAB2BGR)[0, 0]
    cv2.imwrite(str(t / "ref.png"), ref)
    src = t / "scene.mp4"
    media.ffmpeg(["-f", "lavfi", "-i", "color=c=0xa0a0a0:s=320x180:r=30,noise=c0s=20:c0f=t[g];"
                  "color=c=0xc8a060:s=40x30:r=30[a];[g][a]overlay=140:75:shortest=1", "-t", 0.3,
                  "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", src])
    job = Job(name="look", sources={"s": {"path": str(src)}, "r": {"path": str(t / "ref.png")}}, output=str(t / "look.mkv"),
              profile="archive", steps=[{"id": "l", "op": "look_match", "inputs": ["s", "r"], "params": {"vignette": 0}}])
    r = jobs.run(job, ws=t / "ws", log=lambda m: None)
    f = list(media.read_frames(r["output"], 320, 180))[-1]
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(int) - [0, 128, 128]
    scene, subj = lab[10:60, 10:120].reshape(-1, 3).mean(0), lab[82:98, 150:170].reshape(-1, 3).mean(0)
    assert r["ok"] and scene[1] > 20 and scene[2] > 15, scene  # grey became terracotta
    assert subj[2] > scene[2] + 8 and subj[0] > scene[0], (subj, scene)  # subject stays warmer + brighter
