"""Accelerator discovery. Reports what exists; nothing GPU-side is enabled without an explicit opt-in."""
import os
import subprocess
import sys
from functools import lru_cache

from .media import tool


def _lines(args):
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", *args], capture_output=True, text=True)
    return r.stdout.splitlines()


def discover() -> dict:
    hwaccels = [x.strip() for x in _lines(["-hwaccels"])[1:] if x.strip()]
    encoders = [x.split()[1] for x in _lines(["-encoders"]) if x.startswith(" V") and
                any(k in x for k in ("_amf", "_nvenc", "_qsv", "_vaapi", "_mf"))]
    out = {"cpu_threads": os.cpu_count(), "ffmpeg": tool("ffmpeg"), "ffmpeg_hwaccels": hwaccels,
           "ffmpeg_hw_encoders": encoders, "gpus": []}
    try:
        import onnxruntime as ort
        out["onnxruntime"] = {"version": ort.__version__, "available": ort.get_available_providers()}
    except ImportError:
        out["onnxruntime"] = None
    try:
        import torch
        out["torch"] = {"version": torch.__version__, "cuda_or_rocm": torch.cuda.is_available(),
                        "hip": getattr(torch.version, "hip", None)}
    except ImportError:
        out["torch"] = None
    if sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.Status }"],
                           capture_output=True, text=True)
        out["gpus"] = [dict(zip(("name", "status"), x.split("|"))) for x in r.stdout.splitlines() if x.strip()]
    from .vision import ORT_PROVIDERS
    out["selected"] = {"onnx_providers": ORT_PROVIDERS,
                       "hw_encode": os.environ.get("STUDIO_HW_ENCODE") == "1",
                       "note": "CPU by default. Opt in: STUDIO_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider "
                               "(needs onnxruntime-directml) and STUDIO_HW_ENCODE=1 (AMF/NVENC, probed per render)."}
    return out


@lru_cache(1)
def ffmpeg_filters() -> frozenset:
    return frozenset(x.split()[1] for x in _lines(["-filters"]) if len(x.split()) > 2 and x.startswith(" "))


def capabilities() -> dict:
    """What each capability can run on this machine (routing table for the agent)."""
    import importlib.util
    import urllib.request

    from .agent import op_signature
    from .ops import OPS

    have = lambda m: importlib.util.find_spec(m) is not None  # noqa: E731
    f = ffmpeg_filters()
    comfy = os.environ.get("STUDIO_COMFYUI_URL", "http://127.0.0.1:8188")
    try:
        urllib.request.urlopen(comfy + "/system_stats", timeout=1)
        comfy_up = True
    except OSError:
        comfy_up = False
    return {
        "stt": {"faster-whisper": have("faster_whisper")},
        "segmentation": {"rembg": have("rembg"), "grabcut": True},
        "stabilize": {"vidstab": "vidstabdetect" in f, "deshake": "deshake" in f},
        "denoise": {"hqdn3d": "hqdn3d" in f, "nlmeans": "nlmeans" in f},
        "interpolate": {"minterpolate": "minterpolate" in f, "blend": "framerate" in f},
        "onnx_models": {"onnxruntime": have("onnxruntime")},
        "generators": {"procedural": "gradients" in f, "command": bool(os.environ.get("STUDIO_GEN_CMD")),
                       "comfyui": comfy_up and bool(os.environ.get("STUDIO_COMFYUI_WORKFLOW"))},
        "workers": _workers(),  # per worker: unavailable (reason) / configured / ready (reviewed smoke)
        "ops": {k: op_signature(k) for k in sorted(OPS)},
    }


def _workers() -> dict:
    from .workers import listing
    return listing()
