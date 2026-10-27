"""Local generation workers: validated configuration, fingerprints and readiness from real smoke evidence.

States: unavailable (something required is missing: concrete reason given) -> configured (everything present,
no matching reviewed smoke output) -> ready (a smoke output for this exact fingerprint passed technical checks
and a recorded content review). Listing never loads models or downloads anything."""
import json
import math
import os
import time
import urllib.parse
from hashlib import sha256 as _h
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import media

KINDS = {"image", "video", "voice", "music", "sfx"}
AUDIO_KINDS = {"voice", "music", "sfx"}
FIELDS = {"prompt", "text", "voice", "seed", "duration", "width", "height", "fps", "references", "mask", "lang", "speed"}


class Binding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str
    input: str


class Worker(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    backend: Literal["command", "comfyui"]
    kinds: list[str]
    about: str = ""
    argv: list[str] = Field(default_factory=list)  # command: argv list; {request} {out} placeholders
    url: str = "http://127.0.0.1:8188"
    workflow: str | None = None  # comfyui API-format workflow .json
    bindings: dict[str, Binding] = Field(default_factory=dict)  # request field -> node input
    uploads: dict[str, Binding] = Field(default_factory=dict)  # reference name -> node input (filename)
    output_node: str | None = None
    output_type: str | None = None  # images | videos | gifs | audio
    node_classes: list[str] = Field(default_factory=list)  # required ComfyUI node classes
    models: list[str] = Field(default_factory=list)  # files that must exist (hashed into the fingerprint)
    fields: list[str] = Field(default_factory=lambda: ["prompt", "seed"])  # request fields this worker honours
    limits: dict[str, float] = Field(default_factory=dict)  # max_seconds, max_width, max_height, max_chars
    device: Literal["cpu", "gpu"] = "cpu"
    timeout: float = Field(1800, gt=0, le=6 * 3600)
    version: str = "1"


def _catalog() -> dict:
    base = json.loads(files("studio").joinpath("workers.json").read_text(encoding="utf-8"))
    local = _workspace() / "workers.json"
    if local.exists():  # user configuration overrides packaged candidates by name
        base |= json.loads(local.read_text(encoding="utf-8"))
    return base


def _workspace() -> Path:
    from .jobs import workspace
    return workspace()


ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str) -> str:
    """Relative worker paths (workers/<name>/...) are relative to the studio checkout."""
    return str(ROOT / p) if p and not Path(p).is_absolute() and "{" not in p and "/" in p.replace("\\", "/") else p


def worker_config(name) -> dict:
    cat = _catalog()
    if name not in cat:
        raise ValueError(f"unknown worker {name!r}; have {sorted(cat)}")
    w = Worker.model_validate(cat[name] | {"name": name})
    w.argv = [_abs(a) for a in w.argv]
    w.models = [_abs(m) for m in w.models]
    w.workflow = _abs(w.workflow) if w.workflow else None
    bad = set(w.kinds) - KINDS
    if bad:
        raise ValueError(f"worker {name}: unsupported kinds {sorted(bad)}")
    unknown = set(w.fields) - FIELDS
    if unknown:
        raise ValueError(f"worker {name}: unknown request fields {sorted(unknown)}")
    if w.backend == "comfyui":
        host = urllib.parse.urlparse(w.url).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"worker {name}: ComfyUI url must be loopback (local only), got {w.url}")
        if not (w.workflow and w.output_node and w.models):
            raise ValueError(f"worker {name}: comfyui needs workflow, output_node and models (the checkpoint files "
                             "the workflow loads, so replacing one invalidates readiness)")
    elif not w.argv:
        raise ValueError(f"worker {name}: command needs argv")
    return w.model_dump()


def request_check(cfg: dict, req: dict) -> None:
    """Fail before any model loads: wrong kind, unsupported fields or limits."""
    if req["kind"] not in cfg["kinds"]:
        raise ValueError(f"worker {cfg['name']} does not generate {req['kind']} (kinds: {cfg['kinds']})")
    # Options the caller chose; a visual request's size/fps/duration are inherent, not options.
    optional = {"text", "voice", "references", "mask", "lang", "speed"} | ({"duration"} if req["kind"] in AUDIO_KINDS else set())
    given = {k for k, v in req.items() if v not in (None, {}, []) and k in optional}
    unsupported = given - set(cfg["fields"])
    if unsupported:
        raise ValueError(f"worker {cfg['name']} does not support {sorted(unsupported)}; supports {cfg['fields']}")
    if req["kind"] == "video":
        for field in ("fps", "duration"):
            value = float(req.get(field) or 0)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"worker {cfg['name']}: {field} must be finite and positive")
    lim = cfg["limits"]
    for key, field in (("max_seconds", "duration"), ("max_width", "width"), ("max_height", "height"),
                       ("max_fps", "fps")):
        if key in lim and req.get(field) and float(req[field]) > lim[key]:
            raise ValueError(f"worker {cfg['name']}: {field}={req[field]} over its tested limit {lim[key]}")
    if req["kind"] == "video":
        frames = max(1, round(float(req["duration"]) * float(req["fps"])))
        if frames > lim.get("max_frames", float("inf")):
            raise ValueError(f"worker {cfg['name']}: {frames} frames over its tested limit {lim['max_frames']}")
        step = int(lim.get("frame_step", 1))
        if step > 1 and (frames - 1) % step:
            raise ValueError(f"worker {cfg['name']}: requires {step}n+1 frames, got {frames}")
    if "max_chars" in lim and len(req.get("text") or req.get("prompt") or "") > lim["max_chars"]:
        raise ValueError(f"worker {cfg['name']}: text longer than {int(lim['max_chars'])} characters")
    for name in (req.get("references") or {}):
        if cfg["backend"] == "comfyui" and name not in cfg["uploads"]:
            raise ValueError(f"worker {cfg['name']}: no upload binding for reference {name!r}")


def _paths(cfg) -> list[str]:
    out = list(cfg["models"]) + ([cfg["workflow"]] if cfg["workflow"] else [])
    if cfg["backend"] == "command":
        out += [a for a in cfg["argv"] if "{" not in a and Path(a).suffix and Path(a).is_file()]
    return out


def _file_id(p) -> str:
    p = Path(p)
    if p.is_dir():  # a model bundle folder: every file under it, by relative path
        return _h(json.dumps(sorted((str(f.relative_to(p)), _file_id(f)) for f in p.rglob("*") if f.is_file()))
                  .encode()).hexdigest()
    st = p.stat()
    # ponytail: multi-GB model files are identified by size+mtime, small files by content
    return media.sha256(p) if st.st_size < 64 << 20 else f"{st.st_size}:{st.st_mtime_ns}"


def missing(cfg) -> list[str]:
    import shutil
    need = list(cfg["models"]) + ([cfg["workflow"]] if cfg["workflow"] else [])
    need += [a for a in cfg["argv"][1:] if "{" not in a and Path(a).is_absolute()]
    gone = [p for p in need if not Path(p).exists()]
    if cfg["argv"] and not (Path(cfg["argv"][0]).is_file() or shutil.which(cfg["argv"][0])):
        gone.insert(0, cfg["argv"][0])
    return gone


def fingerprint(cfg) -> str:
    """Config + contents/identity of every file it depends on: changing any of them invalidates readiness."""
    ids = {p: _file_id(p) for p in _paths(cfg) if Path(p).exists()}
    return _h(json.dumps([cfg, ids], sort_keys=True).encode()).hexdigest()[:20]


def _evidence_file(name) -> Path:
    d = _workspace() / "workers"
    d.mkdir(exist_ok=True)
    return d / f"{name}.evidence.json"


def _server_up(cfg) -> str | None:
    import urllib.request
    try:
        urllib.request.urlopen(cfg["url"].rstrip("/") + "/system_stats", timeout=1)
        return None
    except OSError as e:
        return f"ComfyUI not reachable at {cfg['url']} ({e.__class__.__name__})"


def readiness(cfg, evidence_file=None) -> dict:
    gone = missing(cfg)
    if gone:
        return {"state": "unavailable", "reason": f"missing: {gone}", "kinds": {}}
    if cfg["backend"] == "comfyui":
        why = _server_up(cfg)
        if why:
            return {"state": "unavailable", "reason": why, "kinds": {}}
    fp = fingerprint(cfg)
    ev = json.loads(Path(evidence_file or _evidence_file(cfg["name"])).read_text(encoding="utf-8")) \
        if Path(evidence_file or _evidence_file(cfg["name"])).exists() else {}
    kinds = {}
    for kind in cfg["kinds"]:
        e = ev.get(kind)
        ok = (e and e["fingerprint"] == fp and e["validation"]["ok"] and (e.get("review") or {}).get("status") == "pass"
              and Path(e["output"]).exists() and media.sha256(e["output"]) == e["sha256"])
        kinds[kind] = {"state": "ready" if ok else "configured",
                       "smoke": {k: e[k] for k in ("output", "device", "seconds", "native", "date") if k in e} if ok else None,
                       "why": None if ok else ("no smoke output" if not e else "smoke evidence stale or unreviewed")}
    state = "ready" if kinds and all(k["state"] == "ready" for k in kinds.values()) else "configured"
    return {"state": state, "fingerprint": fp, "kinds": kinds, "device": cfg["device"]}


def record_smoke(cfg, result: dict, evidence_file=None) -> dict:
    """Store a smoke output's technical evidence (review pending)."""
    f = Path(evidence_file or _evidence_file(cfg["name"]))
    ev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    ev[result["kind"]] = {"fingerprint": fingerprint(cfg), "output": result["output"], "sha256": media.sha256(result["output"]),
                          "validation": result["validation"], "device": result.get("device"), "seconds": result["seconds"],
                          "native": result.get("native"), "date": time.strftime("%Y-%m-%d"), "review": None}
    f.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev[result["kind"]]


def record_review(cfg, kind, reviewer, status, findings, evidence_file=None) -> dict:
    f = Path(evidence_file or _evidence_file(cfg["name"]))
    ev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if kind not in ev:
        raise ValueError(f"no smoke output recorded for {cfg['name']} {kind}; run `studio workers smoke` first")
    if status not in ("pass", "fail", "needs-work"):
        raise ValueError("status must be pass | fail | needs-work")
    ev[kind]["review"] = {"reviewer": reviewer, "status": status, "findings": findings, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    f.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev[kind]


def listing() -> dict:
    out = {}
    try:
        names = list(_catalog())
    except (ValueError, OSError) as e:
        return {"_config": {"state": "unavailable", "reason": f"workers.json unreadable: {e}"}}
    for name in names:
        try:
            cfg = worker_config(name)
            out[name] = {"backend": cfg["backend"], "generates": cfg["kinds"], "about": cfg["about"]} | readiness(cfg)
        except (ValueError, OSError) as e:
            out[name] = {"state": "unavailable", "reason": str(e)}
    return out


def gpu_workers_in(steps) -> bool:
    for st in steps:
        w = (st.params or {}).get("worker")
        if w:
            try:
                if worker_config(w)["device"] == "gpu":
                    return True
            except ValueError:
                pass
    return False


if os.name == "nt":  # hidden child consoles for worker processes
    import subprocess
    NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    NO_WINDOW = 0
