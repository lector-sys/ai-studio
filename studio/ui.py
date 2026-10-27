"""Optional local UI (Phase 7). The CLI stays the source of truth: every button calls the same
functions the `studio` commands call. Stdlib http.server only, bound to 127.0.0.1.

Security: per-launch random token on every API call (other websites can't read it), Host header
pinned to localhost (blocks DNS rebinding), and files are served only from the workspace or from
paths named in manifests this UI loaded.
"""
import base64
import hashlib
import json
import mimetypes
import secrets
import subprocess
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import jobs
from .media import tool

ROOT = Path(__file__).resolve().parents[1]
TOKEN = secrets.token_urlsafe(24)
TASKS: dict = {}
ALLOWED: set = set()
MEDIA = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".wav", ".mp3", ".png", ".jpg", ".jpeg", ".webp", ".srt", ".vtt"}


def _allow_job(job):
    for s in job.sources.values():
        ALLOWED.add(str(Path(s.path).resolve()))
    o = Path(job.output).resolve()
    for p in (o, o.with_name(o.stem + ".preview.mp4"), o.with_suffix(".mov"), o.with_suffix(".mkv"), o.with_suffix(".wav")):
        ALLOWED.add(str(p))


def _servable(p: Path) -> bool:
    p = p.resolve()
    ws = jobs.workspace().resolve()
    return p.suffix.lower() in MEDIA | {".json"} and (str(p) in ALLOWED or ws in p.parents)


def _manifests() -> list[dict]:
    ws = jobs.workspace()
    files = sorted(ws.glob("manifests/*.json"), key=lambda f: -f.stat().st_mtime)[:40] + sorted(ROOT.glob("examples/*.json"))
    return [{"path": str(f), "name": f.name, "example": f.parent.name == "examples"} for f in files]


def _logs() -> list[dict]:
    out = []
    for f in sorted(jobs.workspace().glob("logs/*.json"), key=lambda f: -f.stat().st_mtime)[:30]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({"path": str(f), "name": f.name, "ok": d.get("ok"), "output": d.get("output"), "mode": d.get("mode")})
            ALLOWED.add(str(Path(d.get("output", "")).resolve()))
        except (OSError, ValueError):
            pass
    return out


def _task(kind, fn):
    tid = secrets.token_hex(6)
    t = TASKS[tid] = {"id": tid, "kind": kind, "status": "running", "log": [], "result": None}

    def go():
        try:
            t["result"] = fn(lambda m: t["log"].append(str(m)))
            t["status"] = "done"
        except Exception as e:  # noqa: BLE001 - surfaced to the UI
            t["log"].append(traceback.format_exc(limit=3))
            t["result"], t["status"] = {"ok": False, "error": str(e)}, "error"

    threading.Thread(target=go, daemon=True).start()
    return {"task": tid}


def _run(body):
    from . import agent, speech
    mode = body.get("mode", "preview")
    if mode == "transcribe":
        path = body["path"]
        if not _servable(Path(path)):
            raise PermissionError("not a known source")
        return _task("transcribe", lambda log: speech.transcribe(path, body.get("model", "base"),
                                                                 cache_dir=jobs.workspace() / "cache"))
    job = jobs.Job.model_validate(body["manifest"])
    _allow_job(job)
    if mode == "check":
        return agent.check(job)
    if mode == "auto":
        return _task("auto", lambda log: agent.auto(job, int(body.get("rounds", 3)), log))
    if mode not in ("preview", "final"):
        raise ValueError(f"unknown mode {mode}")
    return _task(mode, lambda log: jobs.run(job, mode=mode, log=log))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _guard(self, q) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self._send(403, {"error": "bad host"})
            return False
        if self.path.split("?")[0] != "/" and self.headers.get("X-Studio-Token") != TOKEN and q.get("t", [""])[0] != TOKEN:
            self._send(403, {"error": "bad token"})
            return False
        return True

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._guard(q):
            return
        try:
            if u.path == "/":
                html = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
                return self._send(200, html.encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                                              "script-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self'"})
            if u.path == "/api/list":
                return self._send(200, {"manifests": _manifests(), "logs": _logs(),
                                        "workspace": str(jobs.workspace())})
            if u.path == "/api/ops":
                from .ops import OPS
                return self._send(200, {k: " ".join((f.__doc__ or "").split()) for k, f in sorted(OPS.items())})
            if u.path == "/api/task":
                t = TASKS.get(q.get("id", [""])[0])
                return self._send(200 if t else 404, t or {"error": "no such task"})
            path = Path(q.get("path", [""])[0])
            if u.path in ("/api/json", "/api/file", "/api/frame", "/api/info") and not _servable(path):
                return self._send(403, {"error": f"not allowed: {path}"})
            if u.path == "/api/json":
                d = json.loads(path.read_text(encoding="utf-8"))
                try:
                    _allow_job(jobs.Job.model_validate(d))
                except ValueError:
                    pass
                return self._send(200, d)
            if u.path == "/api/info":
                from .media import info
                return self._send(200, info(path))
            if u.path == "/api/frame":
                r = subprocess.run([tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", str(float(q.get("time", ["0"])[0])),
                                    "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
                                   capture_output=True)
                return self._send(200 if r.returncode == 0 else 500, r.stdout or b"", "image/jpeg")
            if u.path == "/api/file":
                return self._file(path)
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def _file(self, path: Path):
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            start, end = (int(a), int(b) if b else size - 1) if a else (size - int(b), size - 1)
            end = min(end, size - 1)
        n = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(n))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            while n > 0 and (chunk := f.read(min(n, 1 << 20))):
                try:
                    self.wfile.write(chunk)
                except (ConnectionError, OSError):
                    return  # browser cancelled the range request
                n -= len(chunk)

    def do_POST(self):
        u = urlparse(self.path)
        if not self._guard({}):
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/run":
                return self._send(200, _run(body))
            if u.path == "/api/paint":  # mask-fix strokes from the paint tool -> workspace PNG
                raw = base64.b64decode(body["png"].split(",", 1)[-1])
                if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) > 20 << 20:
                    raise ValueError("paint must be a PNG under 20 MB")
                f = jobs.workspace() / "paint" / f"paint-{hashlib.sha256(raw).hexdigest()[:16]}.png"
                f.parent.mkdir(exist_ok=True)
                f.write_bytes(raw)
                ALLOWED.add(str(f.resolve()))
                return self._send(200, {"path": str(f)})
            if u.path == "/api/save":
                p = Path(body["path"])
                if p.suffix.lower() != ".json":
                    raise ValueError("manifests must be saved as .json")
                job = jobs.Job.model_validate(body["manifest"])  # only valid manifests get written
                p.write_text(json.dumps(body["manifest"], indent=2), encoding="utf-8")
                _allow_job(job)
                ALLOWED.add(str(p.resolve()))
                return self._send(200, {"saved": str(p)})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})


def serve(port=8765, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(json.dumps({"ui": url, "note": "Ctrl+C to stop"}), flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
