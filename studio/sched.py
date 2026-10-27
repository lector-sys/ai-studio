"""GPU/VRAM scheduling. A no-op on CPU (the default); once a GPU provider is opted in it:
  - serializes GPU jobs across processes (a lock file in the workspace),
  - budgets model VRAM (STUDIO_VRAM_MB, default 12000) and evicts least-recently-used models,
  - refuses (instead of silently degrading) a model that can't fit the budget.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path

# Rough peak VRAM per worker (MB) incl. activations at the default inference size.
VRAM_MB = {"rembg:u2net_human_seg": 700, "rembg:u2net": 700, "rembg:isnet-general-use": 900,
           "rembg:birefnet-portrait": 2600, "rembg:birefnet-general": 2600, "rembg:sam": 1500,
           "onnx": 2000, "whisper:base": 500, "whisper:small": 1100, "whisper:medium": 2600,
           "whisper:large-v3": 4700}
_loaded: dict = {}  # name -> (mb, unload callback); insertion order = LRU order


def providers() -> list[str]:
    return os.environ.get("STUDIO_ORT_PROVIDERS", "CPUExecutionProvider").split(",")


def gpu_active() -> bool:
    return any(p != "CPUExecutionProvider" for p in providers())


def budget() -> int:
    return int(os.environ.get("STUDIO_VRAM_MB", 12000))


def acquire(name, unload):
    """Register a model about to load; evict older models until it fits the VRAM budget."""
    if not gpu_active():
        return
    kind = name if name in VRAM_MB else name.split(":")[0]
    need = VRAM_MB.get(kind, VRAM_MB["onnx"])
    if need > budget():
        raise RuntimeError(f"{name} needs ~{need} MB VRAM > STUDIO_VRAM_MB={budget()}; "
                           "use a smaller model or the CPU provider")
    _loaded.pop(name, None)
    while _loaded and sum(m for m, _ in _loaded.values()) + need > budget():
        _, (_, free) = next(iter(_loaded.items()))
        free()
        for k in [k for k, (_, f) in _loaded.items() if f is free]:  # one cache_clear frees all its models
            _loaded.pop(k)
    _loaded[name] = (need, unload)


def in_use() -> dict:
    return {k: m for k, (m, _) in _loaded.items()}


@contextmanager
def gpu_lock(ws: Path, force=False):
    """Exclusive cross-process GPU lock while a job runs (when a GPU provider is active, or `force`:
    a GPU generation worker runs even though the editor's own ONNX provider is CPU)."""
    if not (gpu_active() or force):
        yield
        return
    f = open(ws / "gpu.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            deadline = time.time() + float(os.environ.get("STUDIO_GPU_LOCK_TIMEOUT", 3600))
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # retries ~10 s, then raises
                    break
                except OSError:
                    if time.time() > deadline:
                        f.close()
                        raise RuntimeError(f"GPU busy: another job held {ws / 'gpu.lock'} past "
                                           "STUDIO_GPU_LOCK_TIMEOUT") from None
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        f.close()


if __name__ == "__main__":  # self-check: LRU eviction under a budget
    os.environ |= {"STUDIO_ORT_PROVIDERS": "DmlExecutionProvider,CPUExecutionProvider", "STUDIO_VRAM_MB": "3000"}
    freed = []
    acquire("rembg:u2net_human_seg", lambda: freed.append("a"))
    acquire("whisper:small", lambda: freed.append("b"))
    acquire("rembg:birefnet-portrait", lambda: freed.append("c"))
    assert freed == ["a", "b"] and list(in_use()) == ["rembg:birefnet-portrait"], (freed, in_use())
    try:
        acquire("whisper:large-v3", lambda: None)
        raise SystemExit("should have refused")
    except RuntimeError:
        pass
    print("sched ok")
