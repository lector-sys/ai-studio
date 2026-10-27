"""Intelligent editorial: local STT (faster-whisper), captions, silence/filler detection,
transcript-driven cuts and a semantically searchable media index (MiniLM sentence embeddings).

Transcripts are cached by audio content hash + model, so captions, cuts and search reuse one pass.
"""
import json
import re
import subprocess
from functools import lru_cache
from hashlib import sha256 as _h
from pathlib import Path

import numpy as np

from .media import ffmpeg, fpath, info, sha256, tool
from .ops import has_audio, op

FILLERS = {"um", "uh", "erm", "er", "ah", "uhm", "hmm", "mm", "mhm"}


@lru_cache(2)
def _model(name):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8")  # ponytail: CPU only; GPU via ctranslate2 CUDA builds


def audio16k(path) -> np.ndarray:
    """Mono 16 kHz float32 PCM via our ffmpeg (avoids faster-whisper's PyAV decoder)."""
    r = subprocess.run([tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-ac", "1",
                        "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"audio decode failed: {r.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(r.stdout, np.float32)


def transcribe(path, model="base", language=None, task="transcribe", cache_dir=None) -> dict:
    """Word-timestamped transcript: {language, model, duration, segments: [{start, end, text, words}]}.
    task=translate gives English text (Whisper's built-in translation adapter)."""
    if not has_audio(path):
        raise ValueError(f"{path} has no audio stream to transcribe")
    key = _h(json.dumps([sha256(path), model, language, task]).encode()).hexdigest()[:20]
    cf = Path(cache_dir) / f"transcript-{key}.json" if cache_dir else None
    if cf and cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    import faster_whisper
    a = audio16k(path)
    segs, inf = _model(model).transcribe(a, language=language, task=task, word_timestamps=True, vad_filter=False)
    out = {"language": inf.language, "model": f"faster-whisper {faster_whisper.__version__}:{model}", "task": task,
           "duration": round(len(a) / 16000, 3), "segments": [
               {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                "words": [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                           "p": round(w.probability, 3)} for w in (s.words or [])]} for s in segs]}
    if cf:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def words(t) -> list[dict]:
    return [w for s in t["segments"] for w in s["words"]]


def _ts(t, sep):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}{sep}{ms % 1000:03}"


def cues(t, max_chars=42, max_dur=5.0, gap=0.6) -> list[tuple]:
    """Group words into caption cues (start, end, text) by length, duration and pauses."""
    out, cur = [], []
    for w in words(t):
        if cur and (len(" ".join(x["w"] for x in cur + [w])) > max_chars or w["end"] - cur[0]["start"] > max_dur
                    or w["start"] - cur[-1]["end"] > gap):
            out.append(cur)
            cur = []
        cur.append(w)
        if w["w"].endswith((".", "?", "!")):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [(c[0]["start"], c[-1]["end"], " ".join(x["w"] for x in c)) for c in out]


def to_srt(t, **kw) -> str:
    return "\n".join(f"{i}\n{_ts(a, ',')} --> {_ts(b, ',')}\n{txt}\n" for i, (a, b, txt) in enumerate(cues(t, **kw), 1))


def to_vtt(t, **kw) -> str:
    return "WEBVTT\n\n" + "\n".join(f"{_ts(a, '.')} --> {_ts(b, '.')}\n{txt}\n" for a, b, txt in cues(t, **kw))


def silences(path, noise=-35, min_dur=0.5) -> list[list[float]]:
    """[start, end] of silent stretches (ffmpeg silencedetect)."""
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:a:0", "-af",
                        f"silencedetect=n={noise}dB:d={min_dur}", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    dur = info(path)["duration"]
    return [[max(0.0, s), ends[i] if i < len(ends) else dur] for i, s in enumerate(starts)]


def _norm(s):
    return re.sub(r"[^\w']", "", s.lower())


def filler_ranges(t, fillers=FILLERS) -> list[list[float]]:
    return [[w["start"], w["end"]] for w in words(t) if _norm(w["w"]) in fillers]


def phrase_ranges(t, phrase) -> list[list[float]]:
    """Time ranges of every occurrence of a phrase (case/punctuation-insensitive word match)."""
    ws, q = words(t), [_norm(x) for x in phrase.split() if _norm(x)]
    toks = [_norm(w["w"]) for w in ws]
    return [[ws[i]["start"], ws[i + len(q) - 1]["end"]] for i in range(len(ws) - len(q) + 1) if toks[i:i + len(q)] == q]


def merge(ranges, pad=0.0, total=None) -> list[list[float]]:
    rs = sorted([max(0.0, a - pad), b + pad if total is None else min(total, b + pad)] for a, b in ranges if b > a)
    out = []
    for a, b in rs:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def keep_from_remove(remove, total) -> list[list[float]]:
    keep, t = [], 0.0
    for a, b in remove:
        if a > t:
            keep.append([t, a])
        t = max(t, b)
    if t < total:
        keep.append([t, total])
    return [k for k in keep if k[1] - k[0] > 1e-3]


@op("cut")
def cut(ctx, ins, p, out):
    """Remove time ranges (ripple delete) - transcript/silence driven editing. params:
    remove [[start, end], ...] (s) | keep [[start, end], ...]; silence (true or {noise dB -35, min s 0.5});
    fillers (true or list of words); phrases (list of transcript phrases to remove); pad (s, 0.05: kept
    around speech); model (whisper size, base); language."""
    total = info(ins[0])["duration"]
    if p.get("keep"):
        keep = merge(p["keep"], total=total)
        removed = keep_from_remove(keep, total)
    else:
        rm = [list(map(float, r)) for r in p.get("remove", [])]
        pad = float(p.get("pad", 0.05))
        if p.get("silence"):
            s = p["silence"] if isinstance(p["silence"], dict) else {}
            # shrink silences by pad so word edges aren't clipped
            rm += [[a + pad, b - pad] for a, b in silences(ins[0], s.get("noise", -35), s.get("min", 0.5))]
        if p.get("fillers") or p.get("phrases"):
            t = transcribe(ins[0], p.get("model", "base"), p.get("language"), cache_dir=ctx.workdir)
            if p.get("fillers"):
                rm += filler_ranges(t, set(p["fillers"]) if isinstance(p["fillers"], list) else FILLERS)
            for ph in p.get("phrases", []):
                rm += phrase_ranges(t, ph)
            ctx.meta["transcript_model"] = t["model"]
        removed = merge(rm, total=total)
        keep = keep_from_remove(removed, total)
    if not keep:
        raise ValueError("cut would remove the whole clip")
    ctx.meta["cut"] = {"removed": [[round(a, 3), round(b, 3)] for a, b in removed],
                       "removed_seconds": round(sum(b - a for a, b in removed), 3),
                       "kept_seconds": round(sum(b - a for a, b in keep), 3)}
    expr = "+".join(f"gte(t,{a:.4f})*lt(t,{b:.4f})" for a, b in keep)  # half-open: no doubled boundary frames
    fc = f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[v]"
    a = None
    if has_audio(ins[0]):
        fc += f";[0:a]aselect='{expr}',asetpts=N/SR/TB[a]"
        a = "[a]"
    # ponytail: hard cuts at frame boundaries; add micro-crossfades if pops are audible
    ctx.encode(["-i", ins[0]], fc, "[v]", a, out)


@op("captions")
def captions(ctx, ins, p, out):
    """Transcribe and burn in captions. params: model (base), language, task (transcribe|translate),
    max_chars (42), style (ASS force_style, default "Fontsize=22,Outline=2,MarginV=40")."""
    t = transcribe(ins[0], p.get("model", "base"), p.get("language"), p.get("task", "transcribe"), ctx.workdir)
    srt = ctx.workdir / f"captions-{_h(json.dumps(t).encode()).hexdigest()[:16]}.srt"
    srt.write_text(to_srt(t, max_chars=int(p.get("max_chars", 42))), encoding="utf-8")
    style = p.get("style", "Fontsize=22,Outline=2,MarginV=40")
    ctx.meta["captions"] = {"model": t["model"], "language": t["language"], "cues": len(cues(t)), "srt": str(srt)}
    ctx.encode(["-i", ins[0]], f"[0:v]subtitles={fpath(srt)}:force_style='{style}'[v]", "[v]", "0:a?", out)


# --- media index / search -------------------------------------------------------------------------

MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm", ".mxf", ".mts", ".wav", ".mp3", ".m4a", ".flac"}


@lru_cache(1)
def _embedder():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from . import models
    sess = ort.InferenceSession(models.path("minilm"), providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(models.sibling("minilm", "tokenizer.json"))
    tok.enable_truncation(256)
    tok.enable_padding()
    return sess, tok


def embed(texts) -> np.ndarray:
    """L2-normalized sentence embeddings (all-MiniLM-L6-v2, mean-pooled), one row per text."""
    sess, tok = _embedder()
    out = []
    for i in range(0, len(texts), 64):
        enc = tok.encode_batch(list(texts[i:i + 64]))
        ids = np.array([e.ids for e in enc], np.int64)
        mask = np.array([e.attention_mask for e in enc], np.int64)
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        h = sess.run(None, {k: v for k, v in feed.items() if k in {x.name for x in sess.get_inputs()}})[0]
        v = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def scenes(path, threshold=0.3) -> list[float]:
    r = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-vf",
                        f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [0.0] + [round(float(x), 3) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]


def index(folder, ws, model="base", transcribe_audio=True, log=print) -> dict:
    """Index every media file under folder: metadata, scene cuts + thumbnails, transcript."""
    idx_file = ws / "index" / "index.json"
    idx_file.parent.mkdir(parents=True, exist_ok=True)
    idx = json.loads(idx_file.read_text(encoding="utf-8")) if idx_file.exists() else {}
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in MEDIA_EXTS or ws in f.resolve().parents:
            continue
        digest = sha256(f)
        old = idx.get(digest, {})
        if old.get("path") == str(f.resolve()):
            if old.get("segments") and not old.get("emb"):  # indexed before semantic search existed
                old["emb"] = _save_emb(ws, digest, [s["text"] for s in old["segments"]])
            continue
        log(f"indexing {f}")
        d = info(f)
        e = {"path": str(f.resolve()), "duration": d["duration"],
             "video": d["video"] and {k: d["video"][k] for k in ("width", "height", "fps")}, "scenes": [], "thumbs": []}
        if d["video"]:
            e["scenes"] = scenes(f)
            for i, t in enumerate(e["scenes"][:50]):
                th = ws / "index" / f"{digest[:16]}-{i:03}.jpg"
                ffmpeg(["-ss", t, "-i", f, "-frames:v", 1, "-vf", "scale=320:-2", th])
                e["thumbs"].append(str(th))
        if d["audio"] and transcribe_audio:
            t = transcribe(f, model, cache_dir=ws / "cache")
            e["language"], e["segments"] = t["language"], [{k: s[k] for k in ("start", "end", "text")}
                                                           for s in t["segments"]]
            e["emb"] = _save_emb(ws, digest, [s["text"] for s in e["segments"]])
        idx[digest] = e
    idx_file.write_text(json.dumps(idx, indent=1), encoding="utf-8")
    return {"file": str(idx_file), "items": len(idx)}


def _save_emb(ws, digest, texts):
    """Segment embeddings next to the index (.npy, rows = segments); None when the model is unavailable."""
    if not texts:
        return None
    try:
        v = embed(texts)
    except Exception:  # noqa: BLE001 - offline / no model: search falls back to word overlap
        return None
    f = ws / "index" / f"{digest[:16]}-emb.npy"
    np.save(f, v.astype(np.float16))
    return str(f)


def search(query, ws, limit=20, min_score=0.3) -> list[dict]:
    """Rank transcript segments by meaning (cosine of sentence embeddings) blended with word overlap,
    plus file-name matches. Falls back to word overlap alone when no embeddings exist.
    ponytail: transcripts only; add CLIP frame embeddings to search what's on screen."""
    idx_file = ws / "index" / "index.json"
    if not idx_file.exists():
        raise FileNotFoundError("no index yet: run `studio index <folder>` first")
    q = {_norm(x) for x in query.split()} - {""}
    entries = json.loads(idx_file.read_text(encoding="utf-8")).values()
    qv = None
    if any(e.get("emb") for e in entries):
        try:
            qv = embed([query])[0]
        except Exception:  # noqa: BLE001 - model unavailable: lexical only
            qv = None
    hits = []
    for e in entries:
        name = {_norm(x) for x in re.split(r"[\W_]+", Path(e["path"]).stem)}
        if q & name:
            hits.append({"score": round(len(q & name) / len(q), 3), "path": e["path"], "start": 0.0,
                         "text": Path(e["path"]).name, "match": "name"})
        segs = e.get("segments", [])
        sims = (np.load(e["emb"]).astype(np.float32) @ qv if qv is not None and e.get("emb") and Path(e["emb"]).exists()
                else None)
        for i, s in enumerate(segs):
            lex = len(q & {_norm(x) for x in s["text"].split()}) / len(q) if q else 0.0
            score = 0.7 * float(sims[i]) + 0.3 * lex if sims is not None else lex
            if score >= (min_score if sims is not None else 1e-9):
                hits.append({"score": round(score, 3), "path": e["path"], "start": s["start"], "end": s["end"],
                             "text": s["text"], "match": "semantic" if sims is not None else "words"})
    return sorted(hits, key=lambda h: -h["score"])[:limit]
