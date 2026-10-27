# AI Studio: local AI video studio

Built from `../Local_AI_Video_Studio_Architecture_v1.docx`. All seven phases are built: foundation, the deterministic
editor, the AI rotoscope, intelligent editorial (STT), restoration & motion, generative editing, agent reliability, and
an optional local UI. FFmpeg is the media backbone and the AI models are
swappable workers. The coding agent drives everything through one CLI contract (see `AGENTS.md`).

## Setup
```
winget install Gyan.FFmpeg            # already installed (9.0.2)
py -3.12 -m venv .venv && .venv\Scripts\pip install -e .[test]
.venv\Scripts\studio devices          # what accelerators exist
.venv\Scripts\python -m pytest -q tests
```
Models download on first use: rembg into `~/.rembg/models` (u2net_human_seg is 176 MB, SAM is about 16 MB + 90 MB),
and Whisper into the Hugging Face cache (`base` is about 145 MB).

## Commands
| Command | What it does |
|---|---|
| `studio inspect in.mp4 [--keyframes] [--hash]` | streams, size, fps/VFR, duration, color/HDR, audio, rotation |
| `studio edit <op> in [more…] -o out -p k=v` | one deterministic op (`studio ops` lists all 21 with params) |
| `studio color …` / `studio audio …` | shortcuts for `edit color` / `edit audio` |
| `studio mask person\|salient\|object in -o m.mkv [--box x,y,w,h \| --point x,y]` | matte video + quality stats; a box with person/salient isolates that one subject (identity-tracked) |
| `studio track in --box x,y,w,h` | optical-flow box track (per-frame JSON) |
| `studio background remove\|blur\|replace in [--bg img/vid] [--mask m.mkv] [--target object --box …]` | full background workflow |
| `studio recipes` / `studio recipe <name> --request req.json -o job.json` | creative recipe catalog / expand a request into a checked manifest (cards rendered, nothing else) |
| `studio preview job.json` / `studio render job.json [--profile p]` | proxy preview / master render of a manifest |
| `studio review <render log> -o <dir>` / `studio review-record <dir> <kind> --reviewer … --status pass` | shared review bundle (frames, 100% crops, checks, SDR viewing copy, audio) / record a delegated or human review |
| `studio validate out [--against src] [--duration s] [--ssim]` | technical QA |
| `studio profiles` | 4k60-master, native-hdr-master, h264-master, av1-master, prores-hq, alpha-master, archive, preview, social-vertical |
| `studio transcribe in [-o t.json] [--model base] [--translate]` | local STT with word timestamps (faster-whisper, CPU int8) |
| `studio captions in -o out.srt / .vtt / .json / video.mp4` | sidecar captions, or burned in when `-o` is a video |
| `studio cut in -o out [--remove 1-2,5-6] [--silence] [--fillers] [--phrase "..."]` | ripple-delete ranges, silences, fillers, transcript phrases |
| `studio index <folder>` / `studio search "words"` | media index (scenes, thumbnails, transcripts + embeddings) and semantic, timecoded search |
| `studio object remove/replace/insert in --box x,y,w,h [--with img] -o out` | tracked object removal (clean plate), replacement, insert |
| `studio generate image/video --prompt "..." -o out` | generator adapters: procedural, command, comfyui |
| `studio capabilities` / `studio schema` | routing table (what backends exist here) / manifest JSON schema |
| `studio check job.json` | dry run: graph errors, unknown params, models, VRAM/disk estimates |
| `studio auto job.json` | preview, then refine low-confidence masks (stride, model ladder), then render and validate |
| `studio bench [--size 1920x1080] [--ai]` | per-op timings, saved to `workspace/bench/` |
| `studio models [download name…]` | model registry: YuNet, GFPGAN, Real-ESRGAN x2/x4, RIFE 4.9, MiniLM (download on first use) |
| `studio cache prune [--older-than 7]` | delete cached step outputs/proxies unused for N days (or everything) |
| `studio ui` | local web UI on 127.0.0.1:8765 |

Ops (42): compose (timed image/text cards, one pass, source picture/clock/HDR kept), trim, crop, resize (fit/fill/stretch), rotate/flip, speed, reverse, fps, freeze, concat, transition (any
xfade), fade, overlay (PiP/logo/watermark), text (titles/lower thirds), subtitles (burn-in), audio
(replace/mix/duck/mute), loudnorm, audio_clean, color (eq/temperature/LUT), chromakey, mask, mask_edit (paint fixes), background ·
**editorial** cut, captions · **restoration** stabilize (vidstab/deshake), denoise (hqdn3d/nlmeans/onnx),
sharpen (cas/unsharp), deinterlace, lens, upscale (lanczos/Real-ESRGAN), interpolate (minterpolate/blend/dup/RIFE),
face_restore (YuNet + GFPGAN), match_color (shot matching), look_match (regrade into a reference image's look) · **generative** inpaint (clean plate/telea/ns), insert (tracked), relight (masked grade),
recolor (change one region's colour: ground / mask / hue + saturation selection), stylize (recipes/onnx), extend (canvas outpaint-by-blur / hold / push-in), generate.

## How it works
- **Job manifest** (`examples/background_replace.json`): sources (with sha256), steps (op, inputs, params),
  output, profile, proxy height, retries, and validation rules. The runner fills in hashes and saves a copy
  to `workspace/manifests/`.
- **Step cache / resume**: each step's output is keyed on its op, params, input hashes, and the worker source code.
  Outputs are stored as lossless intermediates (x264 qp0 / FFV1 + FLAC) in `workspace/cache/`. A failed or
  interrupted job re-runs only the unfinished steps, and editing an op's code invalidates its old outputs.
- **4K60 strategy**: masks are inferred at 512 px and stored as lossless gray video at the source fps. The
  `background` op upsamples them at master resolution with an edge-aware guided filter, guided by the master
  luma, then composites. The master is never downscaled for the model.
- **Preview**: sources are swapped for cached 540p proxies, and pixel params are rescaled automatically.
- **QA** after every render: clean decode, frame count (catches dropped frames), resolution, fps, duration,
  A/V sync, and optional SSIM.
- **Provenance**: `workspace/logs/<job>-<time>.json` records every ffmpeg command, hash, timing, model/backend
  version, and mask stats.
- **Devices**: discovery reports AMF/NVENC/QSV encoders, ONNX providers, torch, and GPUs. Everything runs on CPU
  by default. GPU is opt-in with `STUDIO_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider` (needs
  `onnxruntime-directml`) and `STUDIO_HW_ENCODE=1` (the encoder is probed before use). `STUDIO_WORKSPACE`
  moves the workspace.

## Phase notes
- **Editorial**: audio is decoded by our ffmpeg into 16 kHz PCM, then handed to faster-whisper (its bundled PyAV
  decoder clashes with the installed `av`). Transcripts are cached by audio hash + model, so captions, cuts, and the
  index share one pass. `cut` uses half-open ranges, so no frames get doubled at the boundaries.
- **Restoration**: FFmpeg backends are the defaults. `backend=onnx` + `model=<.onnx>` runs any NCHW RGB 0-1
  image-to-image net (Real-ESRGAN-class SR, denoise, style), tiled with overlap. Upscale/interpolate exist for the 4K
  strategy: generate or repair at low resolution, then SR/VFI only that segment back up to the master.
- **Generative**: object removal builds a clean plate (per-pixel median of unmasked frames) for static shots and
  inpaints the rest. `insert` follows the tracked box with `sendcmd`. Generators sit behind one function:
  - `procedural` is built in.
  - `command` is `STUDIO_GEN_CMD`, e.g. `"C:\\tools\\gen.exe" --prompt {prompt} --out {out} --w {width} --h {height}`
    (or a JSON argv list). Each field becomes one argv item and no shell is used.
  - `comfyui` uses `STUDIO_COMFYUI_URL` (default `http://127.0.0.1:8188`) plus `STUDIO_COMFYUI_WORKFLOW`, an
    API-format workflow with `{{prompt}}`, `{{width}}`, `{{height}}`, `{{seed}}`, `{{frames}}`, `{{fps}}` placeholders.
  - `STUDIO_GEN_BACKEND` sets the default.
- **Reliability**: `sched.py` is a no-op on CPU. With a GPU provider opted in, it holds a cross-process GPU lock per
  job, keeps a VRAM budget (`STUDIO_VRAM_MB`, default 12000) with LRU model eviction, and refuses a model that can't
  fit (it never silently downgrades). `auto` is the spec's acceptance test 7. Saved manifests record absolute paths.
- **UI**: stdlib server bound to 127.0.0.1. It uses a per-launch token, pins the Host header (no DNS rebinding), and
  serves only workspace files or files named in loaded manifests. Features: manifests, check/preview/render/auto,
  a frame box picker that writes `params.box` in source px, transcript to cut step, validation chips.

## Creative recipes and compose
- Recipes (`studio/recipes.json` + `recipes.py`): image-popup, lower-third, kinetic-captions, comparison,
  clip-transitions, look, narration-mix. A request (see `examples/creative_benchmark/request.json`) expands to an
  ordinary manifest; text is drawn with Pillow from literal strings (never FFmpeg expressions), cards are cached by
  content. Overlong text shrinks to 60% then fails (no silent truncation).
- `compose`: cards are converted into the source's colour space once (HLG/PQ via explicit zscale, white =
  `graphics_white_nits`, default 203) and repeated inside the filter graph, so timing is deterministic and the
  base is never scaled, graded or re-timed. Untouched pixels are bit-identical in the lossless intermediate.
  Cue ranges are half-open `[start, end)`; card frames run on a grid anchored at the cue start (≤1 frame offset
  from base frames on VFR/phone clocks).
- `native-hdr-master`: HEVC Main10, AMF constant QP 10/12 with `STUDIO_HW_ENCODE=1` (else x265 CRF 12); unchanged
  timelines keep the source time base exactly (`-enc_time_base`).
- Safety: final renders publish atomically and never replace an existing file (an identical completed request
  returns its verified result); cache keys hash the files behind params (fonts, LUTs, subtitles, models).
- Audio cleanup (`audio_clean`, compose `audio`) measures the filter chain's delay and removes it (1199 samples at
  48 kHz for the default chain); `qc.audio_alignment` reports silence/ambiguity as inconclusive, never as sync.
- Measured (62.5 s 4K HLG iPhone clip, ten popups, i7-14700K + RX 9060 XT AMF): see
  `D:\edited(studio)\Creative toolkit benchmark\BENCHMARK.md`.

## v0.3 notes
- **Plugin**: `.claude-plugin/marketplace.json` + `plugin/` make this repo a Claude Code marketplace. Installed with
  `claude plugin marketplace add D:\projects\AI_studio` and `claude plugin install ai-studio@ai-studio`; the `studio`
  skill then works from any folder and points the agent at `AGENTS.md`.
- **Models** (`models.py`): ops take `model=<registry name>` or a `.onnx` path. Names download once into the Hugging
  Face cache. All run on CPU by default. Rough CPU cost: GFPGAN ~1 s per face per frame, Real-ESRGAN x2 ~5 s per
  640 px frame, RIFE ~0.05 s per 256 px frame (scales with pixels). Use them on short or low-res segments.
- **Identity tracking**: with a `box`, each keyframe keeps the one connected subject that best matches the tracked box
  (overlap) and the subject's colour histogram. While it's hidden, the box coasts on its last velocity (up to 2 s),
  then re-acquires. `identity_unsure_ratio` feeds `low_confidence`.
- **Chunking**: `mask`, `denoise`, `sharpen`, `upscale`, `stylize` and `face_restore` steps on clips longer than
  2 × `STUDIO_CHUNK_SECONDS` (default 60) run in chunks with a 1 s lead-in that is trimmed off. Finished chunks
  survive an interruption. Per step: `"chunk": 30` (seconds) or `0` (never). Steps with a box/point never chunk.
- **HDR10**: mastering display + MaxCLL/MaxFALL are read from the source and written by the x265 and SVT-AV1 masters.
- **Cache**: keys hash the op's own module + shared helpers, so editing one op only invalidates that op. Cache hits
  refresh the entry's age; `studio cache prune --older-than N` frees the rest.
- **UI**: timeline strip (thumbnails, cut ranges, speech), mask overlay of any rendered mask step, and a paint tool
  (Paint + / Paint −) whose strokes become a `mask_edit` step after the selected mask step.

## Layout
`studio/media.py` ffmpeg/ffprobe I/O · `ops.py` deterministic ops · `vision.py` mask/track/background ·
`speech.py` STT/captions/cut/index · `restore.py` restoration + motion · `generate.py` generative ops + adapters ·
`sched.py` VRAM scheduler · `agent.py` check/auto/bench · `jobs.py` manifest, cache, profiles, provenance ·
`qc.py` validation + audio metrics/alignment · `compose.py` card compositing · `recipes.py` recipes ·
`review.py` review bundles · `devices.py` discovery/capabilities · `models.py` model registry · `cli.py` contract ·
`ui.py` + `ui.html` UI.

## Known limits
- Identity is colour-histogram + motion, not a learned re-ID model: two similar-looking subjects crossing can still
  swap (flagged by `low_confidence`), and a subject merged with another into one blob can't be split. Upgrade path:
  SAM2 video propagation behind the same `mask` op (needs torch and roughly 10 GB of RAM per 10 s of 4K on CPU;
  practical only with a GPU).
- Masks assume constant frame rate. VFR phone footage should be conformed first (`edit fps`); `check` warns about this.
- HDR: HDR10 static metadata is carried (x265, SVT-AV1); dynamic HDR10+/Dolby Vision metadata is not. Ops use the
  first audio track: iPhone spatial-audio (APAC) tracks have no FFmpeg decoder and are not carried.
- Local workers are configured for FLUX images, experimental reference edits and Wan clips, Kokoro narration,
  and Stable Audio music/SFX. Weights are machine-local, not shipped. Read [LOCAL_WORKERS](docs/LOCAL_WORKERS.md)
  for tested limits, GPU session handling and pending listening review. Neural edits can regenerate details;
  use the source-preserving deterministic workflow for faithful colour changes. SR/RIFE remain CPU, slow at 4K.
- Search covers what was said (transcripts) and file names, not what's on screen (CLIP frames would add that).
- `mask_edit` paint is static: it applies at a fixed screen position over its time range (add tracking if needed).
