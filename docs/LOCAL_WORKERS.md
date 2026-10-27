# Local generation workers

AI Studio runs no foundation model itself. Generation goes through **workers**, which are local programs or a local ComfyUI server. Each worker lives in its own isolated environment under `workers/<name>/`, and none of its dependencies are installed into the editor's `.venv`. Workers never fall back to the cloud. A neural request never silently becomes a procedural placeholder.

## Commands

```
studio workers                      # state of every worker: unavailable (+ exact reason) / configured / ready
studio workers smoke NAME --kind K --prompt "..." [--text ...] [--voice ...] [--duration s] [--reference init=img.png]
studio workers review NAME --kind K --reviewer WHO --status pass|fail|needs-work --finding "..."
studio generate voice|music|sfx|image|video --worker NAME --prompt "..." -o out   # never replaces a file
```

A job step can use a worker too:
- `generate_audio` with `{kind, worker, text|prompt, voice, duration, seed}` produces a WAV asset. Mix it into a clip with an `audio` step (`mode: mix`, `delay`).
- `generate` with `{worker, prompt, width, height, fps, duration}` produces a visual clip.

Every output has a `<file>.gen.json` file beside it. It records the request, worker, worker fingerprint, device, native size/rate/length, SHA-256 and elapsed time.

## Readiness

| State | Meaning |
|---|---|
| unavailable | A runtime, model file, workflow or server is missing (the listing names it) |
| configured | Everything required is present, but no reviewed smoke output matches the current fingerprint |
| ready | A smoke output for this exact fingerprint passed technical validation and a recorded content review |

The fingerprint covers:
- the worker's configuration;
- the contents of its small files (workflow, scripts);
- the identity (size and modification time) of its large model files.

Replacing a model, workflow or script at the same path drops the worker back to configured. Listing never loads a model. `studio check` errors on unavailable workers and warns on unverified ones.

## Configuration

Packaged candidates are in `studio/workers.json`. Local overrides go in `workspace/workers.json`, keyed by worker name. The fields are:

| Field | Meaning |
|---|---|
| `backend` | `command` or `comfyui` |
| `kinds` | which of image, video, voice, music, sfx the worker generates |
| `fields` | the request options it honours; anything else is rejected before a model loads |
| `limits` | `max_seconds`, `max_width`, `max_height`, `max_chars`, `max_fps`, `max_frames`; video `frame_step` requires `step*n+1` native frames |
| `device` | `cpu` or `gpu`; GPU workers take the studio GPU lock |
| `timeout` | seconds before the worker is stopped |
| `models` | files that must exist |
| `argv` (command) | an argv list containing `{request}` and `{out}`. The worker reads a JSON request file and writes `out`. No shell is involved, so text is data. |
| `url` (comfyui) | must be a loopback address |
| `workflow` (comfyui) | an API-format JSON workflow |
| `bindings` (comfyui) | `{field: {node, input}}` |
| `uploads` (comfyui) | `{reference name: {node, input}}`; the filename the server returns is bound |
| `output_node` (comfyui) | the designated final node; a preview node is never delivered |
| `node_classes` (comfyui) | node classes that must exist |

On timeout or interrupt, the ComfyUI client cancels only its own prompt. It uses `POST /api/jobs/<id>/cancel`, or removes the prompt from the queue if it is still pending. It never calls the global `/interrupt`.

## Workers on this machine (11 October 2026)

The editor and recipes remain separate from model setup. The isolated ComfyUI install is v0.39.2, Python 3.12.10, PyTorch 2.9.1 + ROCm 7.2.1, on an RX 9060 XT (gfx1200, 16 GB), driver 26.7.1. Exact packages and local-wheel hashes are recorded in `workers/comfyui/requirements.lock.txt`. Model weights, environments, downloaded sources and results are ignored by Git.

| Worker | Native output and measured generation | Review / limitation |
|---|---|---|
| `kokoro` | 9.64 s narration, mono 24 kHz WAV, 4.75 s CPU synthesis | Full decode, clipping checks and ASR 27/27 words passed; human listening pending |
| `stable-audio` music | 20 s stereo 44.1 kHz WAV, about 52 s CPU synthesis | Technical review passed; final ~2 s decay, generate longer and trim; human listening pending |
| `stable-audio` SFX | 5 s stereo 44.1 kHz WAV, 3.08 s warm CPU synthesis | Technical review passed; event count is not controllable (five knocks for a three-knock request); human listening pending |
| `flux-klein` | FLUX.2 klein 4B distilled, four steps, GPU sampling and text encoding, up to 1024 px per side | Real native stills require a content review: one seed produced an extra finger and was rejected. Read [final measurements](research/2026-10-11-local-worker-smoke-results.md) |
| `flux-klein-edit` | One `init` reference, aspect ratio retained, rescaled to ~1 megapixel; width/height arguments do not control native size | **Experimental / needs work for strict selective edits.** A red-polish smoke kept a similar hand/pose/backdrop but regenerated skin, cuticle and reflection details. Use deterministic `recolor` for faithful colour changes; this model resynthesizes an image |
| `wan` | Wan2.1 T2V 1.3B, short 320Ã—192 clips, 16 fps, at most 33 frames (2.0625 s) in the first pilot | Experimental; read the actual smoke and delegated visual review before advertising readiness. No native 4K/60 claim |

Generation timings include the worker request, model loading, generation, output transfer and technical validation. They exclude one-time model downloads and independent visual/human listening review. A cached ComfyUI result is not a fresh generation benchmark. Neural generation time is separate from the existing 207 s ordinary ten-popup edit benchmark.

FLUX with `--lowvram` ran the Qwen encoder on the CPU and took 597.84 s for a 768Ã—768 still. GPU text encoding plus full model offloading produced that size in 102.30 s and a 1024 reference edit in 61.66 s. Full-frame 1024 decoding later crossed the 14,000 MiB cap; the final workflows therefore use the stock overlapping `VAEDecodeTiled` node (256 px tiles, 64 px overlap) without reducing output dimensions. The final native 1024 tiled still took 228.39 s during a cold reload plus concurrent model hashing and passed root/delegated image QA. See [the measured evidence report](research/2026-10-11-local-worker-smoke-results.md). These earlier numbers are observations, not a guarantee for every prompt or future desktop load.

### Operating a bounded GPU session

GPU use requires the user's existing consent. Read `tools/gpu_watch.ps1` and run it for the entire inference/render session. ComfyUI stays loopback-only and is started on demand; it is closed when the task finishes. Do not run a second server on an occupied port. The validated launch arguments are:

```
main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --reserve-vram 5 --disable-smart-memory --fp8_e4m3fn-text-enc --disable-pinned-memory --disable-async-offload
```

Use `workers/comfyui/.venv/Scripts/python.exe` from `workers/comfyui/app`. Start it with `Start-Process -WindowStyle Hidden`, redirect logs into `workspace`, and retain its launcher PID. Wait for `/system_stats` to respond, then find the actual listener PID with `Get-NetTCPConnection -LocalPort 8188 -State Listen`. Verify through `Win32_Process` that its parent is the launcher you just created. **Windows Store Python starts a `python3.12.exe` child and may hide its command line.** Protect that actual child, not just its launcher:

```
powershell.exe -NoProfile -File tools/gpu_watch.ps1 -Every 2 -Minutes 30 -MaxVramMB 14000 -AlsoStopPid <actual-owned-python-PID>
```

On expiry, the explicit-owned watchdog also stops its owned inference, preventing an unprotected continuation. Start the watchdog hidden too, keep its PID/log, and confirm it is alive with no startup error before submitting a prompt. `-AlsoStopPid` snapshots the process start time so a reused PID is not killed. On a cap/health alert the owned inference process is stopped; record the failed smoke, do not raise the cap, and never change drivers or reboot. Finish by stopping only the recorded child, launcher and watcher whose start times still match, then verify the port and those processes are gone. Leave the user's other terminals alone.

### Examples and review gates

```
studio workers
studio workers smoke flux-klein --kind image --prompt "studio product photograph of a ceramic cup" --width 768 --height 768 --seed 7
studio workers smoke flux-klein-edit --kind image --prompt "a creative variation of the reference" --reference init=D:/path/reference.png --width 1024 --height 1024 --seed 8
studio workers smoke wan --kind video --prompt "a red fox walking through snow, cinematic natural light" --width 320 --height 192 --fps 16 --duration 2.0625 --seed 7
studio generate voice --worker kokoro --text "Welcome to the studio." --voice af_heart -o D:/path/narration.wav
studio generate music --worker stable-audio --prompt "gentle instrumental background music" --duration 20 --seed 7 -o D:/path/music.wav
```

The current Wan latent uses `4n+1` native frames. FPS above 16, more than 33 frames, and unsupported latent counts are rejected before model loading. Use durations `(4n+1)/fps` (17/16 or 33/16 seconds); native frame count/duration are reported and must be checked before conforming a generated insert. The first pilot has no reference/mask inputs; **VACE reference/masked video editing has not been installed or advertised**. Add that worker only when an actual edit needs it and its memory/latency gate passes. Upscaling/interpolation does not make a low-resolution generated clip native 4K/60.

Inspect native images and all frames of these tiny clips, delegate content review, record failures as failures, and only then use `studio workers review`. Each asset still needs its own creative review even after the worker is marked ready. For real footage, retain the source-preserving editing workflow. Audio waveform/ASR checks cannot replace human listening.

### Pinned sources

- [Official FLUX workflow and model guide](https://docs.comfy.org/tutorials/flux/flux-2-klein): distilled BFL 4B fp8 diffusion model, Qwen 3 4B encoder, FLUX2 VAE. Do not substitute a 9B encoder.
- [Official Wan guide](https://docs.comfy.org/tutorials/video/wan/wan-video), [official template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/text_to_video_wan.json); model repository pinned to `123acf1cc74bccbb9bfff8ac1ee72edc08c2341d`. All three downloaded files are SHA-256 verified against the repository; see `workers/comfyui/wan-models.json`.
- [AMD Windows compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html). This install uses the already pinned ROCm 7.2.1 wheels, not a driver upgrade.
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx), `kokoro-onnx` 0.6.1, `model-files-v1.0` release.
- [Official Stable Audio 3](https://github.com/Stability-AI/stable-audio-3), source commit `77e0398`, LiteRT 2.3.0, [optimized weights / Community License](https://huggingface.co/stabilityai/stable-audio-3-optimized).

The three ComfyUI workers also fingerprint the installed ComfyUI version file, PyTorch version file and `pyvenv.cfg`; changing these runtime identities invalidates prior smoke evidence. The launch flags are checked by the operator against `/system_stats`; listing does not silently start a server or load a model. After intentional session shutdown, ComfyUI workers report unavailable until the local server is restarted; their reviewed evidence remains on disk.

The diagnostic `--fp32-vae` run is not a recommended fix: it failed content review and took 455.30 s. The default launch above matches the reviewed FLUX image session. Do not claim Wan ready from a technical decode pass.

## Speech enhancement (`speech_enhance` op, DeepFilterNet3)

Not a generator: the `speech_enhance` edit op calls `workers/deepfilter/worker.py` in its own CPU venv
(torch 2.14.1+cpu, torchaudio 2.11.0+cpu, deepfilternet 0.5.6; full pin list in `requirements.lock.txt`).

```
.venv\Scripts\python -m venv workers\deepfilter\.venv
workers\deepfilter\.venv\Scripts\python -m pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
workers\deepfilter\.venv\Scripts\python -m pip install deepfilternet==0.5.6 soundfile==0.14.0
```

The model checkpoint downloads on first use to `%LOCALAPPDATA%\DeepFilterNet`. About 10 s per minute of
audio. Removes steady noise *and* mic-handling knocks (nail taps on a handheld mic) that the `audio_clean`
spectral tools miss. On the Nail Revolution podcast (2026-10-26) the user picked `mono: true` (mics mixed
first) over stereo and over attenuation limits of 12-24 dB; listening is the user's call, not a metric.
