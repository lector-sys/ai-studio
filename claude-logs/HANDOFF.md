# AI Studio: handoff (last updated 2026-10-10)

## Where things stand
- v0.3.x, in git (`git log`). All spec phases 0–7 are built, plus the v0.3 backlog.
- Tests: `.venv\Scripts\python -m pytest -q tests`. The last full run passed 59 in about 90 s. `tests/test_v03.py`
  now has 9 tests, and the full suite wasn't re-run after the last changes.
- It's also a Claude Code plugin (`ai-studio@ai-studio`, user scope, v0.3.1): the `studio` skill works from any folder.
- The first real-footage job (4K60 ants clip, "make the sand red") is **done and delivered**. See session 3 below.

| Session | Transcript | What was done |
|---|---|---|
| 1 | `session-2026-10-08.jsonl` | Phases 0–2: CLI, 21 deterministic ops, mask/track/background, cache, QC, provenance |
| 2 | `session-2026-10-08-phases3-7.jsonl` | Phases 3–7: STT/captions/cut/index, restoration, generative ops, check/auto/bench + VRAM scheduler, local UI |
| 3 | `session-2026-10-09-v03-plugin-4k-test.jsonl` | v0.3 backlog, plugin, GPU encode + watchdog, first real 4K60 job (red-earth regrade), look_match, parallel frames |

**To resume**, open Claude Code and say: *"D:\projects\AI_studio: read claude-logs/HANDOFF.md and continue"*.
Or run `claude --resume` from `C:\WINDOWS\system32` and pick session `ae36d4de-5bff-495d-95b3-0abf73a4431a`
(session 3). Session 2's id was `30134ee2-a07b-48ab-b8d8-bbf168f25ba0`.

## Rules the user set (also in memory and AGENTS.md)
- **Creative/look edits: confirm first.** Read the prompt the way a colourist would. Mock up 2–3 interpretations as
  stills (full frame plus a 100% crop) in a `look options` folder, ask the user to pick, and only then render. Real
  footage must look photoreal and cinematic: no flat fills, hard mask edges or halos.
- **GPU rendering is allowed (from 2026-10-10), but only monitored.** Run `tools\gpu_watch.ps1` under Monitor
  during any GPU work. It kills AMF encodes on a driver reset (TDR 4101/4116/4117), a bad adapter status, or VRAM
  over the limit. Use `STUDIO_HW_ENCODE=1` for the AMF encode. DirectML for the ONNX models isn't installed or
  tested yet. Vulkan is still off-limits.

## Session 2 changes
- New modules: `studio/speech.py`, `restore.py`, `generate.py`, `sched.py`, `agent.py`, `ui.py`, `ui.html`.
- New commands: `transcribe`, `captions`, `cut`, `index`, `search`, `object`, `generate`, `capabilities`, `schema`,
  `check`, `auto`, `bench`, `ui`.

## Session 3 (2026-10-09/10)
**v0.3 backlog**
- Git repo; Claude Code plugin plus local marketplace (`.claude-plugin/`, `plugin/`).
- `studio cache prune`; per-op cache keys; GPU lock timeout.
- HDR10 passthrough (x265, SVT-AV1); chunked long-footage steps (`Step.chunk`, `STUDIO_CHUNK_SECONDS`).
- Identity-aware mask tracking (overlap + colour histogram, velocity coasting through occlusions).
- `models.py` + `studio models`; `face_restore` (YuNet + GFPGAN); RIFE `interpolate backend=onnx`; Real-ESRGAN.
- Semantic search (MiniLM ONNX); UI timeline, mask overlay and paint tool -> `mask_edit` op.

**First real job:** `D:\footage\18330235-uhd_3840_2160_60fps.mp4` (macro of pale ants on grey stone, 23 s, 4K59.94)
- The ask: "change the colour of the sand to red, trim to 16 s, save in `D:\edited(studio)`".
- Attempt 1 (`recolor` op, flat red on the in-focus ground) was **rejected** as "way off". The user sent a reference
  image (red-earth terracotta world, golden ants), now saved at `D:\edited(studio)\look options\reference.png`.
- I mocked up looks A (reference match), B (daylight red earth) and C (red ground, grey background). **The user
  picked B** and the first 16 s.
- Delivered: `D:\edited(studio)\ants_red_earth_daylight_16s_4k60.mp4`. Job: `workspace/jobs/red_earth_daylight.json`
  (`look_match` with exposure=keep, mid_offset=-8, contrast=1.2, saturation=0.9). All QC passes; GPU (AMF) encode
  took 14 s with no driver events.
- The rejected `ants_red_sand_16s_4k60.mp4` is still in that folder. I offered to delete it; no answer yet.

**Studio changes from the job**
- New ops: `recolor` (region/hue/saturation selection, `spare_warm`) and `look_match` (regrade into a reference
  image's brightness-dependent palette; warm subjects keep and amplify their warmth, no masks).
- `map_frames(parallel=True)`: a thread pool with ordered output, only for stateless ops. `look_match` also uses
  lookup tables. Together that's about 5× (1.3 → 6.7 fps at 4K). `STUDIO_FRAME_WORKERS` overrides the thread count
  (default cpu/2).
- The FFV1 writer is threaded (16 slices). AMF probe fix: HEVC AMF rejects 256 px frames, so the probe now uses
  640×360. Before the fix, GPU renders silently fell back to the CPU.
- `tools/gpu_watch.ps1`: GPU health watchdog.

## Gotchas
- faster-whisper's PyAV decoder breaks against the installed `av`, so audio goes ffmpeg → numpy.
- The installed onnxruntime loads IR ≤ 13 only.
- minterpolate drops tail frames (padded, then trimmed). Use half-open ranges in `select`.
- 540p previews under-show thin-subject detail (the ants look less golden there); judge looks on full-res stills.
- Bash heredocs with nested quotes/escapes broke file edits twice. Write the edit script to the scratchpad and run it.
- Speeds on this PC: the 4K CPU grade runs at about 6–7 fps with parallel frames; the AMF 4K60 encode runs at about
  70 fps; the CPU x265 slow encode takes about 100 s per 16 s of 4K60. GPU files come out about 1.5× bigger at
  similar quality.

## Next
1. Waiting on the user: delete the rejected red-sand file?
2. Ideas, not started:
   - parallelise other stateless per-frame ops
   - DirectML for the ONNX models (needs `onnxruntime-directml`, a monitored test first)
   - turn the look-dev mock-ups into a `studio looks` command
   - CLIP visual search
3. Not done on purpose: a neural generator (GPU + ComfyUI) and SAM2 (torch + GPU).
