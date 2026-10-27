# Plan review (Claude, 11 October 2026) and slice-3 local-worker plan

The two Codex draft plans were reviewed against the approved specification and the handoff notes. They are used
as the executable plan with the corrections below, and the missing third plan follows. Execution method: **native
sequential** (handoff recommendation), with focused tests per task and one fresh whole-branch reviewer per slice.

## Corrections to the foundation plan (slice 1)

1. **Runtime repair is not needed.** `.venv\Scripts\python.exe` and `studio.exe` work outside the Codex sandbox:
   the Windows Store Python 3.12.10 behind `pyvenv.cfg` is installed. The Codex failure came from the sandbox's
   inability to reach the WindowsApps alias. No `repair_runtime.ps1` is written. If the Store Python is ever
   removed, point `home`/`executable` in `pyvenv.cfg` at any Python 3.12.x install; nothing else changes.
2. **Resource hashing** hashes *every* string parameter that names an existing file, recursively, instead of a
   fixed field list. This covers fonts, LUTs, ASS/SRT files, ONNX models and workflows without a list to keep
   current. Missing paths in the documented file fields (`file`, `font`, `lut`, `cue_file`, `workflow`) fail.
3. **Cache verification** checks the sidecar's recorded byte size on every hit; a full hash of a 4K lossless
   intermediate costs about 10 s per step. Partial outputs never become hits (write to `.part`, then rename).
4. **Delivery protection** applies to `final` renders. Previews stay overwritable (they are the iteration loop).
   Step keys are computed before any rendering, so an existing output from a different request fails at once;
   an identical completed request returns its recorded result with the file's hash verified.
5. **Clock preservation.** Lossless `.mkv` intermediates store millisecond timestamps. For time bases coarser
   than 1/2000 (e.g. the iPhone 1/600) the final encode restores exact source ticks; for finer time bases
   (1/60000) the round-trip error is at most 0.5 ms. The clock check reports the measured maximum deviation in
   ticks and seconds and passes at max(1 tick, 0.5 ms), naming the container limit.
6. **HDR graphics white** is a parameter (`graphics_white_nits`, default 203 per ITU-R BT.2408). The approved
   nail-shape look used 500 and the benchmark reuses that value.
7. **Separate-ops comparison.** The existing `overlay` op composites sRGB cards into HLG without a transfer
   conversion, so it cannot produce equal quality. The benchmark measures its render cost for reference and
   records that it fails the colour contract; the full reviewed runs use `compose`.

## Corrections to the generation-contract plan (slice 2)

1. Voice/music/SFX requests never receive the visual 5 s default duration. Voice duration is measured from
   synthesis; music/SFX pass `seconds` only when the worker declares it.
2. `generate_media` still returns a path. Provenance (kind, worker, device, native properties, output hash,
   elapsed seconds) is written beside the output as `<output>.gen.json` and copied into step metadata.
3. ComfyUI support is completed in code and fake-server tests. A real ComfyUI install is part of slice 3, gated
   on AMD runtime verification.

## Slice 3: local workers (plan)

Goal: real, reviewed outputs for each kind on this computer, or an exact unavailable reason.

Constraints: separate pinned environments under `D:\projects\AI_studio\workers\<name>\` (ignored by git); no
change to drivers, registry, system Python or the editor venv; GPU inference runs bounded, announced, under
`tools/gpu_watch.ps1`; never Vulkan; no cloud fallback.

| Order | Worker | Steps | Ready when |
|---|---|---|---|
| 1 | Kokoro-82M ONNX (voice, CPU) | venv with `kokoro-onnx`; download `kokoro-v1.0.onnx` + `voices-v1.0.bin`, record hashes; command-worker wrapper reading the request JSON | narration WAV passes `validate_audio`, listened/reviewed for pronunciation and punctuation |
| 2 | Stable Audio 3 Small (music, sfx, CPU LiteRT) | clone official repo at a pinned commit; venv per its `optimized/tflite` requirements; fetch model files if public; wrapper calls `sa3_tflite.py` via argv | one music bed and one recognisable effect pass audio QA + review; if the weights are gated, report the exact access requirement |
| 3 | FLUX.2 klein 4B (image, ComfyUI) | verify the current official AMD Windows PyTorch/ROCm wheel for gfx1200 against Python version; ComfyUI venv pinned to a release; download klein 4B files from the official guide; API workflow with explicit bindings and output node | a generated still and a reference edit pass image QA and content review, with GPU memory/time recorded |
| 4 | Wan2.1 1.3B / VACE (video, ComfyUI) | same ComfyUI install; official Wan 1.3B workflow at low resolution and short length | a short clip passes video QA and review; otherwise recorded as unavailable or experimental with measured limits |

Each worker gets a `workers.json` entry (workspace override), a smoke via `studio workers smoke <name>`, a recorded
review, and readiness evidence. Changing model/workflow/runtime files invalidates it through the fingerprint.
Install one worker at a time and finish its review before starting the next.
