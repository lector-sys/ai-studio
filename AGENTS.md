# Operating the studio (for Claude Code / Codex)

You are the planner/operator. Call `studio`; never hand-roll FFmpeg commands for a user request.
Every command prints JSON to stdout (progress goes to stderr). Exit code 1 means the op or its validation failed.

Run it as `D:\projects\AI_studio\.venv\Scripts\studio.exe` (or activate the venv).

## Retained workflow for future edits
The user requested on 11 October 2026 that the successful editing workflow be retained for upcoming edits.
Read [EDITING_WORKFLOW.md](EDITING_WORKFLOW.md) before preparing every edit. It adds native-resolution
quality review, source texture/luminance preservation where applicable, exact timing checks, delegated
frame reviews requested by the user, and verified delivery. A 540p preview is an iteration proxy;
review native source/master samples before final delivery. Reuse an approved look without repeating approval.

## Creative edits: confirm the look first
For any colour / look / style request ("make the sand red", "make it cinematic", "golden hour"):
1. Read it the way a colourist would. "Change X to red" on real footage means a believable material and environment
   change, not a flat fill: colour that follows the light (darker, richer shadows, softer highlights), the whole
   environment where it's the same material (defocused background included), texture kept, and subjects
   (people, animals, insects) keeping their natural colour.
2. Mock up 2–3 interpretations on 2–3 real frames: the full frame plus a 100% crop. Save them in a
   `look options` folder next to the output, with a side-by-side comparison against the original. If the user
   gave a reference image, one option should match it (`look_match`).
3. Show the user the paths and ask which matches what they had in mind. Render only after they pick.
4. Real (non-animated) footage must stay photoreal and cinematic: no hard mask edges, no flat paint, no colour halos.

Ops for this: `look_match` (regrade into a reference image's palette, sparing warm subjects), `recolor`
(selective region recolour), `color` (eq/temperature/LUT), `match_color`, `relight` (masked grade).

## Workflow
1. `studio inspect <file>`: never assume fps, resolution, color, or audio layout.
2. `studio ops` lists every operation and its params. Split the request into steps and write a job manifest
   (see `examples/background_replace.json`). Pixel params are always in **source** pixels.
3. For anything AI-dependent (mask/background), run `studio preview job.json` first. It renders at 540p proxies
   and writes `<output>.preview.mp4`.
4. Read the result JSON:
   - `validation.ok`, plus any failed `checks`
   - `steps[].meta.mask.low_confidence` / `temporal_iou_mean` / `empty_frame_ratio`
   On low confidence: try another `model` (`birefnet-portrait` for people, `isnet-general-use` for objects),
   lower `stride`, a tighter `box`, or `target=object` with SAM. Then preview again.
5. `studio render job.json` renders against master media. Completed steps are cached by content hash,
   so re-renders and resumed jobs skip finished analysis.
6. Report `output`, `validation`, and `log`. The log holds full provenance: hashes, ffmpeg commands, and model versions.

## Creative recipes: popups, lower thirds, captions, PiP, transitions, looks, narration
Use a recipe instead of writing per-edit helper scripts:
1. `studio recipes` lists them. Write a request JSON (example: `examples/creative_benchmark/request.json`, ten
   photo-card popups with source-pixel boxes and half-open `[start, end)` cue times) and run
   `studio recipe image-popup --request req.json -o job.json`. It renders the static cards, expands to a manifest
   and runs `check`; it never renders video.
2. `studio render job.json` (HDR sources default to `native-hdr-master`; AMF only when the user allowed GPU and
   `tools/gpu_watch.ps1` is running).
3. `studio review <log from the render JSON> -o <output folder>/review` builds the shared review bundle in about
   a minute: cue checkpoints (before / first / entered / mid / last / after frame of every layer), native full
   frames and 100% crops for edit and source, a playback contact sheet, waveforms, audio excerpts, an SDR viewing
   copy, and checks (exact frame clock, untouched pixels outside the cards on every frame, PSNR of the final encode,
   audio sync/clipping). `technical_ok` must be true.
4. Give the bundle to independent delegated reviewers at the same time (early frames/framing, late frames/moving
   edges, background detail, playback overview), and record each result with
   `studio review-record <dir> <kind> --reviewer … --status pass|fail|needs-work --finding …`. An agent cannot
   listen: the listening pass stays the user's, and the record must say so. Verify any reviewer claim against the
   frames before acting on it (reviewers misread faint first frames).
5. Delivery: never reuse an output name; an identical completed request returns its verified result. A
   byte-identical master (same sha256) can reuse the earlier review records, labelled as reused.
Measured on the 62.5 s 4K HLG ten-popup edit: about 3.5 minutes from inspection to verified delivery with four
delegated reviews (see `D:\edited(studio)\Creative toolkit benchmark\BENCHMARK.md`).

## Beyond masks
- Captions/transcripts: `studio captions in.mp4 -o in.srt`, `studio transcribe in.mp4 -o t.json`. Rough cuts:
  `studio cut in.mp4 -o out.mp4 --silence --fillers --phrase "take two"` (or a `cut` step with `remove` ranges taken
  from the transcript's word timestamps).
- Find footage: `studio index D:/footage`, then `studio search "product demo"`.
- Restoration: steps `stabilize`, `denoise`, `sharpen`, `deinterlace`, `lens`, `upscale`, `interpolate`,
  `match_color` (inputs: clip, reference).
- Objects: `studio object remove in.mp4 --box x,y,w,h -o out.mp4` (clean plate, static camera best), then
  `replace --with img.png`, `insert --with logo.png --box ...` (tracked). Use `relight` for masked grading.
- Generation: `studio generate video --prompt ... -o b.mp4`. The procedural backend gives placeholder plates. Real
  models go through `STUDIO_GEN_CMD` or ComfyUI (see README). Upscale/interpolate generated clips before
  compositing into 4K60.
- Before rendering a hand-written manifest, run `studio check job.json`. To let the studio retry low-confidence
  masks on its own, use `studio auto job.json`. `studio capabilities` shows which backends exist here.

## Models and long footage
- `studio models` lists the named models (face detector, GFPGAN, Real-ESRGAN, RIFE, MiniLM). They download on first
  use; `studio models download <name>` fetches them ahead of time. They run on CPU and are slow at 4K, so apply them
  to trimmed segments: `face_restore` (~1 s per face per frame), `upscale -p backend=onnx`, `interpolate -p backend=onnx`.
- To isolate ONE person among several, give the mask step a `box` with `target=person`; the identity tracker keeps
  that subject through crossings and short occlusions.
- Long clips: chunkable steps split into 60 s chunks automatically (resumable). Disk: `studio cache prune --older-than 7`.

## Rules
- Originals are never overwritten (the runner refuses).
- Don't render the full master after every prompt iteration. Preview first.
- Background removal outputs alpha, so it needs `--profile alpha-master` (ProRes 4444 .mov). The CLI does this by default.
- Object boxes/points refer to **frame 0** of the step's input. Inspect a frame to pick them:
  `ffmpeg -i in.mp4 -frames:v 1 f0.png`.
- Tracking has no identity model. When subjects cross, the mask can jump to the other person.
  The mask stats flag this (`low_confidence`). Trim around crossings, or re-run on sub-ranges.
- GPU use is opt-in. Don't enable it without the user's OK
  (`STUDIO_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider`, `STUDIO_HW_ENCODE=1`).

## Local neural generation
Read `docs/LOCAL_WORKERS.md` before starting a local generation session. Use `studio workers` discovery; do not silently substitute procedural or cloud output. FLUX text-to-image, experimental FLUX reference resynthesis and experimental Wan clips use the isolated loopback ComfyUI runtime; Kokoro/Stable Audio use CPU workers. GPU permission already granted in this conversation remains valid. Protect the actual task-owned Store Python child with `gpu_watch.ps1 -AlsoStopPid`, keep the 14,000 MiB cap, use the documented offloading flags and tiled image decode, and close child/launcher/watchdog at task end. Never advertise a content-failed smoke as ready. Generative reference edits can alter skin/reflections and are not a replacement for the retained faithful recolour workflow; Wan/VACE reference or masked editing is not verified. Voice/music/SFX human listening remains pending. Runtime/version files join the ComfyUI fingerprint; session shutdown makes the server temporarily unavailable, without discarding reviewed evidence.
