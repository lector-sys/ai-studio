# Local creative workflows for AI Studio

Date: 11 October 2026

Status: Written specification approved by the user on 11 October 2026. Implementation-plan review and execution-method selection remain pending.

Project: `D:\projects\AI_studio`

## Outcome and scope

Extend the current local editor so an operator can assemble complex creative jobs from tested recipes and local generation workers, preserving source quality and reporting what this machine can actually do. Retain the existing Studio CLI, version-1 job manifests, operation registry, cache, GPU renderer, review workflow and output safeguards.

The first delivery covers reusable editing styles; timed image/text animation; captions, transitions and sound mixing; standalone voice, music and sound-effect assets; reference-aware image/video generation; and shared quality-review artifacts. Neural workers become usable capabilities only after producing and reviewing real outputs on this computer.

Here, teaching the project means saving researched guidance, executable recipes, provider configuration, checks and measured examples in the project. It does not mean training a foundation model or promising that every imaginable creative task is supported. When an installed tool cannot perform a task, build the missing Studio adapter, composition tool or workflow. Do not label a procedural placeholder as neural generation.

Success is usable local functionality with evidence, rather than a larger menu of unverified features.

## Evidence and constraints

The current project already implements overlays, text, ASS captions, transitions, LUTs, restoration, audio cleanup/mixing and content-addressed steps. Generation adapters exist for procedural plates, a subprocess command and ComfyUI. The baseline procedural/command test passed; neural generation has not been demonstrated. The ComfyUI adapter currently ignores the init reference, chooses the first returned image/video and lacks standalone audio handling.

This machine has an AMD Radeon RX 9060 XT with 16GB VRAM. AMF video encoding has been verified, including a native 10-bit HLG export. Neural inference requires separate verification. The Studio virtual environment points to a missing Windows Store Python installation; bundled Python 3.12.14 was sufficient for the baseline check. Preserve the existing environment and caches while repairing the normal entry point.

The preceding approximately 62-second, ten-popup edit took approximately 39 minutes from inspection to delivery. The final composition plus GPU encode took approximately 1 minute 54 seconds; all recorded render iterations totaled approximately 6 minutes 43 seconds. These are historical observations, not a clean baseline benchmark. Much of the elapsed time came from preparation, custom helpers, repeated checks and review coordination.

The user's target is **5–7 minutes for a comparable basic edit, including careful editing, multiple reviews and verified delivery**. Neural synthesis is benchmarked separately. Neither fewer reviews nor lower-resolution masters qualifies as an optimization.

## Architecture: reuse the current execution path

Use one path: request -> reusable recipe -> checked Studio manifest -> existing operations/local workers -> cached outputs -> shared technical and human review -> verified delivery.

### 1. Recipes and style discovery

Store a small versioned recipe catalog as packaged project data, with a matching guide. Each recipe names its supported inputs, typed parameters, defaults, required capabilities and the existing steps it produces. Expansion produces an ordinary version-1 manifest that can be inspected, checked, previewed and rendered without the recipe system.

Add `studio recipes` to list recipes and `studio recipe <name> --request <json> -o <manifest.json>` to expand a request. Expansion validates paths, IDs, timing, finite numbers, geometry and allowed fields before writing. It never runs supplied scripts or evaluates arbitrary expressions. Existing manifests and CLI commands retain their behavior.

Initial recipes:

| Recipe | Behavior | Inputs that must be explicit |
|---|---|---|
| Image popup with caption | Rounded image card, small text below, eased slide/scale/fade | Asset, text, start/end, placement, source-pixel dimensions |
| Lower third / callout | Legible compact label, controlled entrance/exit | Text, placement, timing, palette/font |
| Kinetic captions | Timed ASS captions with restrained emphasis | Reviewed transcript/cues, font, margins, word timing |
| Comparison / picture in picture | Labeled simultaneous images or clips | Sources, geometry, crop intent, timing |
| Clip transitions | Existing fade/dissolve/wipe with audio continuity | Ordered clips, transition type and duration |
| Editorial / vintage / monochrome | Existing grading or LUT operations | Approved look, strength, source color contract |
| Narration with music and effects | Generated/imported audio, cleanup, level control and ducking | Audio assets, cue times, gain and output policy |

Use current FFmpeg/ASS operations and available fonts first. Avoid a second editor or an unrestricted template language. The retained 2–3 still-option approval workflow applies to new color/look requests; reuse an approved look without asking again. Styling must not move or rescale the underlying footage unless the request calls for it.

### 2. A bounded combined-edit operation

Add `compose` to the existing operation registry for a source clip plus timed image/text layers and compatible audio cleanup/mixing. Its parameters contain a validated list of layers, optional audio treatment and an explicit color policy. Assets are manifest sources referenced by source IDs, not untracked hidden file paths.

Use one source decode and one lossless intermediate for the compatible work, followed by the normal final encoder. Reuse the existing overlay/text/audio helpers where their output satisfies the source contract. Promote the verified popup rendering approach only where it supplies missing HDR or animation behavior; remove source-specific dimensions and wording.

The first implementation supports cards, simple text, slide/scale/fade animation and ordinary audio cleanup/mixing. Multi-clip transitions continue through the existing transition operation. Masks, restoration, inference and complex 3D rendering remain separate steps. Do not construct a general graph optimizer. The explicit compose operation must fail preflight on an unsupported color/format combination, identifying the required conversion; it must never silently flatten HDR to SDR.

Measure the same edit using existing separate operations and compose. Retain the combined path only if it preserves quality and reduces measured work or latency. Its renderer runs at both proxy and source resolution with geometry expressed in source pixels.

### 3. Local generation and audio contract

Extend the existing generator dispatcher, command adapter and ComfyUI adapter rather than implementing an inference engine. Support kinds `image`, `video`, `voice`, `music` and `sfx`. Preserve existing `generate_media` callers and its path return value; keep richer provenance in the existing result/log sidecar. Extend `studio generate` with the three audio kinds and the corresponding named workers. A separate `generate_audio` operation emits an audio asset usable as a later step input; the existing `generate` operation stays visual.

Visual requests retain prompt, seed, width/height, rational fps and duration, adding named references/mask and a designated workflow output. Audio requests carry text/prompt, voice where applicable, seed and duration where the selected worker supports it. Worker metadata declares supported kinds, limits and fields. Voice duration is measured from synthesis; do not silently time-stretch narration to an arbitrary duration. Unknown or unsupported options fail before model loading.

For command workers, pass a structured argument list or a request JSON file without a shell. Retain the current trusted command-template interface for compatibility. Separate workers use isolated environments so their PyTorch/ROCm/audio dependencies do not replace the editor's runtime.

For ComfyUI:

- Upload init/reference assets and masks locally; bind returned filenames to explicit workflow node/input mappings.
- Validate required models, node classes, dimensions and frame constraints before enqueueing.
- Require a designated final output node/type. Ambiguous or absent outputs fail; never deliver an intermediate preview by selecting the first file.
- Surface prompt validation and execution errors promptly. Use bounded HTTP waits and job timeouts. Cancel only the Studio-owned request; do not interrupt unrelated queued work.
- Download into a temporary sibling file, verify the media, then publish atomically. Remove partial outputs on failure. Preserve generated audio when supplied with a video.
- Record prompt ID, workflow and reference hashes, model identity/version, seed, requested and native output properties, actual inference device, conversion steps and timings.

Local endpoints are used by default. No automatic cloud fallback or paid services are part of this design. Originals and existing deliveries cannot be overwritten by either direct generation or job execution.

### 4. Readiness and worker selection

Extend `studio capabilities`, model information and manifest preflight to distinguish:

- **Unavailable:** runtime, worker, model or required node missing; report the concrete reason.
- **Configured:** files/server/options exist but no matching validated smoke output is recorded.
- **Ready:** required dependencies pass discovery and a recent smoke output passed technical checks and a real content review. Record the tested kind, device, model/workflow hashes, date, limits and measured time.

Changing a model, workflow, worker or runtime invalidates the associated readiness evidence. Lightweight capability listing does not load models or download assets. Keep current capability keys compatible, adding detailed worker state instead of changing booleans into objects. A neural request never falls back to the procedural generator. CPU fallback, if used, is explicit in results and timings.

Pilot workers, in order:

| Worker | Intended first test | Execution decision |
|---|---|---|
| Kokoro-82M ONNX | Short narration with punctuation and pronunciation | CPU worker first |
| FLUX.2 klein 4B | Generated still plus reference-conditioned edit | ComfyUI; verify AMD inference and memory before ready |
| Wan2.1 1.3B / VACE 1.3B | Short low-resolution insert and reference/masked edit | Experimental ComfyUI workflow; memory/latency gated |
| Stable Audio 3 Small Music/SFX | Short music bed and a recognizable effect | Official Windows CPU worker; verify model access and applicable license |

Choose and pin a mutually compatible Windows/AMD runtime from the current official matrix at setup time. AMF encoder availability cannot stand in for model compatibility. Download only the selected public model assets, record hashes/license source, and run bounded smoke cases before increasing resolution or length. If gated models require an unavailable account/license action, finish unaffected workers and report that precise dependency; do not substitute an untested model silently.

Blender for complex 2D/3D assets, Remotion, ACE-Step songs, stem separation and stronger neural denoising are documented extensions, not compulsory dependencies for the first delivery. Use existing installed tools when a requested task needs them; add a worker only after an actual capability gap. Do not build bespoke neural models from scratch.

## Quality, timing, caching and review

### Source and delivery contract

Inspect once and reuse the source hash, display geometry/rotation, frame timestamps, rational time base, color/bit depth and audio layout. Ordinary overlays preserve the base framing, frame count and timeline. A frame-rate conversion, crop or SDR conversion is an explicit creative/export decision. Half-open time ranges `[start, end)` apply consistently to layer cues and review checkpoints; select frames using actual source timestamps for VFR footage.

Preserve the source clock through lossless and final encoding with an explicit encoder time base. Check source/master timestamp deltas rather than accepting average fps alone. For unchanged timeline edits, deviations must stay within one source time-base tick; synthesis/conformed clips use their declared timeline. Check first/last enabled layer frames.

HDR rendering preserves source transfer, gamut and precision; composite graphics through a declared color conversion without changing untouched background pixels. Check rotation, full-frame scale, native texture, alpha edges, halos, clipping and native crops. Static HDR10 mastering/CLL and Dolby Vision metadata preservation are encoder-dependent; report any lost metadata and choose a compatible alternative when preservation is required. A tone-mapped viewing copy is separately named and never replaces the native master.

Add audio-only validation for stream presence, full decode, duration, rate/layout, measured peak/clipping and applicable loudness/silence. Keep native generated WAV and resample once for the video timeline. Cleanup filters require measured delay compensation; the observed 25ms denoiser delay is one calibration case, not a universal constant. Verify waveform/onset alignment at start, middle and end alongside listening. Do not treat equal container start times as proof of speech sync.

Existing H.264/HEVC/AV1, ProRes, alpha and archive profiles remain. Add WAV PCM for audio delivery. Keep captions, manifests and editable recipe data beside results. SVG/Lottie assets need a tested raster worker; OTIO/editor interchange needs round-trip tests before advertised support.

### Correct cache and shared review bundle

Cache keys include source and asset contents, operation/recipe versions and parameters. Hash file-valued fonts, LUTs, subtitle files, model/workflow configuration and worker versions where they affect output. A changed file at the same path cannot reuse stale output. Verify cached outputs before use; partial files are never cache hits.

Build one content-addressed review bundle: inspection, cue/timing table, native samples/crops, popup entrance/exit checkpoints, a playback copy, audio excerpts, technical checks, provenance and unresolved findings. Independent delegated reviewers use this bundle concurrently under the retained workflow. Re-decode or re-review only changed or affected evidence; complete a full playback/listening pass and full-file decode before final delivery. Never mark automated checks as human review.

Save hash-verified output in the approved edited folder, retaining the user's existing `D:\edited(studio)` destination unless a new task chooses another. Close task-owned render processes and terminals when finished. Keep user's pre-existing terminals intact.

## Delivery order and affected interfaces

Implement three independently verifiable slices in sequence:

1. **Workflow foundation:** runtime entry-point repair, packaged recipes, bounded compose, resource hashing, source/audio preservation, shared review bundle and complete-edit benchmark. Prove the ordinary edit before adding model overhead.
2. **Generation contract:** direct CLI and job audio outputs, reference uploads, designated outputs, errors/timeouts, safe cancellation, atomic publication, audio QA and accurate readiness. Use small fixtures and a fake local worker/server to exercise failures without downloading models.
3. **Local workers:** isolated pinned runtimes, selected model setup, real image/video/voice/music/SFX smoke outputs, content reviews and per-worker timing/memory records. Advertise only passing kinds and limits. Finish the operating guide with working recipes and evidence.

| Area | Required reach |
|---|---|
| CLI and recipes | `studio/cli.py`, packaged recipe data, schemas/examples, package data configuration |
| Operations and timeline | `studio/ops.py`, new compose helper only if needed, `studio/media.py`, shared audio behavior |
| Runner/cache/profiles | `studio/jobs.py`, asset hashing, audio step/final handling, output protection and provenance |
| Generation | `studio/generate.py`, isolated worker entry points/configuration, visual and audio callers |
| Readiness/preflight | `studio/devices.py`, `studio/models.py`, `studio/agent.py`, GPU scheduling where needed |
| Quality/review | `studio/qc.py`, retained workflow and review-bundle tooling |
| Regression and distribution | Existing tests plus targeted contract/fixture checks, `pyproject.toml`, README, `AGENTS.md`, plugin skill documentation |

Update affected callers and exports together. Keep the current UI functional through compatible manifest/capability changes; avoid a UI redesign. Build a written implementation plan after specification approval, with bounded tasks, tests and execution method review.

## Acceptance evidence

| Test | Passing evidence |
|---|---|
| Compatibility and data safety | Existing relevant CLI/job tests pass; unsupported kinds/params fail clearly; originals and old deliveries remain hash-identical; failed work leaves no published partial output |
| Recipe correctness | Deterministic expansion; invalid geometry/timing/assets rejected; packaged recipes usable after installation; expected boundary frames and captions visible |
| Compose quality | SDR and native 10-bit HDR cases pass decode, geometry, timeline and color checks; untouched source regions preserved in lossless intermediates; final compression measured separately |
| Audio | Audio-only and mixed outputs decode; expected samples/duration/rate/layout; no introduced clipping; calibrated delay and start/middle/end sync checks plus listening pass |
| Cache/readiness | Same request reuses valid artifacts; replacing an asset/workflow at the same path invalidates results/evidence; missing models or changed workers cannot remain ready |
| Worker failures | Structured prompt/execution errors, wrong output type, ambiguous output, timeout and cancellation return actionable failures without affecting another job |
| Local generation | Real reviewed still and reference edit, short video, narration, music and SFX outputs; logs name the model/device/native properties; any unavailable case stays honestly labeled |
| Basic-edit speed | Fixed source/hash and ten-popup recipe, unchanged quality/settings/review requirements; measure inspection, prep, render, QA, review and delivery; record cold and warm runs and at least three comparable runs per path, reporting median and range |
| Speed claim | Complete reviewed/delivered edit meets 5–7 minutes on this computer before claiming the target achieved; otherwise report measured phase costs and fix the demonstrated bottleneck |

The benchmark clock includes active editing, reviews and delivery. Report pauses awaiting a required user choice separately; do not hide processing or reviewer wait time. Record token usage only when reliable counters are available, otherwise mark it unavailable. Neural benchmarks separately include load, generation, peak memory, conversion, retries and technical/content review; do not extrapolate NVIDIA timings to this AMD card.

## Research basis and limits

The accompanying [creative-toolkit research](../../research/2026-10-11-local-creative-toolkit-research.md) records the inspected code and official-source research. Principal implementation references are [FFmpeg filters](https://www.ffmpeg.org/ffmpeg-filters.html), [ComfyUI routes](https://docs.comfy.org/development/comfyui-server/comms_routes), [AMD Windows compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html), [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VACE](https://github.com/ali-vilab/VACE), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx), and [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3).

Research and design are complete at this stage. Product implementation, model downloads, neural smoke outputs and the 5–7-minute benchmark are pending. Model/runtime compatibility, available model access and generation speed remain measured readiness gates, rather than promises.
