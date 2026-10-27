# Retained editing workflow

Saved at the user's request on 11 October 2026. Use this workflow for upcoming edits in `D:\projects\AI_studio`, adapting the actual operations and checks to each request. Explicit instructions from the user take priority.

## Source and agreed look

1. Read the studio's `AGENTS.md`, inspect the original media through the studio, and record its hash, dimensions, framing, frame rate/time base, frame count/duration, colour space/range/transfer, bit depth and audio. Keep originals untouched.
2. For a new colour or style request, show three still options from real footage, with original/edited comparisons and native 100% detail crops. Obtain the user's selection before a video preview. Keep an already approved look; do not ask for the same approval again. Generated references may guide the palette, but must not replace source texture or composition.
3. Write and check a studio job manifest. Preserve the original subject scale, framing, resolution, timing and audio unless the requested edit changes them. Keep child consoles hidden and close any terminals opened for the task. GPU use remains opt-in.

## Edit and sample

4. Use the studio workflow and render from original master pixels. Small proxies can help iteration, but inspect native-resolution samples before the master and clearly identify any proxy shown to the user. Use short samples to resolve problems before rendering the full clip.
5. For recolouring, change chroma while preserving source luminance and texture. Use gamut-safe colour adjustment; HSV brightness alone does not preserve perceived brightness. Use masks for colour eligibility without rescaling the original image. Preserve unrequested details such as eyes, beak and neutral materials.
6. Rebuild or track masks for each clip. Inspect thin edges, concave gaps, mixed-colour pixels, moving or occluded subjects, and changes where separate mask components merge. Use smooth colour transitions on feather boundaries, branches and leaves. Check for halos, highlighted outlines, flat colour, spill, patchy background colour and lost detail. Existing camera blur should stay natural.
7. Keep original luminance and edited chroma aligned by frame index and the actual source clock before combining them. Use a lossless intermediate when needed, then a high-quality compatible master export with correct colour metadata. Preserve HDR/bit depth appropriately when present; this parrot's SDR settings are not universal defaults.

## Independent review and verification

8. The user requests delegated frame review as part of this retained workflow. For footage edits, delegate independent reviews covering early frames/framing, late frames/moving edges, and background detail. Give reviewers original/edited pairs and native detail crops. Include beginning, middle, end, motion/occlusion or mask-transition samples, and adjacent frames for flicker. Resolve findings before final delivery. If delegation is unavailable, perform the review directly and disclose that limitation.
9. Check the complete final file for clean decoding, expected geometry, timing/frame count, audio and colour metadata. For a colour-only edit, compare source and lossless intermediate luminance on every frame, then measure final compressed luminance error and brightness drift. Check requested colours and continued motion. Adapt these checks for cuts, speed changes, exposure changes and other intentional modifications. Mask confidence, hue scores and the studio's automatic validation alone are insufficient.
10. Inspect native original/edited comparisons for subject scale, feather texture, branch/leaf detail, outlines, colour continuity and flicker. Automated checks cover all frames where feasible; state the actual manual review coverage without claiming every frame was visually reviewed.

## Delivery

11. Save a uniquely named native-resolution master and supporting comparisons, notes, job manifest, render log, verification and hashes. The user's default edited folder is `D:\edited(studio)` unless specified otherwise. Keep a user-facing copy in the current chat's `outputs` directory for inline display. Verify delivered copies by hash. Never overwrite original footage or earlier approved versions.

## Time and token efficiency

User clarification on 11 October 2026: a basic edit comparable to the 62-second nail-shape video should target 5–7 minutes from source inspection to verified delivery, including careful editing and multiple reviews. This is an acceptance target, not a benchmark already achieved. Do not reduce the quality or review requirements to meet it.

- Reuse verified Studio operations, card assets and encoding settings. Inspect and transcribe once per source. Verify source-specific HDR, frame timing and audio-filter delay in the native sample before the full master; avoid writing new task-specific helpers for routine edits.
- Prepare one shared review bundle and run the independent reviewers concurrently. Decode comparison frames and detail crops in batches per file, and reuse them across reviews. Keep source metadata, clock and analysis cached by source hash and relevant settings; repeat processing only after a relevant change or a concrete finding.
- Run full-file checks together where possible, retain the checks appropriate to the edit, and render the master after sample findings are resolved. Record end-to-end wall time, phase timings and token use when available. If an actual quality issue will exceed the target, explain the issue and expected extra work promptly. Do not claim the faster workflow is validated until measured end to end with the required checks.

### Measured on 11 October 2026
The fixed benchmark (62.5 s 2160x3840 10-bit HLG `IMG_3532.MOV`, ten nail-shape popups, approved card look,
delay-compensated dialogue cleanup, AMF HEVC Main10 QP 10/12, four delegated reviews, hash-verified delivery) took a
median of about 3.5 minutes over three cold runs and about 2 minutes warm, using `studio recipe image-popup`,
`studio render` and `studio review`. Exact phase timings are in
`D:\edited(studio)\Creative toolkit benchmark\BENCHMARK.md`. The listening pass is not included (an agent cannot
hear audio); it remains the user's.

## Reference from the successful parrot revision

- Master: `D:\edited(studio)\parrot_option_C_revised_720p.mp4`.
- Review bundle: `D:\edited(studio)\parrot revised review`.
- Workspace bundle: `C:\Users\meyaa\Documents\Codex\2026-10-10\d-projects-ai-studio\outputs\parrot-option-C-revised`.
- The corrected master retained 1280 x 720 framing, 30000/1001 fps and 461 frames. Original luminance was identical in the lossless intermediate; the final compressed mean luminance error was 0.325 on the 8-bit scale.
- Source/output hashes and complete measurements are in the reference bundle's `quality_verification.json`.

Reuse the process, not the parrot's coordinates, masks, palette, timing constants or colour-space assumptions. Calibrate them from each new source. Instructions were retained; no new footage was rendered while saving this workflow.
