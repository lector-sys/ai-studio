# Claude handoff: local creative workflows for AI Studio

Date: 11 October 2026

The user explicitly asked Codex to stop the current work, create this handoff and provide a continuation directive for Claude. Codex is stopping after preserving this checkpoint. **No product implementation or neural-model installation was performed for the creative-toolkit task.**

## User objective and approvals

The user wants AI Studio researched and expanded to handle complex creative editing, styles, formats, animation, visuals, voice, music and sound effects with reusable local tools. Their provider preference is: **“try local tools, if the current local tools dont support then build the required tools.”** Use existing local tools first, then build missing Studio adapters/workflows; no cloud fallback or paid generation.

The user approved the design direction and then explicitly approved the written specification. The approved design retains the existing editor/GPU renderer, adds reusable styles and combined popup/text/audio recipes, extends local image/video/audio workers, and adds cached reviews, readiness checks and speed benchmarks. Implement the workflow foundation first, then generation plumbing and actual local workers.

Read these files first:

1. `D:\projects\AI_studio\AGENTS.md`
2. `D:\projects\AI_studio\EDITING_WORKFLOW.md`
3. `D:\projects\AI_studio\docs\superpowers\specs\2026-10-11-local-creative-workflows-design.md` — **approved specification**
4. `D:\projects\AI_studio\docs\research\2026-10-11-local-creative-toolkit-research.md` — completed official-source research
5. `D:\projects\AI_studio\docs\superpowers\plans\2026-10-11-creative-workflow-foundation-plan.md` — **incomplete, unapproved draft**
6. `D:\projects\AI_studio\docs\superpowers\plans\2026-10-11-local-generation-contract-plan.md` — **incomplete, unapproved draft**

The third plan, for installation and verification of actual local neural workers, **has not been written**. The first two drafts have not completed consistency/spec-coverage review. Do not mistake documentation examples for implemented source code or passing tests.

Codex's active brainstorming/writing-plans skills required the user to review the implementation plan and select an execution method before implementation. That review/selection has **not happened**. The recommendation is native sequential implementation with focused tests and one fresh whole-branch reviewer per slice, which avoids excessive per-task context cost. Follow the workflow rules applicable in Claude; preserve required approval gates, and do not repeat the already-granted direction/specification approval.

## Project and checkpoint

- Actual project: `D:\projects\AI_studio`, Git branch `master`.
- Codex workspace and prior media evidence: `C:\Users\meyaa\Documents\Codex\2026-10-10\d-projects-ai-studio`.
- Existing specification/research commit: `4f05956 docs: specify local creative workflows and worker readiness`.
- Previous retained-workflow commit: `de8254d docs: retain verified editing workflow`.
- At the start of handoff packaging, the sole existing dirty tracked file was `EDITING_WORKFLOW.md`: eight lines retaining the user's 5–7 minute target and shared-review approach. Handoff packaging also adds this handoff and the two clearly marked draft plans. Run `git status`/`git log` for the final checkpoint; commit only task-owned changes.
- No active research subagents remain. No render/install/download was started for this task. All shell calls were noninteractive and completed; do not terminate the user's unrelated terminals or processes.

Completed research was delegated to editing/animation, image/video and audio researchers, then synthesized into the report. Do not repeat broad internet research. Verify changing installation instructions and exact model/runtime versions only when needed.

## What the code already provides

Studio 0.3.0 uses Python >=3.12, Typer, Pydantic, NumPy, OpenCV, rembg and faster-whisper. Reuse `studio/ops.py`, `generate.py`, `jobs.py`, `media.py`, `speech.py`, `qc.py`, `devices.py`, `models.py`, `agent.py` and `sched.py`. CLI is the source of truth; the optional local UI calls its same job functions.

Existing editing includes overlays/text/subtitles/transitions, grading/LUTs, masking/restoration and audio cleanup/mixing/ducking. The runner normally writes one lossless intermediate per operation, then final encodes; completed steps are cached. The proposal is packaged recipes and an explicit bounded `compose` operation, not a second editor or general graph optimizer.

Existing generation supports `procedural`, `command` and `comfyui` adapters. **Only procedural is currently configured.** `STUDIO_GEN_CMD` was absent and ComfyUI unavailable. Procedural produces gradient plates, not semantic AI output. `_comfyui` ignores `init`, fills only basic placeholders, selects the first returned video/GIF/image and does not support standalone audio, robust final-node selection or owned cancellation. Visual `generate` currently drops generated audio during conformance.

The job runner assumes visual `.mkv` step outputs and video-oriented QA/final encoding. Add explicit generated-audio handling rather than forcing audio through `0:v`. Cache keys currently hash file parameter paths but can miss changed font/LUT/ASS/workflow contents at those paths. Original-source overwrite protection exists; direct generation and existing-delivery protection need the approved fixes.

## Runtime and hardware facts

- GPU: **AMD Radeon RX 9060 XT, 16GB**. Verified `HardwareInformation.qwMemorySize=17095983104`. Do not infer VRAM from the truncated 32-bit `Win32_VideoController.AdapterRAM` value.
- AMD AMF encoding, including native HEVC Main10 HLG, has worked. This does **not** prove neural model inference compatibility.
- Working bundled runtime: `C:\Users\meyaa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, Python **3.12.14**.
- Project `.venv/pyvenv.cfg` references a missing Windows Store Python 3.12.10 installation. Normal `.venv\Scripts\python.exe` / `studio.exe` are broken. Restore the normal entry point safely, retaining packages/caches and a rollback backup; do not delete/rebuild the existing environment blindly.
- The temporary bootstrap adds `D:\projects\AI_studio` and `D:\projects\AI_studio\.venv\Lib\site-packages` to `sys.path`, disables bytecode and uses task-local caches. Its reference is `work\studio_doctor.py` in the Codex workspace; **do not run that whole script**, because it imports a prior doctor-specific operation and invokes the CLI. Reuse only the prefix before `import doctor_motion` if needed.
- Neural workers belong in separate pinned environments. Current ComfyUI documentation describes AMD Windows gfx1200 support; verify the exact compatible Windows/Python/PyTorch/ROCm versions before setup. Do not change drivers, registry settings or system Python as an incidental install step.

The only new baseline test run was `tests/test_phases.py::test_generate_without_sources`: **1 passed in 1.27s**, using bundled Python plus the existing site-packages bootstrap and a fresh workspace-contained test directory. It exercises actual procedural generation and a fake command adapter; it proves no neural readiness. No full suite or speed benchmark has been run for the proposed upgrade.

## Recommended local workers

The report contains sources, licenses and limitations. Candidates are **uninstalled and unbenchmarked**:

- Kokoro-82M ONNX: first CPU narration worker. Official wrapper example uses `Kokoro(model_path, voices_path).create(text, voice=..., speed=..., lang=...)` and writes returned samples at their native rate. Verify pronunciation and punctuation.
- FLUX.2 klein 4B: first ComfyUI image/reference-edit pilot. Validate AMD inference and memory on this card before claiming ready.
- Wan2.1 1.3B / VACE 1.3B: short experimental video/reference-mask workflows. Native output is low-resolution synthesis; upscale/interpolation does not make it native 4K60 or guarantee original source fidelity.
- Stable Audio 3 Small Music/SFX: official Windows CPU LiteRT worker. Use `optimized/tflite/scripts/sa3_tflite.py` directly via argv, with `--dit sm-music` or `sm-sfx`, `--decoder same-s`, `--prompt`, `--seconds`, `--seed`, `--out`, `--threads`. Explicitly record precision/model version and review the output. Check model access/license dependencies; no API/cloud substitution.

Latest targeted documentation checks during planning:

- ComfyUI source now offers `POST /api/jobs/<job-id>/cancel` and prompt-targeted interruption. Pin and test the installed server; never assume an older server honors a targeted interrupt, and never globally cancel another job.
- Official FLUX klein guide: `https://docs.comfy.org/tutorials/flux/flux-2-klein`.
- Official Wan guides: `https://docs.comfy.org/tutorials/video/wan/wan-video` and `https://docs.comfy.org/tutorials/video/wan/vace`.
- Stable Audio Windows worker: `https://github.com/Stability-AI/stable-audio-3/tree/main/optimized/tflite`.
- Kokoro example: `https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py`.

Do not install every candidate at once. Finish one bounded working output/review before marking its kind ready. Unavailable dependencies must be accurately reported; configuration alone is not readiness. Build missing integrations, not a new neural foundation model.

## Draft-plan review notes

Before treating the drafts as executable:

1. Complete self-review and the third plan; align cross-task interfaces and cover every approved specification requirement. The two drafts were saved immediately on the stop request.
2. Keep narration duration measured from synthesis. The legacy `generate_media` default is five seconds: do not accidentally trim voice output to that visual default. Only pass duration controls a selected worker actually supports.
3. Define the generation result sidecar fields consistently. `generate_media` must still return a path, while smoke evidence needs kind, actual device, native properties, output hash and elapsed time.
4. Preserve identical completed-job reruns without overwriting a delivery; differing requests targeting existing outputs must fail safely. The drafts propose atomic no-replacement hard-link publication, which must report unsupported filesystems clearly.
5. Test literal text/ASS escaping, layer boundary semantics, same-path resource changes, no-audio input, VFR/rotation/HDR and ambiguous audio correlation. Default font/model/runtime dependencies must affect hashes.
6. Keep default CLI/profile compatibility, update audio-only cache/QA/UI callers together, and ensure package data survives wheel installation.
7. Calibrate the actual cleanup filter delay. The prior observed 25ms delay is not a universal hardcoded rule.

## User requirements that must survive the handoff

- Use GPU rendering and retain original video quality, framing, background detail and color precision. GPU encoding and GPU inference are different capabilities.
- Do not zoom the base picture, blur branches/leaves, add highlighted outlines or flatten textures during selective edits.
- Preserve the retained still-options/approved-look workflow and delegated native frame reviews for media delivery.
- Optimize repeated decoding/transcription, custom per-edit scripts and review coordination; **do not reduce careful editing or multiple reviews** to meet the latency goal.
- Comparable basic approximately 62-second, ten-popup edits should aim for **5–7 minutes including inspection, preparation, render, QA, multiple reviews and delivery**. This has not been achieved by a measured benchmark. Report actual phase times; generation has its own timing budget.
- Deliver final media to `D:\edited(studio)` unless a new task selects another destination. Never overwrite originals or old approved deliveries.
- Close task-owned processes/terminals when done; keep the user's own terminals intact. Use hidden/noninteractive process creation.
- Keep the solution minimal: reuse current operations/helpers, install only required dependencies, leave focused checks for nontrivial logic and commit only the work actually verified.

## Prior footage evidence worth reusing

The earlier nail-shape edit is already complete. Do not redo or replace its delivery:

- Original: `D:\footage\IMG_3532.MOV`, SHA256 `7e77980a566944074d2cb687396d492dc4c331569aee284414cf753ceb16de48`.
- Native display 2160x3840, coded 3840x2160 with -90 rotation, 10-bit HLG/BT.2020, 1,875 frames, approximately 62.495s, source time base 1/600.
- Three source images: almond.jpeg, oval.jpeg, square.jpeg in `D:\footage`.
- Delivered native master: `D:\edited(studio)\Nail_shapes_popups_4K_HDR_GPU.mp4`, SHA256 `2ccbcef079e0b16133c94fca9805adb55cfe23ff35430f4ae3f9f79fae067887`.
- Separate SDR viewing copy and `Nail shapes popup review` evidence already exist beside the master.
- Reusable references in the Codex workspace: `work/nails_motion.py`, `prepare_nails_edit.py`, `verify_nails_master.py`, `package_nails_result.py`, `review_nails_samples.py`, plus assets/cues/review evidence. They contain source-specific geometry; do not copy it blindly into another edit.
- Previous final composition + GPU encoding: 89.14s + 25.28s; total inspected-to-delivered elapsed approximately 39 minutes. Ten recorded exports across samples/masters/viewing copies took approximately 6m43s combined. These are historical logs, not the new benchmark.
- All 1,875 lossless background-body crops were identical to source YUV10; compressed master samples had minimum approximately 52.8dB luma PSNR. Final audio lag was -2 samples at 48kHz after calibration. HLG retained; Dolby Vision dynamic metadata and Apple auxiliary tracks were not preserved. The previous edit did not include a continuous full listening pass; the new acceptance specification adds one.

## Continuation directive

Read this handoff, the approved specification, the research and the two draft plans. Continue from implementation-plan completion and review, without repeating the completed broad research or asking again for direction/spec approval. Then implement and verify the three slices in order: reusable native-quality workflow and speed benchmark; reliable local generation/audio contracts; actual local image/video/voice/music/SFX workers. Follow applicable plan-review rules, report unavailable workers honestly, preserve source quality and full reviews, commit verified task-owned changes, deliver real samples and benchmark evidence, and close every process/terminal you opened when finished.
