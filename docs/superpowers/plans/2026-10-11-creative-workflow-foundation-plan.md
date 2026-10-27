# Creative Workflow Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native-quality popup, caption, transition and audio edits reusable, safe and measurably faster through the current Studio editor.

**Architecture:** Add packaged recipes that expand to existing version-1 manifests and a bounded `compose` operation for compatible layers/audio. Repair runtime and cache correctness in place, then reuse one review bundle and benchmark the complete edit. This is slice 1 of 3; the generation-contract and local-worker plans follow it.

**Tech Stack:** Python 3.12+, Typer, Pydantic 2, FFmpeg/FFprobe, installed Pillow, AMD AMF; pytest and stdlib helpers.

**Spec:** [Approved local creative workflows design](../specs/2026-10-11-local-creative-workflows-design.md)

**Handoff status:** Incomplete draft preserved at the user's request to stop Codex and hand off to Claude. Not self-reviewed or user-approved; the plan set and execution method remain pending. Check the handoff's review notes before use.

## Global Constraints

- Retain the existing Studio CLI, version-1 job manifests, operation registry, cache, GPU renderer, review workflow and output safeguards.
- The user's target is **5–7 minutes for a comparable basic edit, including careful editing, multiple reviews and verified delivery**.
- Neither fewer reviews nor lower-resolution masters qualifies as an optimization.
- Styling must not move or rescale the underlying footage unless the request calls for it.
- A frame-rate conversion, crop or SDR conversion is an explicit creative/export decision.
- Originals and existing deliveries cannot be overwritten by either direct generation or job execution.
- Keep user's pre-existing terminals intact.
- Use source-pixel geometry, rational time bases and half-open cue ranges. Editor dependencies stay separate from model-worker environments. Run commands without visible terminals.

## Review Focus

- Font/LUT/subtitle contents change at an unchanged path: invalidate the affected cache; covered by Task 1.
- A destination is created by another process during encoding: refuse publication and leave that file intact; covered by Task 1.
- VFR/rotated HDR source with a silent audio track: preserve geometry, timestamps, transfer and native precision; covered by Tasks 2 and 5.
- Literal braces, commas and backslashes in labels or a Unicode asset path: render the supplied content without expression interpretation; covered by Tasks 2 and 3.
- A noisy or silent source makes delay correlation ambiguous: mark measurement inconclusive rather than invent sync evidence; covered by Tasks 4 and 5.

## File map and contracts

All paths are relative to `D:\projects\AI_studio`. Existing modules keep their responsibilities.

| File | Responsibility |
|---|---|
| `tools/repair_runtime.ps1` | Back up and repair the stale venv base, verify the normal CLI |
| `studio/media.py`, `studio/jobs.py` | Asset hashes, source clock information, verified cache hits and safe output publication |
| `studio/compose.py` | Validated source-preserving layer composition; imported for op registration |
| `studio/recipes.py`, `studio/recipes.json` | Catalog, request expansion, cached static cards and ASS cues |
| `studio/cli.py`, `studio/agent.py` | Recipe commands, compose discovery and preflight |
| `studio/ops.py`, `studio/qc.py` | Shared audio cleanup timing and source/audio checks |
| `studio/review.py`, `tools/benchmark_creative.py` | Cached review bundle and complete-edit timings |
| `pyproject.toml`, `README.md`, `AGENTS.md`, `EDITING_WORKFLOW.md`, `plugin/skills/studio/SKILL.md` | Distribution and retained operating guidance |
| `tests/test_creative_foundation.py` | Focused real-media and trust-boundary regression tests |

New functions used across tasks:

```python
# media.py
def publish(tmp: Path, destination: Path) -> None: ...
def frame_clock(path: str) -> dict: ...  # time_base, pts: list[int], duration_ticks

# jobs.py
def resource_hashes(params: dict) -> dict[str, str]: ...

# compose.py
def validate_compose(ins: list[str], params: dict, scale: float = 1.0) -> dict: ...

# recipes.py
def listing() -> dict: ...
def expand(name: str, request: dict, asset_dir: Path) -> jobs.Job: ...

# qc.py
def audio_metrics(path: str) -> dict: ...
def audio_alignment(reference: str, output: str) -> dict: ...

# review.py
def bundle(job: jobs.Job, result: dict, root: Path) -> dict: ...
```

The signatures above are interface declarations, not implementation placeholders. Each is implemented and checked by the owning task below. Existing `media.info`, `media.sha256`, `jobs.run`, `Ctx.encode` and `qc.validate` remain the shared entry points.

### Task 1: Restore normal execution and correct cache/publication safety

**Files:** Create `tools/repair_runtime.ps1`; modify `studio/media.py`, `studio/jobs.py`; test `tests/test_creative_foundation.py`.

**Interfaces:** Produces `publish`, `resource_hashes`; uses existing `sha256`, `_key`, `op_version`, `Job` and `run`.

- [ ] **Step 1: Add failing trust-boundary tests.** Include the following tests, plus a cached subtitle render whose subtitle bytes are replaced at the same path and a corrupt cached intermediate that must rerender.

```python
from pathlib import Path
import pytest
from studio import jobs, media

def test_resource_content_changes_cache_key(tmp_path):
    font = tmp_path / 'typeface.ttf'
    font.write_bytes(b'first')
    before = jobs.resource_hashes({'layers': [{'font': str(font)}]})
    font.write_bytes(b'other')
    assert before != jobs.resource_hashes({'layers': [{'font': str(font)}]})

def test_publish_preserves_existing_destination(tmp_path):
    tmp, dst = tmp_path / 'out.part.wav', tmp_path / 'out.wav'
    tmp.write_bytes(b'new'); dst.write_bytes(b'user file')
    with pytest.raises(FileExistsError):
        media.publish(tmp, dst)
    assert dst.read_bytes() == b'user file'
```

- [ ] **Step 2: Run the failing tests.** Use working bundled Python with the existing site-packages bootstrap until the venv works. Run `pytest tests/test_creative_foundation.py -q -k 'resource or publish or cache'`; expect missing helpers or stale-cache failures. Set `HF_HUB_OFFLINE=1`, disable bytecode and cache-provider writes, and use a fresh workspace-contained basetemp.

- [ ] **Step 3: Repair and implement narrowly.** Save `.venv/pyvenv.cfg` to a timestamped backup under the ignored workspace. The repair script takes `-Python` pointing to a verified Python >=3.12, confirms its version, updates `home`, `executable` and `version` to that base, then verifies `.venv/Scripts/python.exe -c "import studio,typer,pydantic"` and `.venv/Scripts/studio.exe --help`. Restore the backup on failure. Do not recreate or delete packages, assets or the old environment. The tested base is `C:\Users\meyaa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.

For resource hashing, recursively traverse dicts/lists and hash existing files in documented file fields `file`, `font`, `lut`, `template`, `workflow`, `cue_file`; reject a supplied missing path. Model names are not file paths. Normalize resolved paths; include the mapping in the step cache key and sidecar. Fonts selected by defaults are resolved into recipe parameters. Add worker/model config hashes in slice 2.

Use this publication core for same-directory temporary files; no source file is ever passed as the temporary file:

```python
def publish(tmp, destination):
    tmp, destination = Path(tmp), Path(destination)
    if tmp.resolve() == destination.resolve():
        raise ValueError('temporary file equals destination')
    try:
        # Same-directory hard link gives atomic creation with no replacement.
        import os
        os.link(tmp, destination)
    except FileExistsError:
        raise FileExistsError(f'output already exists: {destination}') from None
    tmp.unlink()
```

On a filesystem without hard-link support, fail clearly and leave the old destination intact; do not degrade to an overwrite race. Encode finals to a temporary sibling with the proper extension, validate it, then publish. Store a final-job fingerprint and hash in existing logs. Repeating the identical completed job can return its hash-verified output and recorded validation; a different request targeting an existing file fails before expensive rendering. Cache hits must match their sidecar hash; corrupt/unreadable entries are discarded only within the known cache directory and rebuilt.

- [ ] **Step 4: Verify safety and compatibility.** Run the targeted tests, `studio --help`, `studio ops`, `studio check examples/background_replace.json` and the existing procedural/command generation test. Exercise final-output creation races by calling `publish` after a destination appears. Verify no tracked source/code changed during runtime repair.
- [ ] **Step 5: Commit the independently usable repair.** Stage only the task files and commit `fix: repair runtime and verify cached media publication`.

### Task 2: Promote the verified popup renderer into bounded compose

**Files:** Create `studio/compose.py`; modify `studio/jobs.py`, `studio/media.py`, `studio/agent.py`; test `tests/test_creative_foundation.py`.

**Interfaces:** Consumes `frame_clock`; produces registered `OPS['compose']` and `validate_compose`. Input 0 is the base; each layer's `input` is its integer index in `ins`.

- [ ] **Step 1: Add failing semantic tests.** Reuse fixture `d`, `lavfi` and `run` from `tests/test_studio.py`. Render a layer over `[0.5,1.0)` in a 30fps fixture and check frames 14, 15, 29 and 30. Add invalid NaN/infinite timing, end <= start, out-of-range asset index, negative dimensions and missing audio tests.

```python
from studio.compose import validate_compose

def test_compose_rejects_nonfinite_time(d):
    p = {'layers': [{'input': 1, 'start': float('nan'), 'end': 1,
                     'x': 10, 'y': 10, 'width': 80, 'height': 40}]}
    with pytest.raises(ValueError, match='finite'):
        validate_compose([d['clip'], d['logo']], p)

def test_compose_is_half_open(d):
    p = {'layers': [{'input': 1, 'start': .5, 'end': 1,
                     'x': 20, 'y': 20, 'width': 80, 'height': 40,
                     'animation': 'none'}], 'color_policy': 'preserve'}
    r = run(d, [d['clip'], d['logo']],
            [{'id': 'c', 'op': 'compose', 'inputs': ['s0','s1'], 'params': p}], 'compose-bounds')
    assert r['ok']
    frames = list(media.read_frames(r['steps'][0]['output'], 640, 360))
    assert frames[15][30,30,0] > 200 and frames[29][30,30,0] > 200
    base = list(media.read_frames(d['clip'], 640, 360))
    assert (frames[14][30,30] == base[14][30,30]).all()
    assert (frames[30][30,30] == base[30][30,30]).all()
```

Extend this fixture with a 10-bit HLG clip, `30000/1001` rate, VFR timestamps and rotation. Compare unchanged regions in the lossless intermediate as YUV samples; do not compare tone-mapped RGB or apply whole-frame SSIM over deliberate overlays.

- [ ] **Step 2: Run tests and establish the missing behavior.** Run `pytest tests/test_creative_foundation.py -q -k compose`; expect missing registration/function. Record supported FFmpeg formats/filters once.
- [ ] **Step 3: Implement the validated renderer.** Import `compose` from `jobs.py` to register it. Use the existing `work/nails_motion.py` implementation as a reference, removing fixed card sizes, source names, cues and profile monkey patches. Implement `frame_clock` with FFprobe integer `pts` and stream `time_base`; store coded/display geometry separately in `media.info`.

The timing expressions are bounded values from validation:

```python
enable = f"gte(t,{start})*lt(t,{end})"
ease = f"min(1,max(0,(t-{start})/{entrance}))"
scale_factor = f"({initial_scale}+(1-{initial_scale})*(1-pow(1-{ease},3)))"
y_expression = f"{y}+{slide}*pow(1-{ease},3)"
# One filter graph, one source input, deduplicated/split static assets.
clock_args = ['-fps_mode:v', 'passthrough', '-enc_time_base:v', source_time_base]
```

Support `none`, `fade` and `slide-scale` animation, validated finite opacity/scale/duration, numeric geometry, and already-rasterized text/cards. Inline `text` layers are converted to cached RGBA assets by `recipes.py` using literal text. Scale layer geometry with `Ctx.scale`. Do not pass user text into FFmpeg expressions. SDR uses the source's supported YUV format; HLG/PQ converts foreground sRGB through explicit `zscale` while extracting/rejoining alpha, preserving the base YUV10 picture. PQ/HLG brightness policy is explicit in metadata. Reject unsupported transfer/format combinations in `check` and at execution.

Use the source time base in lossless and final encode, without `fps` coercion. Keep the base frame count and audio when no audio treatment is requested. Add optional cleanup/mix through the shared audio helper in Task 4. Preserve source color tags, bit depth and orientation; detect HDR10/Dolby Vision metadata that the selected encoder cannot carry and report the limitation. Add a high-quality source-preserving GPU profile with the previously tested AMF 10/12 QP settings; keep current profile defaults compatible. Do not add compose to `CHUNKABLE` or mark it format-changing.

- [ ] **Step 4: Verify native samples.** Run boundary tests, SDR/HLG/PQ cases that are supported, a silent/no-audio input, Unicode assets and proxy geometry. Use original `IMG_3532.MOV` short early/middle/late samples for rotation, clock, foreground colors and unchanged background YUV crops. No full master until findings are resolved.
- [ ] **Step 5: Commit.** Commit `feat: compose timed layers while preserving source picture and clock` with only the task files.

### Task 3: Ship recipes and literal, cached creative assets

**Files:** Create `studio/recipes.py`, `studio/recipes.json`, `examples/creative_popup_request.json`; modify `studio/cli.py`, `pyproject.toml`, `README.md`, `plugin/skills/studio/SKILL.md`; test `tests/test_creative_foundation.py`.

**Interfaces:** Consumes `validate_compose`, `Job`; produces `listing` and `expand` with the signatures in the file map. CLI commands are `studio recipes` and `studio recipe <name> --request <json> -o <manifest.json>`.

- [ ] **Step 1: Add expansion and packaging tests.** Use Typer `CliRunner` for both commands. Confirm deterministic manifest/asset hashes, preserved request originals, rejected unknown fields and packaged catalog access. Test Unicode/quotes/braces in a label and overflowing text wrapping/truncation with an explicit reported policy.

```python
from studio.recipes import expand, listing

def test_popup_expands_to_one_compose_step(d, tmp_path):
    request = {'source': d['clip'], 'output': str(tmp_path/'out.mp4'),
               'items': [{'asset': d['logo'], 'text': 'Soft {shape}, C:\\nails',
                          'start': .5, 'end': 1, 'x': 20, 'y': 20,
                          'width': 200, 'height': 130}]}
    job = expand('image-popup', request, tmp_path/'assets')
    assert [s.op for s in job.steps] == ['compose']
    assert job.sources['base'].path == d['clip']
    assert job.steps[0].params['layers'][0]['start'] == .5
    assert 'image-popup' in listing()
```

- [ ] **Step 2: Run the failing recipe tests.** Run `pytest tests/test_creative_foundation.py -q -k 'recipe or popup or literal'`; expect missing module/commands.
- [ ] **Step 3: Build the seven small recipes from the spec.** `recipes.json` records names, description, required fields, defaults, capabilities and version; validate requests through focused Pydantic models with `extra='forbid'`. Keep expansion logic in ordinary Python functions, not evaluated JSON expressions. Cache each Pillow-rendered RGBA card by image bytes, text, resolved font bytes, dimensions, palette and recipe version; preserve EXIF orientation and fit images without stretching.

```python
from importlib.resources import files
catalog = json.loads(files('studio').joinpath('recipes.json').read_text(encoding='utf-8'))
sources = {'base': {'path': request['source']}}
steps = [{'id': 'compose', 'op': 'compose', 'inputs': list(sources),
          'params': {'layers': layers, 'color_policy': 'preserve'}}]
job = jobs.Job(name=name, sources=sources, steps=steps, output=request['output'])
```

Complete asset source entries before creating `steps`; assign each layer its input index. Lower thirds/callouts reuse static asset + compose; comparison/PiP uses existing resize/pad/overlay with explicit crop intent; transitions use `transition`; grading uses `color`/`stylize`; narration/music uses existing `audio` and `loudnorm` plus the generated-audio asset recipe after slice 2. For kinetic captions, consume reviewed word cues, emit an ASS file with a fixed safe style and `\k` centisecond durations, escape literal ASS braces/backslashes and use `subtitles`. No new transcription pass when a transcript is supplied. Invalid/unreviewed cue data fails expansion.

Add `recipes.json` to setuptools package data and declare direct Pillow usage from the installed version. Recipe expansion writes checked manifests atomically; it does not render. Update operator docs with exact example commands and the meaning of source-pixel placement.

- [ ] **Step 4: Verify each catalog entry.** Expand each initial recipe on small fixtures, run `studio check` and a small render, check cue order/boundaries, text legibility and face-safe placement manually. Install a wheel in a temporary environment with existing dependency access and verify `studio recipes` works without the source checkout. Keep the current UI manifest endpoints passing.
- [ ] **Step 5: Commit.** Commit `feat: add reusable creative recipes and cached card assets`.

### Task 4: Calibrate shared audio cleanup and audio metrics

**Files:** Modify `studio/ops.py`, `studio/qc.py`, `studio/compose.py`; test `tests/test_creative_foundation.py`.

**Interfaces:** Produces `audio_metrics(path) -> dict`, `audio_alignment(reference, output) -> dict`; shared `cleanup_audio(path, params, workdir) -> tuple[str, dict]` lives in `ops.py` and returns a filter string plus measured delay metadata.

- [ ] **Step 1: Add tests for impulse delay, empty/no-audio, clipping and ambiguity.** Generate a known 48kHz impulse/speech-like fixture and a truly silent fixture; test that silence yields `conclusive=False`, not zero measured lag. Test both direct `audio_clean` and compose use the same compensation.

```python
def test_silence_is_not_evidence_of_alignment(tmp_path):
    p = tmp_path/'silent.wav'
    media.ffmpeg(['-f','lavfi','-i','anullsrc=r=48000:cl=mono',
                  '-t',1,'-c:a','pcm_s16le',p])
    result = audio_alignment(str(p), str(p))
    assert result['conclusive'] is False
    assert result['lag_samples'] is None
```

- [ ] **Step 2: Run tests and measure filter delay.** Run the targeted audio tests. Calibrate the actual selected cleanup chain at the requested sample rate with a deterministic impulse; compare its onset rather than copying the old 1200-sample constant.
- [ ] **Step 3: Implement one shared cleanup policy.** Validate high-pass/denoise settings and sample rate; cache calibration by FFmpeg version, full filter parameters and rate. Flush filter delay, trim measured leading samples and retain exact intended sample count. Existing `audio_clean` and compose consume the same helper. Use source sample count/rate for unchanged timelines and keep PCM/FLAC intermediates.

```python
delay_seconds = delay_samples / sample_rate
chain = (f'apad=pad_dur={delay_seconds},{filters},'
         f'atrim=start_sample={delay_samples},'
         f'atrim=end_sample={wanted_samples},asetpts=PTS-STARTPTS')
```

`audio_metrics` returns duration, rate/layout, peak, clipped count and loudness where meaningful. `audio_alignment` decodes small normalized PCM windows at start/middle/end, runs bounded correlation, and returns per-window lag/confidence plus a conclusive flag; silence/weak/ambiguous matches cannot pass sync automatically. Preserve existing mixing/ducking and add two-pass loudness only when requested; record both passes. Listen for lost syllables, ringing and pumping.

- [ ] **Step 4: Verify.** Run focused tests and the existing audio op cases. On the original speech fixture compare cleaned/native audio and the complete mix; sample alignment within 1ms when correlation is conclusive, no introduced clipping, and unchanged target sample count. Inspect ambiguous cases manually.
- [ ] **Step 5: Commit.** Commit `fix: preserve audio timing through shared cleanup and metrics`.

### Task 5: Share native review evidence and prove the latency target

**Files:** Create `studio/review.py`, `tools/benchmark_creative.py`, `examples/creative_benchmark.json`; modify `studio/qc.py`, `studio/cli.py`, `AGENTS.md`, `EDITING_WORKFLOW.md`, `README.md`; test `tests/test_creative_foundation.py`.

**Interfaces:** Produces `bundle(job, result, root) -> dict`; `studio review <result-log> -o <directory>` prepares evidence only. The benchmark tool consumes saved requests/jobs and appends phase timing/review results; it does not manufacture reviewer findings.

- [ ] **Step 1: Add clock, bundle reuse and benchmark accounting tests.** Use a VFR fixture with known integer PTS, rotated HDR sample and an edited copy with deliberate one-frame drift. Require native size/color/clock failure on the bad copy. A second bundle call with identical job/output hashes must reuse files; changing a cue or output hash must invalidate affected checkpoints.

```python
def test_benchmark_total_keeps_review_wait():
    from studio.review import measured_phases
    phases = [('inspect',0,2),('prepare',2,8),('render',8,25),
              ('qa',25,32),('review',32,70),('deliver',70,72)]
    result = measured_phases(phases, tokens=None)
    assert result['total_seconds'] == 72
    assert result['review_seconds'] == 38
    assert result['tokens'] is None
```

`measured_phases(phases: list[tuple[str,float,float]], tokens: int | None) -> dict` is implemented in `review.py`; validate finite monotonic times and reject impossible overlaps unless clearly marked parallel spans.

- [ ] **Step 2: Run the failing checks and prepare one shared sample decode.** Probe all requested sample PTS in one invocation where feasible; extract full frames and 100% crops once. Run `pytest tests/test_creative_foundation.py -q -k 'clock or bundle or benchmark'`.
- [ ] **Step 3: Implement reusable evidence and QA.** Bundle includes inspection/source hash, manifest/cue table, native frames/crops, entrance/exit neighbors, a named SDR viewing copy, audio windows, full-file decode result, frame-clock/rotation/color/precision comparison, region-specific lossless preservation checks and compression measurements, unresolved findings, provenance and timing fields. Store content hashes/settings so independent early/late/background/audio reviewers share evidence concurrently. A full playback/listening pass has a named reviewer and recorded findings; an unset review is visibly pending. Do not perform unnecessary repeated transcription or review after unchanged cache hits.

```python
review_record = {'kind': 'playback-and-listening', 'status': 'pending',
                 'reviewer': None, 'findings': []}
timings = {'inspect': None, 'prepare': None, 'render': None,
           'qa': None, 'review': None, 'deliver': None, 'tokens': None}
```

Add exact timestamp comparisons within one source tick for unchanged timelines. Record HDR metadata losses separately from tag preservation. Final delivery requires all applicable technical checks and resolved visual/audio findings; preserve the original output hashes and use unique benchmark names.

- [ ] **Step 4: Benchmark the representative edit and deliver evidence.** Fix source `D:\footage\IMG_3532.MOV` hash `7e77980a566944074d2cb687396d492dc4c331569aee284414cf753ceb16de48`, the three existing nail-shape images and the prior ten cue intervals/styles. Use the tested 2160x3840 10-bit HLG AMF 10/12 output quality. Compare separate ops and compose at equal visual/audio settings, with cold and warm cache runs; run at least three comparable runs per path and report median/range. Every timed result includes QA, review and delivery; cache-reused review evidence is explicitly recorded. Waiting for reviewers counts. Required user-choice pauses are separate. Claim 5–7 minutes only after measured passing results; if missed, report phase costs and fix the demonstrated bottleneck, never reduce review or master quality.

Run the relevant existing tests once after all foundation changes, then one whole-branch review using the selected execution method. Save final samples, recipe and timings under a uniquely named `D:\edited(studio)\Creative toolkit benchmark` folder; hash-verify copies. Finish any task-owned sessions and helpers. Update retained guidance with measured performance and current limits, not a universal speed claim.
- [ ] **Step 5: Commit.** Commit `feat: share creative review evidence and benchmark complete edits`.

## Coverage and handoff

This plan delivers specification slice 1, including runtime, recipes, bounded compose, cache/publication safety, source/HDR/audio preservation, review artifacts and the complete-edit benchmark. Slice 2 implements standalone audio and reference-aware worker contracts; slice 3 installs/verifies the selected local neural workers. Those requirements are mapped to the two accompanying plans, not claimed complete here.

Recommended execution: **Native** for all three plans. Shared cache, timeline, renderer and CLI interfaces make sequential implementation simpler; use focused tests and one fresh whole-branch reviewer after each slice. Native frame/audio review delegation remains part of media delivery, independent of code execution method.

Implementation starts only after the user reviews the plan set and chooses Native or Subagent-driven execution. Do not download models or run benchmark masters while this handoff is pending.
