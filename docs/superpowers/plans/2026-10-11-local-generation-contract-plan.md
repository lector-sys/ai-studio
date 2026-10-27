# Local Generation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local visual and audio generation reference-aware, source-safe, observable and usable inside ordinary Studio jobs.

**Architecture:** Extend the existing dispatcher, subprocess and ComfyUI adapters; retain `generate_media` returning a path. Add audio asset steps and WAV delivery around the current runner, and record readiness from actual reviewed outputs. Implement after the foundation plan passes; no new inference engine or cloud routing.

**Tech Stack:** Existing Python/Typer/Pydantic/FFmpeg, stdlib subprocess/urllib/http.server, pytest fake workers; no neural packages in the editor environment.

**Spec:** [Approved design](../specs/2026-10-11-local-creative-workflows-design.md)

**Handoff status:** Incomplete draft preserved at the user's request to stop Codex and hand off to Claude. Not self-reviewed or user-approved; the local-worker plan has not been written. Check the handoff's review notes before use.

## Global Constraints

- Extend the existing generator dispatcher, command adapter and ComfyUI adapter rather than implementing an inference engine.
- Support kinds `image`, `video`, `voice`, `music` and `sfx`.
- Preserve existing `generate_media` callers and its path return value; keep richer provenance in the existing result/log sidecar.
- A neural request never falls back to the procedural generator.
- Local endpoints are used by default. No automatic cloud fallback or paid services are part of this design.
- Originals and existing deliveries cannot be overwritten by either direct generation or job execution.
- Lightweight capability listing does not load models or download assets.
- Native generated output remains available; conversion into the video timeline is separately recorded.

## Review Focus

- A command exits successfully without a decodable output of the requested kind: fail before publication; Task 1.
- An init path/output path contains Unicode, braces or shell metacharacters: treat it as literal data; Tasks 1 and 2.
- ComfyUI returns intermediate previews before the designated final output: wait/select only the configured final node; Task 2.
- Timeout/cancellation coincides with another user's job: cancel only the owned prompt, never global interrupt; Task 2.
- Model/workflow/runtime changes after a successful smoke: invalidate readiness and cached generation; Task 3.

## File map and interfaces

Modify `studio/generate.py`, `studio/jobs.py`, `studio/cli.py`, `studio/qc.py`, `studio/devices.py`, `studio/models.py`, `studio/agent.py`, `studio/sched.py`, `studio/ui.py`, existing docs and plugin skill. Create `studio/workers.py`, `studio/workers.json`, `tests/test_generation_contract.py`, `tests/fake_generator.py`, `examples/generate_audio.json`, `examples/reference_generation_request.json` and `docs/LOCAL_WORKERS.md`. Update package data in `pyproject.toml`.

```python
# generate.py: append keyword-only options, keeping existing positional callers valid
def generate_media(prompt, kind='image', out=None, width=1920, height=1080,
                   duration=5.0, fps=30, backend=None, seed=None, init=None,
                   *, references=None, mask=None, worker=None, voice=None,
                   workflow=None, output_node=None, timeout=None) -> str: ...

# workers.py
def worker_config(name: str) -> dict: ...
def request_check(config: dict, request: dict) -> dict: ...
def fingerprint(config: dict) -> str: ...
def readiness(config: dict, evidence_dir: Path) -> dict: ...
def record_smoke(config: dict, result: dict, review: dict, evidence_dir: Path) -> dict: ...

# qc.py: additive; existing video callers retain defaults
def validate_audio(out, duration=None, tolerance=None, decode=True) -> dict: ...
```

Worker configuration uses `name`, `backend`, `kinds`, `argv` or `workflow`, `bindings`, `output_node`, `output_type`, `models`, `limits`, `device`, `timeout` and `version`. Bindings map request fields to exact ComfyUI node/input locations; references map a named request asset to an upload and destination field. No arbitrary plugin loading/evaluated configuration language.

### Task 1: Add safe standalone audio generation and audio graph outputs

**Files:** Modify `studio/generate.py`, `studio/qc.py`, `studio/jobs.py`, `studio/cli.py`, `studio/ui.py`; create `tests/fake_generator.py`; test `tests/test_generation_contract.py`; create `examples/generate_audio.json`.

**Interfaces:** Consumes foundation `media.publish`, `audio_metrics`; produces extended `generate_media`, `validate_audio`, registered `generate_audio` and `wav` output profile. A `generate_audio` cache step is `.wav`; visual steps remain `.mkv`.

- [ ] **Step 1: Add a small fake command worker and failing tests.** The worker reads `--request <json>`, writes a PCM tone with stdlib `wave`, or deliberately writes corrupt/wrong-kind output according to the request. Test a pure voice job, voice->audio mix->video job, nonexistent output, corrupt WAV, overwrite protection, literal request text and a model-free procedural visual regression.

```python
def test_audio_only_validation(tmp_path):
    p = tmp_path/'tone.wav'
    media.ffmpeg(['-f','lavfi','-i','sine=f=440:r=48000:d=.5',
                  '-c:a','pcm_s24le',p])
    report = validate_audio(str(p), duration=.5)
    assert report['ok']
    assert media.info(p)['video'] is None
    assert media.info(p)['audio']['sample_rate'] == 48000

def test_procedural_cannot_fake_voice(tmp_path):
    with pytest.raises(ValueError, match='voice'):
        generate_media('Speak this', 'voice', tmp_path/'voice.wav', backend='procedural')
```

- [ ] **Step 2: Run failing audio-contract tests.** Run `pytest tests/test_generation_contract.py -q -k 'audio or voice or command'`; expect unsupported kind/audio-only QA failures. Fake workers do not require any model download.
- [ ] **Step 3: Implement the direct and graph paths together.** Keep procedural limited to image/video. The command worker receives `--request` and `--out` as argv items; retain `STUDIO_GEN_CMD` compatibility for existing visual callers. Use bounded subprocess timeouts, capture actionable stderr, hidden creation flags on Windows, and stop only task-owned subprocesses on cancellation. Generate to a correctly suffixed temporary sibling, probe/decode it, verify requested kind and dimensions/clock or rate/layout, then publish through the foundation helper.

```python
audio_kinds = {'voice','music','sfx'}
suffix = '.wav' if kind in audio_kinds else ('.png' if kind == 'image' else '.mkv')
if kind in audio_kinds and backend == 'procedural':
    raise ValueError(f'procedural backend cannot generate {kind}')
checks = validate_audio(raw) if kind in audio_kinds else validate(raw)
if not checks['ok']:
    raise RuntimeError(f'generated output failed validation: {checks}')
```

`validate_audio` checks actual audio presence, full-file decode, positive sample rate/channels, measured samples/duration, peak/clipping and optional expected duration. Silence is reported and reviewed according to kind; music/SFX intentional silence is not automatically a corrupt file. Extend the runner's explicit operation output-kind table for `generate_audio`; don't infer every step type from file suffixes. Audio-only jobs skip video/proxy assumptions and encode once to `wav` PCM; video jobs accept audio step outputs as inputs to existing `audio`/compose operations. Preflight rejects a video-only op fed audio. The visual generate operation retains available audio while conforming its clip; do not map only `[v]`. Fit or crop generated visuals only under declared conformance parameters, logging native/requested geometry and frame changes.

CLI `generate voice` accepts `--prompt` as text, `--worker`, `--voice`; music/SFX accept duration and seed; old image/video invocation stays valid. Keep `--fps` rational-capable for visual requests. Add audio outputs to the local UI's known allowed paths; no UI redesign or new server.
- [ ] **Step 4: Verify direct and graph callers.** Run fake-worker success/failure tests, mixed video fixture, audio-only preview/final policy, source/destination alias checks, and existing `test_generate_without_sources`. Confirm a successful command with wrong media kind is rejected and unpublished.
- [ ] **Step 5: Commit.** Commit `feat: support validated local audio assets and job outputs`.

### Task 2: Complete reference-aware ComfyUI output and cancellation handling

**Files:** Modify `studio/generate.py`; test `tests/test_generation_contract.py`; create `examples/reference_generation_request.json`.

**Interfaces:** Consumes `media.publish` and extended generation options. Keep `_comfyui` internal; output selection is an explicit `output_node` and media kind. Use a stdlib fake HTTP server in tests.

- [ ] **Step 1: Add fake-server tests.** Emulate `/upload/image`, `/object_info`, `/prompt`, `/history/<id>`, `/view`, `/queue` and targeted cancellation. The fake upload returns a renamed filename; assert the actual returned name appears in the correct bound node. Return both an early preview node and final node; assert only final is retrieved. Add structured errors with no outputs, wrong type, multiple final files without explicit index, external redirect, path traversal filenames, timeout and unrelated queued/running jobs.

```python
def test_select_only_designated_node():
    from studio.generate import select_output
    outputs = {'preview': {'images': [{'filename': 'draft.png'}]},
               'final': {'videos': [{'filename': 'done.mp4'}]}}
    picked = select_output(outputs, node='final', kind='video')
    assert picked['filename'] == 'done.mp4'
    with pytest.raises(ValueError, match='final'):
        select_output(outputs, node='missing', kind='video')
```

`select_output(outputs: dict, node: str, kind: str, index: int | None = None) -> dict` is a pure helper defined by this task, with only allowed media output collections/extensions.

- [ ] **Step 2: Run tests and compare official server behavior.** Run `pytest tests/test_generation_contract.py -q -k comfy`. Read the pinned server routes for targeted cancel support rather than assuming a global interrupt is safe.
- [ ] **Step 3: Implement bounded local API calls.** Validate loopback host and reject redirects to nonlocal hosts. Fill typed prompt/seed/dimensions/frames placeholders as today, adding actual upload results through explicit bindings. Upload image references and masks separately as expected by the workflow; video references require a declared supported loader/upload mechanism, otherwise fail preflight. Validate node classes and required model files before `/prompt`.

```python
# Pure binding: no string substitution into executable code.
for field, binding in bindings.items():
    node_id, input_name = binding['node'], binding['input']
    workflow[node_id]['inputs'][input_name] = request[field]
```

Record an owned prompt ID/client ID. Poll with `time.monotonic`, bounded requests and a finite deadline. Detect terminal error/interrupted status before interpreting outputs. Select only the configured final output; a legacy one-output workflow may still succeed without configuration, but ambiguity must fail clearly. Keep output extension constrained to the requested kind and native file available; stream download into a temporary file rather than loading arbitrary output into RAM.

Cancel through `POST /api/jobs/<owned-id>/cancel` when the pinned server supports it. On older servers, deleting only the owned pending queue item is safe; never call global `/interrupt`. A running job lacking proven targeted cancellation reports that limitation and does not cancel another person's work. Catch KeyboardInterrupt/timeout to perform this bounded owned cleanup, then return an actionable failure. Shut down a server only if this task launched it and no other user work depends on it.

- [ ] **Step 4: Verify adverse cases.** Run fake-server tests; ensure prompt errors return promptly rather than waiting for timeout, references bind uploaded names, no intermediate output is published, and cancellation requests identify exactly one prompt. Verify Unicode/literal filenames and wrong/ambiguous outputs. No model runs are necessary for these checks.
- [ ] **Step 5: Commit.** Commit `fix: bind local generation references and select owned final outputs`.

### Task 3: Make readiness, cache and preflight describe actual local workers

**Files:** Create `studio/workers.py`, `studio/workers.json`, `docs/LOCAL_WORKERS.md`; modify `studio/devices.py`, `studio/models.py`, `studio/agent.py`, `studio/jobs.py`, `studio/generate.py`, `studio/sched.py`, `studio/cli.py`, `pyproject.toml`; test `tests/test_generation_contract.py`.

**Interfaces:** Produces all `workers.py` functions in the file map. Extends capability JSON with a `workers` mapping while retaining existing `generators` booleans.

- [ ] **Step 1: Add readiness and dependency-hash tests.** Configure a fake worker with no files -> unavailable; add required files -> configured; record a valid technically checked and reviewed output -> ready; change model/config at the same path -> configured. Broken/missing output evidence cannot stay ready. Add unavailable-kind preflight and two jobs sharing a GPU worker test.

```python
def test_configuration_is_not_readiness(tmp_path):
    from studio.workers import readiness
    executable = tmp_path/'worker.exe'; executable.write_bytes(b'fixture')
    config = {'name':'fake', 'backend':'command', 'kinds':['voice'],
              'argv':[str(executable),'--request','{request}','--out','{out}'],
              'models':[], 'limits':{}, 'device':'cpu', 'version':1}
    state = readiness(config, tmp_path/'evidence')
    assert state['state'] == 'configured'
    assert state['smoke'] is None
```

- [ ] **Step 2: Run failing readiness tests.** Run `pytest tests/test_generation_contract.py -q -k 'ready or readiness or preflight or fingerprint'`; no heavyweight imports or downloads.
- [ ] **Step 3: Implement a small packaged registry and evidence file.** Package `workers.json`, with default candidates unavailable until configured. Merge optional workspace worker configuration by explicit name; reject unknown fields, shell command strings in new profiles, unsupported kinds, invalid limits, missing model paths and non-loopback endpoints. `fingerprint` hashes worker/profile/workflow/model contents plus resolved runtime identity/version and relevant node versions. Record actual inference device from worker/server, independently of the encoder/ORT provider.

```python
smoke = {'fingerprint': fingerprint(config), 'kind': result['kind'],
         'output': result['output'], 'sha256': media.sha256(result['output']),
         'validation': result['validation'], 'review': review,
         'device': result['device'], 'seconds': result['seconds']}
```

Readiness requires matching fingerprint, current files/discovery, passing technical evidence, a completed content review and a hash-valid output. Store per-kind tested dimensions/length/voice limits, date and benchmark. `studio capabilities` must not run synthesis; smoke is an explicit `studio workers smoke <name>` action, followed by a recorded review. Manifest preflight can allow configured worker tests with a clear unverified warning; ordinary advertised-ready requests require matching evidence. Reject unsupported model sizes/frame constraints before loading.

Add generation dependency fingerprint to step cache keys even when model/workflow paths stay unchanged. Extend existing GPU lock with an explicit worker GPU indicator so ComfyUI inference is serialized when needed, even if the editor's ORT setting is CPU. Do not treat `STUDIO_HW_ENCODE=1` as neural readiness or expand unmeasured model VRAM estimates into facts. Keep discovery read-only and quick.

Document worker setup/configuration, direct generation, reference/mask mappings, native output retention, audio recipes and exact unavailable reasons. Update CLI help, README and the source plugin skill together. Optional formats/workers from the research remain listed as unverified extensions, not ready features.
- [ ] **Step 4: Verify the integrated contract.** Run all generation-contract tests, existing schema/check/auto/UI tests and the foundation tests affected by audio/cache changes. Expand and render the narration/music recipe with fake WAV assets. Confirm ordinary CLI usage preserves path return values and a missing worker yields a clear nonzero result.
- [ ] **Step 5: Commit and review.** Commit `feat: report verified local worker readiness and generation provenance`; run one fresh whole-branch review for slice 2 using the chosen execution method. Leave real neural readiness to the third plan.

## Coverage and handoff

This plan covers specification slice 2: image/video reference contracts, explicit audio output path, direct callers and graph integration, command safety, designated ComfyUI outputs, errors/timeouts/owned cancellation, atomic publication, audio-only QA, accurate readiness, cache provenance and operator docs. Slice 1 supplies native preservation and review artifacts; slice 3 supplies actual model installations and smoke outputs.

Official API evidence: [ComfyUI routes](https://docs.comfy.org/development/comfyui-server/comms_routes) and [current server source](https://github.com/Comfy-Org/ComfyUI/blob/master/server.py). Pin the installed server version and test its targeted cancel route; older servers need a narrower cancellation policy.

Recommended execution is Native, sequentially after the foundation plan, with tests and a fresh whole-branch review. No model compatibility or speed claim is established by fake-worker contract tests.
