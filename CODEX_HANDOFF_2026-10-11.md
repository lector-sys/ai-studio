## Latest task: paused Ronin animation repair

On 11 October 2026 at 19:35 IST the user requested a Claude handoff and pause. Read [LAST_RONIN_CLAUDE_HANDOFF_2026-10-11.md](LAST_RONIN_CLAUDE_HANDOFF_2026-10-11.md) first for the current task. No finished Ronin video exists; CPU decoding confirms corruption is upstream of the VAE. The prepared split/upcast-attention comparison was interrupted for the handoff. All owned inference/watch processes are stopped, port8188 closed, adapters OK and no driver reset observed. The completed worker-integration state below remains background.

# AI Studio continuation state — 11 October 2026

Resumed Claude's handoff from `f13b2db`; implementation is on `codex/local-worker-continuation` in `D:/projects/AI_studio`. Read git log for final commit identity. No need to repeat research, specification approval, slices 1/2 or worker installation.

Approved spec: `docs/superpowers/specs/2026-10-11-local-creative-workflows-design.md`. Reviewed plan: `docs/superpowers/plans/2026-10-11-plan-review-and-worker-plan.md`. Original Codex→Claude handoff: `CLAUDE_HANDOFF_2026-10-11.md`.

## State

- Slice 1 workflow foundation done: compose, seven reusable recipes, cached review bundles, no-overwrite delivery, HDR/source-clock preservation. Ten-popup benchmark cold median 207 s, warm 125 s; listening pending.
- Slice 2 generation contract done: isolated command/loopback Comfy adapters, designated outputs, owned cancellation, fingerprints, native provenance and technical QA.
- Slice 3 bounded local-worker pilot complete with honest limitations: Kokoro voice and Stable Audio music/SFX technically reviewed, listening pending; FLUX text-to-image passed a native 1024 pilot; reference edit remains experimental; Wan outputs decode but fail visual quality, so video remains experimental. VACE/reference-video not installed.
- Inherited lowvram FLUX actually finished at 597.84 s. GPU text encoding/full offloading and stock tiled native decode now work; final1024 image 228.39 s (cold reload plus concurrent hashing). Reject seed 9 six-digit anatomy and selective-reference skin/cuticle regeneration. BF16 Wan832×480 waterfall 164.01 s failed grid/neon artifacts; FP32 decoder 455.30 s also failed severe mosaic. Do not repeat this setup to chase a universal readiness claim.
- Fresh whole-branch review: no Critical; both Important fixes proved RED→GREEN (Wan preflight bounds/native frame grid; owned watchdog expiry). Whole suite 117 passed 121.40 s. Reference width/height field declaration is a deferred Minor; documented native ~1 MP aspect output is derived from init.

## Files and runtime

`docs/LOCAL_WORKERS.md` gives exact local launch/stop procedure. `docs/research/2026-10-11-local-worker-smoke-results.md` contains timings, artifact hashes, failures, review coverage, rulings and cleanup proof. `workers/comfyui/requirements.lock.txt` pins installed dependencies. `workers/comfyui/wan-models.json` records all three locally verified Wan hashes, sizes and official repository revision.

ComfyUI v0.39.2, Python 3.12.10, PyTorch 2.9.1 + ROCm 7.2.1, RX 9060 XT 16 GB, unchanged driver 26.7.1. Isolated environments, model weights, wheels, downloaded app and workspace results are intentionally ignored. Store Python starts a hidden-command-line child; protect the verified actual listener PID with `gpu_watch.ps1 -AlsoStopPid`. The watchdog snapshots birth times, stops only owned inference on alerts/expiry, and preserves unrelated AMF jobs. Keep 14,000 MiB cap. No Vulkan, registry/driver changes or reboot.

Task-owned ComfyUI child, launcher and watcher were closed and verified absent; localhost8188 is closed. CLI Comfy workers now report unavailable until an operator starts a new bounded matching runtime. Reviewed evidence stays on disk. Earlier three independent cap alerts were safety stops; no driver reset observed. Leave unrelated user terminals alone.

## Persistent user requirements

Local first, no cloud/paid fallback. Preserve original framing, texture, colour precision and resolution for footage; use retained deterministic recolour for faithful parrot-style changes. Neural generation cannot guarantee identity/detail preservation. Use GPU for renderer and these neural visual pilots; voice/music/SFX are measured CPU integrations. Delegate native content/frame reviews, keep human listening explicitly pending, never overwrite originals/deliveries. Deliver into unique folders under `D:/edited(studio)`. Close task-owned processes/terminals at task end; commit verified work only.

Next work should come from a concrete creative request. Use the existing recipes/workers and current quality gates. Wan needs a separately scoped correctness investigation before realistic production use; cause of artifacts remains unresolved. No native 4K/60 or all-creative-tasks readiness claim.

