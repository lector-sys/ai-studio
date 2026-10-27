# Local worker results â€” 11 October 2026

Continuation of Claude's slice-3 plan from `f13b2db`, using the existing installed checkout on `codex/local-worker-continuation`. Source editing and its 207-second ten-popup benchmark were not rerun. These are generation measurements, not edit benchmarks.

## Measured pilots

| Pilot | Worker elapsed | Native result / disposition |
|---|---:|---|
| Inherited FLUX lowvram | 597.84 s | 768Ã—768; completed before Claude's limit, five-nail still visually acceptable, encoder on CPU |
| GPU encoder with full offloading | 102.30 s | 768Ã—768; acceptable still, avoids retaining encoder beside diffusion weights |
| Reference polish change | 61.66 s | 1024Ã—1024; similar pose/backdrop, but skin/cuticle/reflection regenerated: needs work for selective edits |
| Different FLUX seed 9 | 56.59 s | 1024Ã—1024; six digits, rejected |
| Full-frame 1024 decode repeat | aborted | Watchdog stopped inference at 14,742 MiB; output not delivered |
| Final stock tiled FLUX, seed 7 | 228.39 s | 1024Ã—1024; root and independent native-image review passed five digits, no obvious tile seams/halos |
| Wan fox pilot | 154.13 s inference + 1.44 s cached save recovery | 320Ã—192, 33 frames at 16 fps; technically valid, weak gait and artificial fur: needs work |
| Wan tiled waterfall | 164.01 s | 832Ã—480, 33 frames at 16 fps; technical pass, visual FAIL: repeated grid texture, neon colour, crushed shadows and smeared water |

The final FLUX timing includes a cold reload and concurrent large-file hashing; it is not a clean speed benchmark. Earlier faster values used a different decoder configuration. The 1.44-second fox recovery reused already-generated cached latents after correcting SaveVideo's flattened dynamic-format fields; it is not generation speed.

FLUX's approved final still is `workspace/workers/smoke/flux-klein-image-20261011-182423.png`, SHA-256 `c15a77d344470799a5990568f0d8cefa217153e7680e166647132c1b750ec2bc`. Waterfall BF16 output is `workspace/workers/smoke/wan-video-20261011-183850.mp4`, SHA-256 `dbf2e4ad6091b6ae02cc01b9a8e8a70a1a005f6fdb8a65c123da88dfecb664d9`. Both provenance sidecars record original requests/native format; output hashes were verified.

The independent video reviewer inspected all 33 frames in contact sheets plus native selected frames. All 33 native files remain in `workspace/workers/review/wan` and `wan-480`; full decode/frame-count checks passed. Still-frame sequence review does not claim an actual playback/listening pass.

## Runtime and protection

ComfyUI v0.39.2; Python 3.12.10; PyTorch 2.9.1 + ROCm 7.2.1; RX 9060 XT 16 GB, installed driver 26.7.1. No driver/registry change, reboot, Vulkan or cloud fallback. Isolated requirements are pinned in `workers/comfyui/requirements.lock.txt`. Wan models were downloaded from official repository revision `123acf1cc74bccbb9bfff8ac1ee72edc08c2341d`; all three full SHA-256 hashes/byte sizes are in `workers/comfyui/wan-models.json`.

Use the launch flags and operator-owned child PID procedure in `docs/LOCAL_WORKERS.md`. Desktop VRAM is included in OS watchdog readings. Earlier normal-memory runs hit 14,249 MiB while retaining encoder weights, and 14,742 MiB during full-frame VAE decode. They stopped under the unchanged 14,000 MiB cap. No reset was observed, and adapter status stayed OK. Sampled final tiled image peak was 11,734 MiB; combined image/first-video watch peak was 12,403 MiB. Sampled peaks cannot guarantee no transient between samples.

The shared watchdog now snapshots owned PID birth times, avoids killing unrelated AMF encodes when explicit ownership is supplied, and stops owned inference on session expiry as well as alerts. Wan preflight rejects non-finite/non-positive fps/duration, untested rates/frame counts and unsupported `4n+1` latent lengths before contacting ComfyUI. No custom Comfy nodes or duplicate renderer were added.

## Readiness limits

- FLUX text-to-image: reviewed pilot at native 1024; every new image still needs content QA because another seed failed anatomy.
- FLUX reference editing: experimental for creative resynthesis. It cannot promise original skin/detail retention, and its final tiled reference variant has not been smoked. Native dimensions derive from reference aspect at ~1 MP; requested width/height do not control this pilot.
- Wan: experimental until an output passes content review. Real clips decode, but native 4K/60, faithful reference-video editing and VACE are not verified.
- Kokoro and Stable Audio: retain Claude's measured technical/ASR results (voice 4.75 s, music 51.77 s, warm SFX 3.08 s). Human listening is pending; readiness records are technical agent reviews. Music has a final ~2-second decay; SFX event count is not reliable.

## Final review and rulings

Fresh whole-branch reviewer: no Critical findings; two Important findings fixed with observed REDâ†’GREEN tests (Wan resource/frame limits; watchdog deadline ownership). The reference-width/height catalog declaration was a Minor metadata finding, deferred; native-size semantics are stated above. No broad changes to the unchanged generation adapter.

Rulings, in order:
1. Continue the installed checkout on a dedicated branch to preserve Claude's dirty files and large local runtime. Cost if wrong: less checkout isolation; master remains at the original base.
2. Use stock full model offloading and reserve 5 GB, disabling pinned/async offload. Cost if wrong: a bounded unsuccessful smoke; safety cap stays in force.
3. Use overlapping stock tiled decode at native dimensions after the full-frame VRAM failure. Cost if wrong: seams or slower decode, caught by output review.
4. Reference native sizing remains ~1 MP from the init image; no new conformance mechanism for this experimental pilot. Cost if wrong: requested dimensions require a separate explicit resize.
5. VACE/reference-video remains deferred and unadvertised, as allowed by the plan. Cost: those creative operations are unavailable.
6. Broad hardening of unchanged adapter/readiness internals is outside this slice; interaction with new configurations was reviewed. Cost: unrelated latent defects may remain.
7. Downloaded dependencies/model internals were outside code-review scope; package identities and Wan file hashes were separately checked by root. Cost: upstream runtime correctness and new prompts remain output-review dependent.

Deferred Minor: remove the reference worker's misleading width/height field declarations; no exact-size control is currently promised.


## Final decoder comparison and verification

The same waterfall prompt/seed with `--fp32-vae` produced 455.30 s elapsed at 832×480 / 16 fps / 33 frames. Full decode and native-size/count checks passed. Root and independent reviewer both **FAIL** it for realistic use: severe full-scene multicoloured mosaic/noise and temporal colour instability. This did not establish the underlying cause. It is diagnostic-only; increasing decoder precision did not fix quality and increased latency. Output: `wan-video-20261011-184545.mp4`, SHA-256 `667df37139401473a962156c81b386bc58111ece8a8a8cc8a7b354a607b168a2`. All 33 native frames are under `workspace/workers/review/wan-fp32`.

Final focused regressions: 10 passed. Whole suite after code fixes: **117 passed in 121.40 seconds**, offline, `pytest -q tests -p no:cacheprovider`. The strict Wan configuration permits at most 832×480, 16 fps, 33 frames, 2.0625 seconds and a 30-minute worker timeout; only `4n+1` lengths are accepted. These bounds constrain experimental requests; they do not certify visual quality.

At 2026-10-11T18:55:38.8447083+05:30, owned ComfyUI child 12116, launcher 37560 and watcher 41168 were stopped after verifying ownership/birth times. No process or localhost8188 listener remained. AMD and Intel adapter status was OK, and no driver-reset event was observed since 17:58. Shutdown proof remains `workspace/slice3_shutdown.json`. Future ComfyUI listings intentionally say unavailable while the local server is closed; retained image evidence becomes reusable when the matching bounded runtime is restarted.
