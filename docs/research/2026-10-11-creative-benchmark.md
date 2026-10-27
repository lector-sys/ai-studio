# Creative toolkit benchmark: ten-popup edit, 11 October 2026

**Source:** `D:\footage\IMG_3532.MOV`, SHA-256 `7e77980a…de48`. It is 2160x3840 display (coded 3840x2160, rotated -90), 10-bit HLG BT.2020, 1,875 frames, 62.49 s, time base 1/600. The hash was verified unchanged after every run.

**Edit:** `studio recipe image-popup` with `examples/creative_benchmark/request.json`:
- ten photo cards with the approved nail-shape look (graphics white 500 nits), using the prior cue intervals;
- slide/scale/fade animation;
- delay-compensated dialogue cleanup (high-pass 85 Hz, afftdn nf -55 nr 10 tn gs 12);
- `native-hdr-master` output via AMD AMF, HEVC Main10 at constant QP 10/12, run under `tools/gpu_watch.ps1` with no alerts.

**Machine:** i7-14700K, 32 GB, RX 9060 XT 16 GB.

## Complete runs: inspection to verified delivery

| Run | Cache | Inspect | Prepare | Render | QA bundle | Reviews (4 delegated, parallel) | Deliver | **Total** |
|---|---|---|---|---|---|---|---|---|
| run-2 | cold | 2.7 s | 1.1 s | 69.3 s | 53.3 s | 78.0 s | 0.9 s | **207.1 s** |
| run-3 | cold | 1.9 s | 1.1 s | 66.3 s | 52.7 s | 80.1 s | 0.9 s | **204.8 s** |
| run-4 | cold | 1.8 s | 1.1 s | 67.2 s | 54.5 s | 85.1 s | 0.9 s | **212.4 s** |
| run-5-warm | warm | 1.9 s | 1.0 s | 45.4 s | 54.4 s | 0.6 s (reused*) | 0.9 s | **125.2 s** |

The cold median is **207.1 s (3 min 27 s)**, with a range of 204.8–212.4 s. The 5–7 minute target is met with the same quality settings and the full review set. No runs paused for a user choice.

\*The warm-run master is byte-identical (SHA-256 `0a39adbc0d290aa6…`) to the cold runs, so its four reviews reuse run-4's records and are labelled as reused.

**Tokens:** no reliable counter was available, so this is recorded as unavailable. Each reviewer used about 50–110k subagent tokens.

## Quality evidence (identical across runs 2–4)

- **Clock:** all 1,875 frame timestamps equal the source's (max deviation 0 ticks at 1/600).
- **Untouched pixels:** outside the card boxes, every frame of the lossless intermediate is bit-identical to the source (YUV 10-bit checksums).
- **Final encode:** PSNR-Y is 54.35 dB on average and 52.70 dB at minimum (the previous delivery's minimum was 52.8 dB).
- **Audio:** 2,999,680 samples, the same count as the source. Lag is -1/-1/-2 samples (start, middle, end), with correlation 0.92–0.94. There are 0 clipped samples. Loudness is -29.7 LUFS against -29.6 LUFS in the source.
- **Determinism:** runs 2–4 produced the same master hash.
- **Reviews:** independent delegated reviewers (early frames/framing, late frames/moving edges, background detail, playback overview) all passed in runs 2–4. Findings are in `run-N\review\reviews.json`.

## Not covered

- **Listening pass:** not performed. An agent cannot hear audio, and the waveform/number review is not a substitute. This pass is the user's.
- **Cue gaps:** the cue data leaves a one-frame gap between back-to-back cues (17.63→17.66 s and 18.63→18.66 s). This comes from the approved cues, not the renderer.
- **Not preserved:** Dolby Vision dynamic metadata, and the iPhone spatial-audio (APAC) track, which has no FFmpeg decoder. The first (AAC) audio track is kept.

## Separate ops vs compose (render phase only)

Ten chained `overlay` steps plus `audio_clean` (`separate_ops_render.json`) took **305 s**: 11 full 4K lossless passes and about 40 GB of intermediates. The cards were also not colour-managed into HLG, so that output is not a valid delivery and was deleted. `compose` took 66–69 s in one pass with correct HDR cards, so it is kept.

## run-1 (superseded)

Run 1 used the compose code from before commit `e5d2a95`. Delegated review found two defects: cue entrances were 1–2 frames late, and first enabled frames were transparent. Both were reproduced and fixed, and regression tests were added. Run 1 is kept only as evidence (see `run-1\SUPERSEDED.txt`).
