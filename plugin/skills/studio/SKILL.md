---
name: studio
description: >
  Local AI video studio. Use for ANY video or audio editing request: trim/cut/concat, resize/crop,
  speed, color grade, titles/overlays/logos, audio mix/loudness, background removal/replacement/blur,
  person/object masks and tracking, transcripts/captions/subtitles, silence/filler removal, rough cuts,
  stabilize/denoise/upscale/frame interpolation/face restoration, object removal, generative B-roll,
  4K60 masters, social vertical exports, or finding footage by what was said. Also when the user says
  "studio", "AI studio", or mentions .mp4/.mov/.mkv files they want changed.
---

# AI Studio

The studio is a local CLI. Run it as:

```
D:\projects\AI_studio\.venv\Scripts\studio.exe <command> ...
```

Before your first command in a session, **read `D:\projects\AI_studio\AGENTS.md`**. It is the operating
manual: workflow, manifest format, the rules (preview before render, never overwrite sources, source-pixel
coordinates, GPU stays opt-in). `studio ops` lists every op with its params; `studio --help` lists commands.

Short version:
1. `studio inspect <file>`: never assume fps, resolution, color or audio layout.
2. Write a job manifest (example: `D:\projects\AI_studio\examples\background_replace.json`), then
   `studio check job.json`.
3. `studio preview job.json` (540p proxies), read `validation` and mask `low_confidence`, adjust.
   `studio auto job.json` does preview -> refine -> render on its own.
4. `studio render job.json`, then report `output`, `validation.ok` and `log` to the user.

For popups, lower thirds, kinetic captions, picture-in-picture, transitions, looks and narration mixes, start
from a recipe: `studio recipes`, then `studio recipe <name> --request req.json -o job.json` (expands to a
checked manifest). After rendering, `studio review <log> -o <dir>` builds the shared review bundle.

For colour/look/style requests: mock up 2–3 interpretations as stills (full frame + 100% crop) and
ask the user to pick before rendering (see "Creative edits" in AGENTS.md). Real footage must look photoreal.

Every command prints JSON on stdout; exit code 1 means the op or its validation failed.
Do not enable GPU providers (`STUDIO_ORT_PROVIDERS`, `STUDIO_HW_ENCODE`) without the user's OK.
