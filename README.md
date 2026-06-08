# Layout Squirrel

Automatically written prototype instructions. This is experimental code, and there is no guarantee that anything works exactly as described here.

![Layout Squirrel architecture diagram](docs/architecture.png)

Download the prototype Krita plugin zip:

[layout-squirrel-prototype-20260608.zip](release/layout-squirrel-prototype-20260608.zip)

## What It Does

- Layout Squirrel is a prototype for prompting image positions and object traits together.
- The user writes a natural-language layout note, such as `a pink fox flying over a blue gorilla`.
- The user writes region tags, such as `"fox, pink fur, flying", "gorilla, blue fur, sitting"`.
- An LLM turns those region tags into editable Krita vector regions.
- Krita AI Diffusion then uses those regions for regional prompting, so each region can carry its own prompt text.
- The user can move and resize the generated shapes before generating the image.
- The current prototype still requires separate layout text and region tags. A more complete version should eventually derive region tags automatically from a natural-language prompt, possibly with a Prompt Squirrel-like tag translation step.

## Requirements

- Krita.
- Krita AI Diffusion / ComfyUI backend.
- ComfyUI Partner/API access for the LLM layout call.
- A compatible Stable Diffusion checkpoint and installed Krita AI Diffusion workload.
- An RGB/Alpha Krita document. Krita AI Diffusion rejects CMYKA documents.

## Basic Use

1. Start the ComfyUI backend and open Krita.
2. Create or open an RGB/Alpha document.
3. Open the AI Image Generation docker.
4. Enter the base prompt in the normal prompt field. This prompt applies broadly to the image.
5. In the Layout Squirrel section, enter an optional layout description.
6. Enter region tags as comma-separated groups.
7. Use quotes around a region if its tag group contains commas, for example:

   ```text
   "fox, pink fur, flying", "gorilla, blue fur, sitting"
   ```

8. Click `Generate Layout`.
9. Move or resize the generated vector regions in Krita if needed.
10. Use `Region controls` if you need to adjust individual region prompts or weights.
11. Click the normal Krita AI Diffusion `Generate` button.

## Region Controls

- `Region`: the prompt text attached to that generated shape.
- `Weight`: per-region prompt weight. It defaults to `1.0`. Higher values are applied by wrapping the region prompt with weighted prompt syntax before it is sent through the normal text-conditioning path.
- `Feather`: softens the regional text-conditioning mask boundary.
- `Color`: experimental per-region colored-noise hint strength.
- `Color denoise`: global switch for the hidden color-hint image. `1.00` disables the color hint path; lower values use a generated color-noise image as an img2img/refine input.

## Debug Files

The prototype may write debug files in the Krita AI Diffusion user data directory:

- `layout_squirrel_debug.json`
- `layout_squirrel_color_hint.png`

These are meant for checking what the plugin sent to the backend and what hidden color-hint image was generated.

## Known Prototype Limits

- This is not a polished public release.
- The code is a patched/forked Krita AI Diffusion prototype, so upstream Krita AI Diffusion updates may break it.
- The LLM is only used for layout planning in this prototype.
- The layout description is not currently sent directly to Stable Diffusion.
- The region tags are the actual region prompts.
- Color hints are experimental and may behave unpredictably.
- Regional prompting is inherently fiddly; small regions, weak prompts, and model priors can still produce bad placement or wrong objects.
- These instructions were automatically written and may be incomplete or out of date.
