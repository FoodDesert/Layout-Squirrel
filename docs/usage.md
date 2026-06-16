# Layout Squirrel Usage

Automatically generated prototype documentation. This project is experimental, and there is no guarantee that every behavior described here works in every Krita AI Diffusion or ComfyUI setup.

## Purpose

- Layout Squirrel helps with image prompts where location and traits need to stay attached to the correct subject.
- Example: `a pink squirrel sits left of a blue deer` should place the squirrel on the left, the deer on the right, and keep the colors attached to the right subject.
- The current prototype uses an LLM for layout planning and Krita AI Diffusion for regional prompting.

## Inputs

- Main Krita AI Diffusion prompt: global/background prompt for the full image.
- Layout description: optional natural-language placement note used only by the LLM layout planner.
- Region tags: comma-separated region prompt groups. These become the actual regional prompts.

Quoted comma groups are supported:

```text
"deer, blue fur","rabbit, brown fur",tree
```

The LLM should create one region for each region tag group.

## Region Controls

- `Region`: prompt text attached to the generated shape.
- `Weight`: text-conditioning weight for the region. It defaults to `1.50x`.
- `Feather`: softness of the regional conditioning mask boundary. It defaults to `100%`.
- `Color`: per-region latent color-hint strength. It defaults to `0.50`.
- Palette swatches: one to three colors for the region color hint. The first swatch is also the visible vector shape color.

## How Generation Works

Layout Squirrel currently combines regional prompting and latent color hints.

Regional prompting:

- Region vector shapes are converted into masks.
- Each region prompt is applied through Krita AI Diffusion's regional conditioning path.
- Region `Weight` is applied by rewriting the prompt text before it is encoded.
- Example: `deer, blue fur` at `1.50x` becomes `(deer, blue fur:1.5)`.
- The global/background prompt still applies to the full image.

Latent color hints:

- Layout Squirrel generates a hidden color-hint image from the region palettes.
- In `Latent` color mode, that hint is projected into the latent image before sampling.
- Normal Comfy sampler noise is not replaced.
- At `Strength = 100%`, generation starts from an empty latent plus Layout Squirrel's color hint.
- At `Strength < 100%`, refine/img2img starts from the current canvas encoded into latent space, then applies the Layout Squirrel color hint before denoising.

## Basic Steps

1. Start ComfyUI and Krita.
2. Open an `RGB/Alpha` Krita document.
3. Open the AI Image Generation docker.
4. Enter the global/background prompt in the normal prompt field.
5. Enter an optional Layout Squirrel layout description.
6. Enter Layout Squirrel region tags.
7. Click `Generate Layout`.
8. Move, resize, or recolor regions if needed.
9. Adjust region weights, feathering, and color strengths if needed.
10. Press the main `Generate` button.

## Refining Existing Images

Displaying a generated result preview is not enough for refine/img2img.

To refine a generated result:

1. Click `Apply` on the generated result thumbnail.
2. Confirm the result appears as a normal layer in Krita's Layers panel.
3. Make that image layer visible.
4. Hide other normal image layers if you do not want them included in the refine input.
5. Keep Layout Squirrel regions available so the masks can still be read.
6. Set `Strength` below `100%`.
7. Press `Generate`.

Krita AI Diffusion captures the current document image by compositing visible normal image layers. It excludes internal AI preview layers, Layout Squirrel region layers, and control layers that are not part of the image.

## Practical Notes

- Regional prompting is a bias, not a strict guarantee.
- Very small regions may not give the model enough room to draw the requested subject.
- The visible vector colors help you read the layout; they are not directly painted into the output.
- The hidden latent color hint can bias colors and materials, but it does not guarantee correct generation.
- The layout description is not sent directly to Stable Diffusion in this prototype.
- The region tags are the actual regional prompts.
