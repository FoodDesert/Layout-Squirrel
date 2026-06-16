from __future__ import annotations
from copy import copy
from dataclasses import dataclass
import time
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4
from PIL import Image
import numpy as np
import base64
import torch
import torch.nn.functional as F
from io import BytesIO
from server import PromptServer, BinaryEventTypes

import comfy.sample
from comfy.clip_vision import ClipVisionModel
from comfy.sd import StyleModel
from comfy_api.latest import io


class LoadImageBase64(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LoadImageBase64",
            display_name="Load Image (Base64)",
            category="external_tooling",
            inputs=[io.String.Input("image", multiline=False)],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, image: str):
        _strip_prefix(image, "data:image/png;base64,")
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = None

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)


class LoadMaskBase64(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LoadMaskBase64",
            display_name="Load Mask (Base64)",
            category="external_tooling",
            inputs=[io.String.Input("mask", multiline=False)],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, mask: str):
        _strip_prefix(mask, "data:image/png;base64,")
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img.unsqueeze(0),)


class SendImageWebSocket(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_SendImageWebSocket",
            display_name="Send Image (WebSocket)",
            category="external_tooling",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("format", options=["PNG", "JPEG"], default="PNG"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, format: str):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                [format, image, None],
                server.client_id,
            )
            results.append({
                "source": "websocket",
                "content-type": f"image/{format.lower()}",
                "type": "output",
            })

        return io.NodeOutput(ui={"images": results})


class ImageCache:
    timeout = 600  # 10 minutes
    max_size = 100 * 1024 * 1024  # 100 MB

    @dataclass
    class Entry:
        data: bytes
        content_type: str
        timestamp: float
        retrieved: int

    class OldEntry(NamedTuple):
        last_used: float
        deleted: float
        size: int
        retrieved: int

    def __init__(self):
        self.images: dict[str, ImageCache.Entry] = {}
        self.old: dict[str, ImageCache.OldEntry] = {}

    def add(self, image: Image.Image, format: str):
        key = uuid4().hex
        with BytesIO() as output:
            image.save(output, format=format, quality=95, compress_level=1)
            image_data = output.getvalue()

        self.insert(key, image_data, f"image/{format.lower()}")
        return key

    def insert(self, key: str, data: bytes, content_type: str):
        self.images[key] = ImageCache.Entry(
            data=data,
            content_type=content_type,
            timestamp=time.time(),
            retrieved=0,
        )

    def get(self, key: str, extend: bool = False):
        entry = self.images.get(key)
        if entry is None:
            if old := self.old.get(key):
                now = time.time()
                print(
                    f"[comfyui-tooling-nodes] requested image {key} has been deleted ",
                    f"(last used {now - old.last_used:.0f}s ago, deleted {now - old.deleted:.0f}s ago, "
                    f"size {old.size / 1024**2:.1f}MB, retrieved {old.retrieved} times)",
                )
            return None, None
        entry.retrieved += 1
        if extend:
            entry.timestamp = time.time()
        self.prune()
        return entry.data, entry.content_type

    def prune(self):
        total_size = sum(len(entry.data) for entry in self.images.values())
        if total_size <= self.max_size:
            return
        # Remove least recently used entries until under max size
        sorted_entries = sorted(self.images.items(), key=lambda item: item[1].timestamp)
        now = time.time()
        for key, entry in sorted_entries:
            age = now - entry.timestamp
            if age > self.timeout or (age > 60 and entry.retrieved > 0):
                self.old[key] = ImageCache.OldEntry(
                    entry.timestamp, now, len(entry.data), entry.retrieved
                )
                del self.images[key]
                total_size -= len(entry.data)
                if total_size <= self.max_size:
                    break

    def __contains__(self, key: str):
        return key in self.images


image_cache = ImageCache()


class LoadImageCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LoadImageCache",
            display_name="Load Image from Cache",
            category="external_tooling",
            inputs=[io.String.Input("id", multiline=False)],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, id: str):
        image_data, content_type = image_cache.get(id, extend=True)
        if image_data is None:
            raise ValueError(f"Image with ID {id} not found in cache.")

        img = Image.open(BytesIO(image_data))
        w, h = img.size
        c = len(img.getbands())
        normalized = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).reshape(1, h, w, c)
        match c:
            case 1:
                image = tensor.expand(1, h, w, 3)
                mask = tensor.reshape(1, h, w)
            case 3:
                image = tensor
                mask = tensor[..., 0]
            case 4:
                image = tensor[..., :3]
                mask = tensor[..., 3]

        return io.NodeOutput(image, mask)


class SaveImageCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_SaveImageCache",
            display_name="Save Image to Cache",
            category="external_tooling",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("format", options=["PNG", "JPEG"], default="PNG"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, format: str):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            key = image_cache.add(image, format)

            results.append({
                "source": "http",
                "id": key,
                "content-type": f"image/{format.lower()}",
                "type": "output",
            })
        return io.NodeOutput(ui={"images": results})


def to_bchw(image: torch.Tensor):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.movedim(-1, 1)


def to_bhwc(image: torch.Tensor):
    return image.movedim(1, -1)


def mask_batch(mask: torch.Tensor):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask


class ApplyMaskToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ApplyMaskToImage",
            display_name="Apply Mask to Image",
            category="external_tooling",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
            ],
            outputs=[io.Image.Output(display_name="masked")],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, mask: torch.Tensor):
        out = to_bchw(image)
        if out.shape[1] == 3:  # Assuming RGB images
            out = torch.cat([out, torch.ones_like(out[:, :1, :, :])], dim=1)
        mask = mask_batch(mask)

        assert mask.ndim == 3, f"Mask should have shape [B, H, W]. {mask.shape}"
        assert out.ndim == 4, f"Image should have shape [B, C, H, W]. {out.shape}"
        assert out.shape[-2:] == mask.shape[-2:], (
            f"Image size {out.shape[-2:]} must match mask size {mask.shape[-2:]}"
        )
        is_mask_batch = mask.shape[0] == out.shape[0]

        # Apply each mask in the batch to its corresponding image's alpha channel
        for i in range(out.shape[0]):
            alpha = mask[i] if is_mask_batch else mask[0]
            out[i, 3, :, :] = alpha

        return (to_bhwc(out),)


class LayoutSquirrelLatentColorHint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LayoutSquirrelLatentColorHint",
            display_name="Layout Squirrel Latent Color Hint",
            category="external_tooling/latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("strength", default=0.05, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Latent.Output("samples", "samples")],
        )

    @classmethod
    def execute(
        cls,
        samples: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        strength: float,
    ):
        result = copy(samples)
        latent = samples["samples"].clone()
        if strength <= 0:
            result["samples"] = latent
            return (result,)

        batch, _channels, height, width = latent.shape
        dtype = latent.dtype
        device = latent.device

        rgb = to_bchw(image).to(device=device, dtype=dtype)
        if rgb.shape[0] == 1 and batch > 1:
            rgb = rgb.repeat(batch, 1, 1, 1)
        elif rgb.shape[0] != batch:
            rgb = rgb[:1].repeat(batch, 1, 1, 1)
        rgb = F.interpolate(rgb, size=(height, width), mode="area")

        alpha = mask.to(device=device, dtype=dtype)
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(0)
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(1)
        if alpha.shape[0] == 1 and batch > 1:
            alpha = alpha.repeat(batch, 1, 1, 1)
        elif alpha.shape[0] != batch:
            alpha = alpha[:1].repeat(batch, 1, 1, 1)
        alpha = F.interpolate(alpha, size=(height, width), mode="area").clamp(0, 1)

        # SDXL VAE latent color slopes estimated from flat color patches. This is deliberately
        # small and mask-gated so it nudges the initial noise rather than replacing it.
        color_to_latent = latent.new_tensor(
            [
                [-1.91268, -26.00622, 6.73422, -15.22388],
                [12.78446, 10.06698, 20.21957, 3.96531],
                [18.11768, 12.69493, -22.71445, 2.37386],
            ]
        )
        centered = rgb - latent.new_tensor(0.5019608)
        delta = torch.einsum("bchw,ck->bkhw", centered, color_to_latent)
        if delta.shape[1] != latent.shape[1]:
            adjusted = torch.zeros_like(latent)
            channels = min(delta.shape[1], adjusted.shape[1])
            adjusted[:, :channels] = delta[:, :channels]
            delta = adjusted
        latent = latent + delta * alpha * strength
        _save_layout_squirrel_latent_debug(latent)
        result["samples"] = latent
        return (result,)


class _LayoutSquirrelColorNoise:
    def __init__(self, seed: int, image: torch.Tensor, mask: torch.Tensor, strength: float):
        self.seed = seed
        self.image = image
        self.mask = mask
        self.strength = strength

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        base_noise = comfy.sample.prepare_noise(
            latent_image,
            self.seed,
            input_latent.get("batch_index"),
        )
        if self.strength <= 0 or latent_image.is_nested:
            return base_noise

        batch, _channels, height, width = latent_image.shape
        dtype = base_noise.dtype
        device = base_noise.device

        rgb = to_bchw(self.image).to(device=device, dtype=dtype)
        if rgb.shape[0] == 1 and batch > 1:
            rgb = rgb.repeat(batch, 1, 1, 1)
        elif rgb.shape[0] != batch:
            rgb = rgb[:1].repeat(batch, 1, 1, 1)
        rgb = F.interpolate(rgb, size=(height, width), mode="area")

        alpha = self.mask.to(device=device, dtype=dtype)
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(0)
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(1)
        if alpha.shape[0] == 1 and batch > 1:
            alpha = alpha.repeat(batch, 1, 1, 1)
        elif alpha.shape[0] != batch:
            alpha = alpha[:1].repeat(batch, 1, 1, 1)
        alpha = F.interpolate(alpha, size=(height, width), mode="area").clamp(0, 1)

        color_to_latent = base_noise.new_tensor(
            [
                [-1.91268, -26.00622, 6.73422, -15.22388],
                [12.78446, 10.06698, 20.21957, 3.96531],
                [18.11768, 12.69493, -22.71445, 2.37386],
            ]
        )
        centered = rgb - base_noise.new_tensor(0.5019608)
        delta = torch.einsum("bchw,ck->bkhw", centered, color_to_latent)
        if delta.shape[1] != base_noise.shape[1]:
            adjusted = torch.zeros_like(base_noise)
            channels = min(delta.shape[1], adjusted.shape[1])
            adjusted[:, :channels] = delta[:, :channels]
            delta = adjusted
        color_noise = base_noise + delta * alpha * self.strength
        _save_layout_squirrel_latent_debug(color_noise)
        return color_noise


class LayoutSquirrelColorNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LayoutSquirrelColorNoise",
            display_name="Layout Squirrel Color Noise",
            category="external_tooling/latent",
            inputs=[
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("strength", default=0.05, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Noise.Output()],
        )

    @classmethod
    def execute(cls, seed: int, image: torch.Tensor, mask: torch.Tensor, strength: float):
        return (_LayoutSquirrelColorNoise(seed, image, mask, strength),)


class LayoutSquirrelColorNoisePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_LayoutSquirrelColorNoisePreview",
            display_name="Layout Squirrel Color Noise Preview",
            category="external_tooling/latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("strength", default=0.05, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Latent.Output("samples", "samples")],
        )

    @classmethod
    def execute(
        cls,
        samples: dict,
        seed: int,
        image: torch.Tensor,
        mask: torch.Tensor,
        strength: float,
    ):
        result = copy(samples)
        result["samples"] = _LayoutSquirrelColorNoise(
            seed, image, mask, strength
        ).generate_noise(samples)
        return (result,)


def _save_layout_squirrel_latent_debug(latent: torch.Tensor):
    try:
        sample = latent[0].detach().float().cpu()
        if sample.shape[0] < 3:
            return
        rgb = sample[:3]
        low = torch.quantile(rgb.flatten(1), 0.01, dim=1).view(3, 1, 1)
        high = torch.quantile(rgb.flatten(1), 0.99, dim=1).view(3, 1, 1)
        rgb = ((rgb - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        array = (rgb.movedim(0, -1).numpy() * 255.0).astype(np.uint8)
        path = Path.cwd() / "temp" / "layout_squirrel_latent_noise_after.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, "RGB").save(path)
    except Exception as e:
        print(f"[comfyui-tooling-nodes] could not save Layout Squirrel latent debug image: {e}")


class _ReferenceImageData(NamedTuple):
    image: torch.Tensor
    weight: float
    range: tuple[float, float]


class ReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ReferenceImage",
            display_name="Reference Image",
            category="external_tooling",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("weight", default=1.0, min=0.0, max=10.0),
                io.Float.Input("range_start", default=0.0, min=0.0, max=1.0),
                io.Float.Input("range_end", default=1.0, min=0.0, max=1.0),
                io.Custom("ReferenceImage").Input("reference_images", optional=True),
            ],
            outputs=[io.Custom("ReferenceImage").Output(display_name="reference_images")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        weight: float,
        range_start: float,
        range_end: float,
        reference_images: list[_ReferenceImageData] | None = None,
    ):
        imgs = copy(reference_images) if reference_images is not None else []
        imgs.append(_ReferenceImageData(image, weight, (range_start, range_end)))
        return (imgs,)


class ApplyReferenceImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ETN_ApplyReferenceImages",
            display_name="Apply Reference Images",
            category="external_tooling",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.ClipVision.Input("clip_vision"),
                io.StyleModel.Input("style_model"),
                io.Custom("ReferenceImage").Input("references"),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(
        cls,
        conditioning: list[list],
        clip_vision: ClipVisionModel,
        style_model: StyleModel,
        references: list[_ReferenceImageData],
    ):
        delimiters = {0.0, 1.0}
        delimiters |= set(r.range[0] for r in references)
        delimiters |= set(r.range[1] for r in references)
        delimiters = sorted(delimiters)
        ranges = [(delimiters[i], delimiters[i + 1]) for i in range(len(delimiters) - 1)]

        embeds = [_encode_image(r.image, clip_vision, style_model, r.weight) for r in references]
        base = conditioning[0][0]
        result = []
        for start, end in ranges:
            e = [
                embeds[i]
                for i, r in enumerate(references)
                if r.range[0] <= start and r.range[1] >= end
            ]
            options = conditioning[0][1].copy()
            options["start_percent"] = start
            options["end_percent"] = end
            result.append((torch.cat([base] + e, dim=1), options))

        return (result,)


def _encode_image(
    image: torch.Tensor, clip_vision: ClipVisionModel, style_model: StyleModel, weight: float
):
    e = clip_vision.encode_image(image)
    e = style_model.get_cond(e).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
    e = _downsample_image_cond(e, weight)
    return e


def _downsample_image_cond(cond: torch.Tensor, weight: float):
    if weight >= 1.0:
        return cond
    elif weight <= 0.0:
        return torch.zeros_like(cond)
    elif weight >= 0.6:
        factor = 2
    elif weight >= 0.3:
        factor = 3
    else:
        factor = 4

    # Downsample the clip vision embedding to make it smaller, resulting in less impact
    # compared to other conditioning.
    # See https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl
    (b, t, h) = cond.shape
    m = int(np.sqrt(t))
    cond = F.interpolate(
        cond.view(b, m, m, h).transpose(1, -1),
        size=(m // factor, m // factor),
        mode="area",
    )
    return cond.transpose(1, -1).reshape(b, -1, h)


def _strip_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s
