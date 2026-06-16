from __future__ import annotations

import colorsys
from enum import Enum

from PyQt5.QtCore import QMetaObject, QObject, Qt, QUuid, pyqtSignal

from .. import eventloop, util
from ..backend import workflow
from ..backend.api import ConditioningInput, RegionInput
from ..backend.client import Client
from ..document import Layer, LayerType
from ..image import Bounds, Extent, Image
from ..settings import settings
from ..style import Style
from . import model
from .control import ControlLayerList
from .jobs import JobRegion
from .properties import ObservableProperties, Property


class RegionLink(Enum):
    direct = 0  # layer is directly linked to a region
    indirect = 1  # layer is in a group which is linked to a region
    any = 3  # either direct or indirect link


class Region(QObject, ObservableProperties):
    """A sub-area of the image where region-specific text prompts and control layers are applied.
    A region is linked to one or more layers. The layer's coverage mask defines the area of the region.
    """

    _parent: RootRegion
    _layers: list[QUuid]

    layer_ids = Property("", persist=True, setter="_set_layer_ids")
    positive = Property("", persist=True)
    conditioning_strength = Property(1.0, persist=True)
    conditioning_feather = Property(0.25, persist=True)
    color_hint_strength = Property(0.45, persist=True)
    hint_colors = Property([], persist=True)
    full_strength_mask = Property(False, persist=True)
    control: ControlLayerList

    layer_ids_changed = pyqtSignal(str)
    positive_changed = pyqtSignal(str)
    conditioning_strength_changed = pyqtSignal(float)
    conditioning_feather_changed = pyqtSignal(float)
    color_hint_strength_changed = pyqtSignal(float)
    hint_colors_changed = pyqtSignal(list)
    full_strength_mask_changed = pyqtSignal(bool)
    modified = pyqtSignal(QObject, str)

    def __init__(self, parent: RootRegion, model: model.DocumentModel):
        super().__init__()
        self._parent = parent
        self._layers = []
        self.control = ControlLayerList(model)

    def _get_layers(self):
        col = self._parent._model.layers.updated()
        all = (col.find(id) for id in self._layers)
        pruned = [l for l in all if l is not None]
        self._set_layers([l.id for l in pruned])
        return pruned

    def _set_layers(self, ids: list[QUuid]):
        self._layers = ids
        new_ids_string = ",".join(id.toString() for id in ids)
        if self.layer_ids != new_ids_string:
            self._layer_ids = new_ids_string
            self.layer_ids_changed.emit(self._layer_ids)

    def _set_layer_ids(self, ids: str):
        if self._layer_ids == ids:
            return
        self._layer_ids = ids
        self._layers = [QUuid(id) for id in ids.split(",") if id]
        self.layer_ids_changed.emit(ids)
        self.modified.emit(self, "layer_ids")

    @property
    def layers(self):
        return self._get_layers()

    @property
    def first_layer(self):
        layers = self.layers
        return layers[0] if len(layers) > 0 else None

    @property
    def name(self):
        if len(self._layers) == 0:
            return "No layers linked"
        return ", ".join(l.name for l in self.layers)

    def link(self, layer: Layer):
        if layer.id not in self._layers:
            self._set_layers(self._layers + [layer.id])

    def unlink(self, layer: Layer):
        if layer.id in self._layers:
            self._set_layers([l for l in self._layers if l != layer.id])

    def is_linked(self, layer: Layer, mode=RegionLink.any):
        target = layer
        if mode is not RegionLink.direct:
            target = Region.link_target(layer)
        if mode is RegionLink.indirect and target is layer:
            return False
        if mode is RegionLink.direct or target is layer:
            return layer.id in self._layers
        return self.root.find_linked(target) is self

    def link_active(self):
        self.link(self._parent.layers.active)

    def unlink_active(self):
        self.unlink(self._parent.layers.active)

    def toggle_active_link(self):
        if self.is_active_linked:
            self.unlink_active()
        else:
            self.link_active()

    @property
    def has_links(self):
        return len(self._layers) > 0

    @property
    def is_active_linked(self):
        return self.is_linked(self._parent.layers.active)

    def remove(self):
        self._parent.remove(self)

    @property
    def root(self):
        return self._parent

    @property
    def siblings(self):
        return self._parent.find_siblings(self)

    @staticmethod
    def link_target(layer: Layer):
        if layer.type is LayerType.group:
            return layer
        if parent := layer.parent_layer:
            if not parent.is_root and parent.type is LayerType.group:
                return parent
        return layer

    async def translate_prompt(self, client: Client):
        if positive := self.positive:
            translated = await client.translate(positive, settings.prompt_translation)
            if positive == self.positive:
                self.positive = translated


class RootRegion(QObject, ObservableProperties):
    """Manages a collection of regions, each of which is linked to one or more layers in the document.
    Defines text prompt and control layers which are applied to all regions in the collection.
    If there are no regions, the root region is used as a default for the entire document.
    """

    positive = Property("", persist=True)
    negative = Property("", persist=True)
    negative_enabled = Property(True)
    negative_enabled_live = Property(True)
    control: ControlLayerList

    positive_changed = pyqtSignal(str)
    negative_changed = pyqtSignal(str)
    negative_enabled_changed = pyqtSignal(bool)
    negative_enabled_live_changed = pyqtSignal(bool)
    active_changed = pyqtSignal(Region)
    active_layer_changed = pyqtSignal()
    added = pyqtSignal(Region)
    removed = pyqtSignal(Region)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: model.DocumentModel):
        super().__init__()
        self._model = model
        self._regions: list[Region] = []
        self.control = ControlLayerList(model)
        self._active: Region | None = None
        self._active_layer: QUuid | None = None
        self._style_connection: QMetaObject.Connection | None = None
        model.layers.active_changed.connect(self._update_active)
        model.layers.parent_changed.connect(self._update_group)
        model.style_changed.connect(self._handle_style_changed)
        self._handle_style_changed(model.style)

    def _find_region(self, layer: Layer):
        return next((r for r in self._regions if r.is_linked(layer, RegionLink.direct)), None)

    def emplace(self):
        region = Region(self, self._model)
        self._regions.append(region)
        return region

    @property
    def active(self):
        self._update_active()
        return self._active

    @active.setter
    def active(self, region: RootRegion | Region | None):
        if isinstance(region, RootRegion):
            region = None
        if self._active != region:
            self._active = region
            self.active_changed.emit(region)
            self._track_layer(region)

    @property
    def active_or_root(self):
        return self.active or self

    @property
    def region_for_active_layer(self):
        if layer := self._get_active_layer()[0]:
            return self.find_linked(layer)

    def get_active_region_layer(self, use_parent: bool):
        result = self.layers.root
        target = Region.link_target(self.layers.active)
        if self.is_linked(target):
            result = target
        if use_parent and result.parent_layer is not None:
            result = result.parent_layer
        return result

    def add_control(self):
        self.active_or_root.control.add()

    def is_linked(self, layer: Layer, mode=RegionLink.any):
        return any(r.is_linked(layer, mode) for r in self._regions)

    def find_linked(self, layer: Layer, mode=RegionLink.any):
        return next((r for r in self._regions if r.is_linked(layer, mode)), None)

    def create_region_layer(self):
        self.create_region(group=False)

    def create_region_group(self):
        self.create_region(group=True)

    def create_region(self, group=True):
        """Create a new region. This action depends on context:
        If the active layer can be linked to a group it will be used as the initial link
        target for the new group. Otherwise, a new layer is inserted (or a group if group==True)
        and that will be linked instead.
        """
        layers = self._model.layers
        target = Region.link_target(layers.active)
        can_link = target.type in [LayerType.paint, LayerType.group] and not self.is_linked(target)
        if can_link:
            layer = target
        elif group:
            layer = layers.create_group(f"Region {len(self)}")
            layers.create("Paint layer", parent=layer)
        else:
            layer = layers.create(f"Region {len(self)}")
        return self._add(layer)

    def add_region_for_layer(self, layer: Layer):
        existing = self.find_linked(layer, RegionLink.direct)
        if existing is not None:
            return existing
        return self._add(layer)

    def remove(self, region: Region):
        if region in self._regions:
            if self.active == region:
                self.active = None
            self._regions.remove(region)
            self.removed.emit(region)

    def _get_regions(self, layers: list[Layer], exclude: Region | None = None):
        regions = []
        for l in layers:
            r = self._find_region(l)
            if r is not None and r is not exclude and r not in regions:
                regions.append(r)
        return regions

    def find_siblings(self, region: Region):
        if layer := region.first_layer:
            below, above = layer.siblings
            return self._get_regions(below, region), self._get_regions(above, region)
        return [], []

    @property
    def siblings(self):
        if self.layers:
            layer = self.layers.root
            if active_layer := self._get_active_layer()[0]:
                active_layer = Region.link_target(active_layer)
                if self.is_linked(active_layer):
                    layer = active_layer.parent_layer or active_layer
            return [], self._get_regions(layer.child_layers)
        return [], []

    def last_unlinked_layer(self, parent: Layer):
        result = None
        for node in parent.child_layers:
            if self.is_linked(node):
                break
            result = node
        return result

    def _get_active_layer(self):
        if not self.layers:
            return None, False
        layer = self.layers.active
        if layer.id == self._active_layer:
            return layer, False
        self._active_layer = layer.id
        self.active_layer_changed.emit()
        return layer, True

    def _update_active(self):
        layer, changed = self._get_active_layer()
        if layer and changed and (region := self.find_linked(layer)):
            self.active = region

    def _track_layer(self, region: Region | None):
        if region and region.first_layer:
            layer, changed = self._get_active_layer()
            if layer and not changed and not region.is_linked(layer):
                target = region.first_layer
                if target.type is LayerType.group and len(target.child_layers) > 0:
                    target = target.child_layers[-1]
                self.layers.active = target

    def _update_group(self, layer: Layer):
        """If a layer is moved into a group, promote the region to non-destructive apply workflow."""
        if layer.type is not LayerType.group:
            if region := self.find_linked(layer, RegionLink.direct):
                if (parent := layer.parent_layer) and not parent.is_root:
                    region.unlink(layer)
                    region.link(parent)

    def _handle_style_changed(self, style: Style):
        if self._style_connection:
            QObject.disconnect(self._style_connection)
        self._style_connection = style.changed.connect(self._handle_style_update)
        self._update_negative_enabled()

    def _handle_style_update(self, name: str, value: object):
        self._update_negative_enabled()

    def _update_negative_enabled(self):
        supported = self._model.arch.supports_cfg
        self.negative_enabled = supported and self._model.style.cfg_scale > 1
        self.negative_enabled_live = supported and self._model.style.live_cfg_scale > 1

    def _add(self, layer: Layer):
        region = Region(self, self._model)
        region.link(layer)
        self._regions.append(region)
        self.added.emit(region)
        self.active = region
        return region

    @property
    def layers(self):
        return self._model.layers

    async def translate_prompt(self, client: Client):
        if positive := self.positive:
            translated = await client.translate(positive, settings.prompt_translation)
            if positive == self.positive:
                self.positive = translated
        if self.negative:
            negative = self.negative
            translated = await client.translate(negative, settings.prompt_translation)
            if negative == self.negative:
                self.negative = translated

    def __len__(self):
        return len(self._regions)

    def __iter__(self):
        return iter(self._regions)


def translate_prompt(region: Region | RootRegion):
    from .root import root

    if client := root.connection.client_if_connected:
        if settings.prompt_translation and client.features.translation:
            eventloop.run(region.translate_prompt(client))


def get_region_inpaint_mask(region_layer: Layer, max_extent: Extent, min_size=0):
    region_bounds = region_layer.compute_bounds()
    padding = int((settings.selection_padding / 100) * region_bounds.extent.average_side)
    bounds = Bounds.pad(region_bounds, padding, min_size=min_size, square=min_size > 0)
    bounds = Bounds.clamp(bounds, max_extent)
    mask_image = region_layer.get_mask(bounds)
    return mask_image.to_mask(bounds)


def process_regions(
    root: RootRegion,
    bounds: Bounds,
    parent_layer: Layer | None = None,
    min_coverage=0.02,
    time: int | None = None,
):
    parent_region = None
    if parent_layer and not parent_layer.is_root:
        parent_region = root.find_linked(parent_layer)

    parent_prompt = ""
    job_info = []
    control = root.control.to_api(bounds, time)
    if parent_layer and parent_region:
        parent_prompt = parent_region.positive
        control += parent_region.control.to_api(bounds, time)
        job_info = [JobRegion(parent_layer.id_string, parent_prompt, bounds)]
    result = ConditioningInput(
        positive=workflow.merge_prompt(parent_prompt, root.positive),
        negative=root.negative,
        control=control,
    )

    # Collect layers with linked regions. Optionally restrict to to child layers of a region.
    if parent_layer is not None and not parent_layer.is_root:
        child_layers = parent_layer.child_layers
    else:
        child_layers = root.layers.all
        parent_layer = root.layers.root
    layer_regions = ((l, root.find_linked(l, RegionLink.direct)) for l in child_layers)
    layer_regions = [(l, r) for l, r in layer_regions if r is not None]
    if len(layer_regions) == 0:
        return result, job_info

    # Get region masks. Filter out regions with:
    # * no content (empty mask)
    # * less than minimum overlap (estimate based on bounding box)
    result_regions: list[tuple[RegionInput, JobRegion]] = []
    for layer, region in layer_regions:
        layer_bounds = layer.compute_bounds()
        if layer_bounds.area == 0:
            continue

        coverage_rough = Bounds.intersection(bounds, layer_bounds).area / bounds.area
        if coverage_rough < 2 * min_coverage:
            continue

        mask = layer.get_mask(bounds)

        weighted_prompt = _weighted_region_prompt(region.positive, region.conditioning_strength)
        region_result = RegionInput(
            mask,
            layer_bounds,
            workflow.merge_prompt(weighted_prompt, root.positive),
            conditioning_feather=region.conditioning_feather,
            color_hint_strength=region.color_hint_strength,
            control=region.control.to_api(bounds, time),
        )
        job_params = JobRegion(layer.id_string, region.positive, layer_bounds)
        result_regions.append((region_result, job_params))

    # Remove from each region mask any overlapping areas from regions above it.
    accumulated_mask = None
    for i in range(len(result_regions) - 1, -1, -1):
        region, job_region = result_regions[i]
        assert region.mask is not None
        mask = region.mask
        if accumulated_mask is not None:
            mask = Image.mask_subtract(mask, accumulated_mask)

        coverage = mask.average()
        if coverage > 0.9 and min_coverage > 0:
            # Single region covers (almost) entire image, don't use regional conditioning.
            result.positive = region.positive
            result.control += region.control
            return result, [job_region]
        elif coverage < min_coverage:
            # Region has less than minimum coverage, remove it.
            result_regions.pop(i)
        else:
            # Accumulate mask for next region, and store modified mask.
            if accumulated_mask is None:
                accumulated_mask = Image.copy(region.mask)
            accumulated_mask = Image.mask_add(accumulated_mask, region.mask)
            region.mask = mask

    # If there are no regions left, don't use regional conditioning.
    if len(result_regions) == 0:
        return result, job_info

    # If the region(s) don't cover the entire image, add a final region for the remaining area.
    assert accumulated_mask is not None, "Expecting at least one region mask"
    total_coverage = accumulated_mask.average()
    if total_coverage < 0.95:
        accumulated_mask.invert()
        input = RegionInput(accumulated_mask, bounds, result.positive)
        job = JobRegion(parent_layer.id_string, "background", bounds, is_background=True)
        result_regions.insert(0, (input, job))

    result.regions = [r for r, _ in result_regions]
    return result, [j for _, j in result_regions]


def _uses_full_strength_mask(layer: Layer, region: Region):
    parent = layer.parent_layer
    return region.full_strength_mask or (
        parent is not None
        and parent.name.startswith(("Layout Squirrel Regions", "LLM Layout Regions"))
    )


def _weighted_region_prompt(prompt: str, strength: float):
    prompt = prompt.strip()
    if prompt == "" or abs(strength - 1.0) < 0.005:
        return prompt
    strength_text = f"{strength:.2f}".rstrip("0").rstrip(".")
    return f"({prompt}:{strength_text})"


def get_layout_squirrel_color_hint(
    root: RootRegion,
    bounds: Bounds,
    parent_layer: Layer | None = None,
    seed: int = 0,
):
    if parent_layer is not None and not parent_layer.is_root:
        child_layers = parent_layer.child_layers
    else:
        child_layers = root.layers.all
    layer_regions = ((l, root.find_linked(l, RegionLink.direct)) for l in child_layers)
    layer_regions = [(l, r) for l, r in layer_regions if r is not None]

    hint = _create_seeded_noise(bounds.extent, seed)
    used_hint = False
    for layer, _region in layer_regions:
        if not _uses_full_strength_mask(layer, _region):
            continue
        color_strength = _region.color_hint_strength
        if color_strength <= 0.0:
            continue
        used_hint = (
            _draw_noisy_color_hint(hint, layer, _region, bounds, seed, color_strength) or used_hint
        )

    if used_hint:
        try:
            hint.save(util.user_data_dir / "layout_squirrel_color_hint.png")
        except Exception as e:
            util.client_logger.warning(
                f"Could not save Layout Squirrel color hint debug image: {e}"
            )
        return hint
    return None


def get_layout_squirrel_latent_hint(
    root: RootRegion,
    bounds: Bounds,
    parent_layer: Layer | None = None,
    seed: int = 0,
):
    if parent_layer is not None and not parent_layer.is_root:
        child_layers = parent_layer.child_layers
    else:
        child_layers = root.layers.all
    layer_regions = ((l, root.find_linked(l, RegionLink.direct)) for l in child_layers)
    layer_regions = [(l, r) for l, r in layer_regions if r is not None]

    hint = Image.create(bounds.extent, fill=0)
    used_hint = False
    for layer, _region in layer_regions:
        if not _uses_full_strength_mask(layer, _region):
            continue
        color_strength = _region.color_hint_strength
        if color_strength <= 0.0:
            continue
        used_hint = (
            _draw_latent_color_hint_map(hint, layer, _region, bounds, seed, color_strength)
            or used_hint
        )

    if used_hint:
        try:
            hint.save(util.user_data_dir / "layout_squirrel_latent_hint.png")
        except Exception as e:
            util.client_logger.warning(
                f"Could not save Layout Squirrel latent hint debug image: {e}"
            )
        return hint
    return None


def _create_seeded_noise(extent: Extent, seed: int):
    image = Image.create(extent)
    for y in range(extent.height):
        for x in range(extent.width):
            hue = _unit_noise(seed, 0xBACE, x, y, 0)
            saturation = 0.78 + 0.22 * _unit_noise(seed, 0xBACE, x, y, 1)
            value = 0.35 + 0.65 * _unit_noise(seed, 0xBACE, x, y, 2)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            image.set_pixel(x, y, (round(r * 255), round(g * 255), round(b * 255), 255))
    return image


def _draw_noisy_color_hint(
    hint: Image, layer: Layer, region: Region, bounds: Bounds, seed: int, color_strength: float
):
    layer_bounds = Bounds.intersection(bounds, layer.compute_bounds())
    if layer_bounds.area == 0:
        return False

    pixels = layer.get_pixels(layer_bounds)
    palette = _region_hint_palette(region, pixels)
    if not palette:
        return False

    layer_key = _text_hash(layer.name)
    rel = layer_bounds.relative_to(bounds)
    used = False

    for y in range(layer_bounds.height):
        for x in range(layer_bounds.width):
            pixel = pixels.pixel(x, y)
            if not isinstance(pixel, tuple):
                continue
            alpha = pixel[3]
            if alpha <= 0:
                continue

            dot_probability = _per_color_probability(color_strength * (alpha / 255), len(palette))
            for palette_index, color in reversed(list(enumerate(palette))):
                channel_offset = 10 + palette_index * 8
                if _unit_noise(seed, layer_key, x, y, channel_offset) >= dot_probability:
                    continue

                luma_noise = round(
                    -60 + 120 * _unit_noise(seed, layer_key, x, y, channel_offset + 1)
                )
                channel_noise = 35
                r = _byte(
                    color[0]
                    + luma_noise
                    + _signed_noise(seed, layer_key, x, y, channel_offset + 2, channel_noise)
                )
                g = _byte(
                    color[1]
                    + luma_noise
                    + _signed_noise(seed, layer_key, x, y, channel_offset + 3, channel_noise)
                )
                b = _byte(
                    color[2]
                    + luma_noise
                    + _signed_noise(seed, layer_key, x, y, channel_offset + 4, channel_noise)
                )
                hint.set_pixel(rel.x + x, rel.y + y, (r, g, b, 255))
                used = True

    return used


def _draw_latent_color_hint_map(
    hint: Image, layer: Layer, region: Region, bounds: Bounds, seed: int, color_strength: float
):
    layer_bounds = Bounds.intersection(bounds, layer.compute_bounds())
    if layer_bounds.area == 0:
        return False

    pixels = layer.get_pixels(layer_bounds)
    palette = _region_hint_palette(region, pixels)
    if not palette:
        return False

    rel = layer_bounds.relative_to(bounds)
    cells: list[tuple[int, int, int]] = []
    latent_x0 = rel.x // 8
    latent_y0 = rel.y // 8
    latent_x1 = (rel.x + layer_bounds.width + 7) // 8
    latent_y1 = (rel.y + layer_bounds.height + 7) // 8
    for ly in range(latent_y0, latent_y1):
        for lx in range(latent_x0, latent_x1):
            sample_x = min(max(lx * 8 + 4 - rel.x, 0), layer_bounds.width - 1)
            sample_y = min(max(ly * 8 + 4 - rel.y, 0), layer_bounds.height - 1)
            pixel = pixels.pixel(sample_x, sample_y)
            if not isinstance(pixel, tuple):
                continue
            alpha = pixel[3]
            if alpha <= 0:
                continue
            cells.append((lx, ly, alpha))

    if not cells:
        return False

    layer_key = _text_hash(layer.name)
    cells.sort(key=lambda cell: _unit_noise(seed, layer_key, cell[0], cell[1], 0))
    assignments = _palette_assignments(len(cells), len(palette))
    used = False
    for cell, palette_index in zip(cells, assignments):
        lx, ly, alpha = cell
        color = palette[palette_index]
        a = _byte(round(alpha * color_strength))
        for y in range(ly * 8, min((ly + 1) * 8, hint.height)):
            for x in range(lx * 8, min((lx + 1) * 8, hint.width)):
                hint.set_pixel(x, y, (color[0], color[1], color[2], a))
                used = True
    return used


def _palette_assignments(cell_count: int, color_count: int):
    color_count = max(1, min(color_count, cell_count))
    weights = [0.55, 0.30, 0.15][:color_count]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    counts = [1] * color_count
    remaining = cell_count - color_count
    raw = [remaining * w for w in weights]
    counts = [count + int(value) for count, value in zip(counts, raw)]
    assigned = sum(counts)
    fractions = sorted(
        ((raw[i] - int(raw[i]), i) for i in range(color_count)),
        reverse=True,
    )
    for _, index in fractions[: max(0, cell_count - assigned)]:
        counts[index] += 1

    assignments = []
    for index, count in enumerate(counts):
        assignments.extend([index] * count)
    return assignments[:cell_count]


def _per_color_probability(total_probability: float, color_count: int):
    total_probability = max(0.0, min(1.0, total_probability))
    color_count = max(1, color_count)
    return 1.0 - ((1.0 - total_probability) ** (1.0 / color_count))


def _region_hint_palette(region: Region, pixels: Image):
    palette = []
    for color in region.hint_colors[:3]:
        parsed = _parse_hex_color(color)
        if parsed is not None and parsed not in palette:
            palette.append(parsed)
    if palette:
        return palette

    primary = _average_visible_color(pixels)
    if primary is None:
        return []
    palette = [primary]
    return palette


def _parse_hex_color(color: object):
    if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
        return None
    try:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    except ValueError:
        return None


def _signed_noise(seed: int, layer_key: int, x: int, y: int, channel: int, amount: int):
    return round(-amount + 2 * amount * _unit_noise(seed, layer_key, x, y, channel))


def _unit_noise(seed: int, layer_key: int, x: int, y: int, channel: int):
    return _hash32(seed, layer_key, x, y, channel) / 0xFFFFFFFF


def _text_hash(text: str):
    result = 2166136261
    for char in text:
        result ^= ord(char)
        result = (result * 16777619) & 0xFFFFFFFF
    return result


def _hash32(*values: int):
    result = 0x811C9DC5
    for value in values:
        value &= 0xFFFFFFFF
        result ^= value
        result = (result * 0x01000193) & 0xFFFFFFFF
        result ^= result >> 16
    result ^= result >> 13
    result = (result * 0x85EBCA6B) & 0xFFFFFFFF
    result ^= result >> 16
    return result


def _average_visible_color(image: Image, mask: Image | None = None):
    total_r = total_g = total_b = total_a = 0
    for y in range(image.height):
        for x in range(image.width):
            pixel = image.pixel(x, y)
            if not isinstance(pixel, tuple):
                continue
            r, g, b, a = pixel
            if mask is not None:
                mask_pixel = mask.pixel(x, y)
                mask_alpha = mask_pixel[0] if isinstance(mask_pixel, tuple) else mask_pixel
                a = min(a, mask_alpha)
            if a == 0:
                continue
            total_r += r * a
            total_g += g * a
            total_b += b * a
            total_a += a
    if total_a == 0:
        return None
    return total_r // total_a, total_g // total_a, total_b // total_a


def _byte(value: int):
    return max(0, min(255, value))
