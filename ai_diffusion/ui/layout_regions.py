from __future__ import annotations

import asyncio
from html import escape
from typing import TYPE_CHECKING

from PyQt5.QtCore import QMetaObject, QObject, Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..backend.client import ClientEvent, TextOutput
from ..backend.api import (
    CustomWorkflowInput,
    ImageInput,
    InpaintMode,
    InpaintParams,
    SamplingInput,
    WorkflowInput,
    WorkflowKind,
)
from ..layout_regions import (
    LayoutError,
    build_layout_prompt,
    parse_region_tags,
    region_to_svg,
    validate_layout_response,
)
from ..image import Bounds, Extent
from ..localization import translate as _
from ..model.connection import ConnectionState
from ..model.root import root
from ..settings import settings
from ..layer import Layer, LayerType
from ..util import trim_text

if TYPE_CHECKING:
    from ..model.model import DocumentModel

LAYOUT_REGION_GROUP_NAMES = ("Layout Squirrel Regions", "LLM Layout Regions")


class LayoutRegionWidget(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._model: DocumentModel = root.active_model
        self._task: asyncio.Task | None = None

        self.setFrameStyle(QFrame.Shape.StyledPanel)

        self.description = QPlainTextEdit(self)
        self.description.setPlaceholderText(
            _("Optional layout description, e.g. The fox is on the left and the cat is on the right.")
        )
        self.description.setMaximumBlockCount(8)
        self.description.setFixedHeight(54)
        self.description.setPlainText(settings.llm_layout_description)
        self.description.textChanged.connect(self._save_defaults)

        self.tags = QLineEdit(self)
        self.tags.setPlaceholderText(
            _(
                'Region tags, comma-separated. Use quotes for commas, e.g. '
                '"red fox, orange fur", "domestic cat, grey fur"'
            )
        )
        self.tags.setText(settings.llm_layout_tags)
        self.tags.textChanged.connect(self._save_defaults)

        self.model_select = QComboBox(self)
        self.model_select.addItems(["gpt-5-nano", "gpt-5-mini", "gpt-5"])
        model_index = self.model_select.findText(settings.llm_layout_model)
        if model_index >= 0:
            self.model_select.setCurrentIndex(model_index)
        self.model_select.setToolTip(_("Comfy Partner OpenAI model used to plan the layout"))
        self.model_select.currentTextChanged.connect(self._save_defaults)

        self.color_hint_mode = QComboBox(self)
        self.color_hint_mode.addItem(_("Pixel"), "pixel")
        self.color_hint_mode.addItem(_("Latent"), "latent")
        mode_index = self.color_hint_mode.findData(settings.llm_layout_color_hint_mode)
        self.color_hint_mode.setCurrentIndex(max(0, mode_index))
        self.color_hint_mode.setToolTip(
            _(
                "Color hint strategy. Pixel uses the current img2img colored-noise image. "
                "Latent injects seeded palette color directions into the initial latent noise."
            )
        )
        self.color_hint_mode.currentIndexChanged.connect(self._save_defaults)

        self.generate_button = QPushButton(_("Generate Layout"), self)
        self.generate_button.clicked.connect(self.generate_layout)

        self.panel_toggle = QPushButton(self)
        self.panel_toggle.setCheckable(True)
        self.panel_toggle.setChecked(True)
        self.panel_toggle.setToolTip(_("Show or hide Layout Squirrel planning controls."))
        self.panel_toggle.toggled.connect(self._set_panel_expanded)

        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.region_controls = QFrame(self)
        self.region_controls_layout = QGridLayout()
        self.region_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.region_controls_layout.setHorizontalSpacing(6)
        self.region_controls_layout.setVerticalSpacing(2)
        self.region_controls.setLayout(self.region_controls_layout)
        self.region_controls_scroll = QScrollArea(self)
        self.region_controls_scroll.setWidgetResizable(True)
        self.region_controls_scroll.setMaximumHeight(150)
        self.region_controls_scroll.setWidget(self.region_controls)
        self.region_controls_toggle = QPushButton(_("Region controls"), self)
        self.region_controls_toggle.setCheckable(True)
        self.region_controls_toggle.setChecked(False)
        self.region_controls_toggle.setToolTip(
            _("Show or hide per-region prompt, weight, and feather controls.")
        )
        self.region_controls_toggle.toggled.connect(self.region_controls_scroll.setVisible)
        self.region_controls_scroll.setVisible(False)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(2)
        llm_label = QLabel(_("LLM"), self)
        llm_label.setToolTip(_("Comfy Partner/API model used only to plan layout regions."))
        mode_label = QLabel(_("Color mode"), self)
        mode_label.setToolTip(self.color_hint_mode.toolTip())
        controls.addWidget(llm_label, 0, 0)
        controls.addWidget(mode_label, 0, 1)
        controls.addWidget(self.model_select, 1, 0)
        controls.addWidget(self.color_hint_mode, 1, 1)
        controls.addWidget(self.generate_button, 1, 2)

        self.body = QFrame(self)
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)
        body_layout.addWidget(self.description)
        body_layout.addWidget(self.tags)
        body_layout.addLayout(controls)
        body_layout.addWidget(self.region_controls_toggle)
        body_layout.addWidget(self.region_controls_scroll)
        self.body.setLayout(body_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.panel_toggle)
        layout.addWidget(self.body)
        layout.addWidget(self.status)
        self.setLayout(layout)
        self._set_panel_expanded(True)
        self._apply_base_prompt_default()
        self._refresh_region_controls()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: "DocumentModel"):
        self._model = model
        self._apply_base_prompt_default()
        self._refresh_region_controls()

    def _save_defaults(self, *ignored):
        settings.llm_layout_description = self.description.toPlainText()
        settings.llm_layout_tags = self.tags.text()
        settings.llm_layout_model = self.model_select.currentText()
        settings.llm_layout_color_hint_mode = self.color_hint_mode.currentData()
        settings.save()

    def _apply_base_prompt_default(self):
        prompt = self._model.regions.positive.strip()
        if prompt == "":
            self._model.regions.positive = settings.llm_layout_base_prompt
        elif prompt == "duo, white background, chibi":
            self._model.regions.positive = settings.llm_layout_base_prompt

    def _set_panel_expanded(self, expanded: bool):
        self.body.setVisible(expanded)
        self.panel_toggle.setText(_("Layout Squirrel controls") + (" [-]" if expanded else " [+]"))

    def generate_layout(self):
        if self._task and not self._task.done():
            return
        try:
            tags = parse_region_tags(self.tags.text())
        except LayoutError as e:
            self._set_status(str(e), error=True)
            return

        extent = self._model.document.extent
        prompt = build_layout_prompt(
            self.description.toPlainText(),
            tags,
            extent.width,
            extent.height,
        )
        self.generate_button.setEnabled(False)
        self._set_status(_("Requesting layout from Comfy Partner OpenAI node..."))
        self._task = eventloop.run(self._generate_layout(prompt, tags, extent.width, extent.height))

    async def _generate_layout(
        self,
        prompt: str,
        tags: list[str],
        canvas_width: int,
        canvas_height: int,
    ):
        try:
            text = await self._request_llm_layout(prompt)
            regions = validate_layout_response(text, tags, canvas_width, canvas_height)
            self._create_region_layers(tags, regions, canvas_width, canvas_height)
            self._refresh_region_controls()
            status = _("Created {count} editable layout regions.").format(count=len(regions))
            if warnings := _layout_visibility_warnings(tags, regions, canvas_width, canvas_height):
                status += " " + warnings
            self._set_status(status, error=bool(warnings))
            if not warnings:
                self.panel_toggle.setChecked(False)
        except Exception as e:
            self._set_status(str(e), error=True)
        finally:
            self.generate_button.setEnabled(True)

    async def _request_llm_layout(self, prompt: str) -> str:
        connection = root.connection
        if connection.state is not ConnectionState.connected:
            raise LayoutError("Connect to a local ComfyUI server before generating a layout.")

        workflow = _openai_layout_workflow(prompt, self.model_select.currentText())
        work = WorkflowInput(
            WorkflowKind.custom,
            ImageInput.from_extent(Extent(1, 1)),
            sampling=SamplingInput("custom", "custom", 1.0, 1000, seed=0),
            inpaint=InpaintParams(InpaintMode.fill, Bounds(0, 0, 1, 1)),
            custom_workflow=CustomWorkflowInput(workflow, {}),
        )
        loop = eventloop._loop
        future: asyncio.Future[str] = loop.create_future()
        job_id = ""

        def handle_message(message):
            if message.job_id != job_id or future.done():
                return
            if message.event is ClientEvent.output and isinstance(message.result, TextOutput):
                future.set_result(message.result.text)
            elif message.event in (ClientEvent.error, ClientEvent.payment_required):
                future.set_exception(LayoutError(message.error or "Comfy layout workflow failed."))
            elif message.event is ClientEvent.finished:
                future.set_exception(LayoutError("Comfy layout workflow finished without text output."))

        binding: QMetaObject.Connection = connection.message_received.connect(handle_message)
        try:
            job_id = await connection.client.enqueue(work, front=True)
            return await asyncio.wait_for(future, timeout=180)
        finally:
            try:
                QObject.disconnect(binding)
            except TypeError:
                pass

    def _create_region_layers(self, tags, regions, canvas_width: int, canvas_height: int):
        layers = self._model.layers
        target_layer = _generation_target_layer(layers.active) or _generation_target_layer(layers.root)
        group = layers.create_group("Layout Squirrel Regions", make_active=False)
        for region in regions:
            tag = tags[region.tag_index]
            name = f"[{region.shape}] {region.tag_index}: {trim_text(tag, 80)}"
            svg = region_to_svg(region, tag, canvas_width, canvas_height)
            layer = layers.create_vector(name, svg, make_active=False, parent=group)
            prompt_region = self._model.regions.add_region_for_layer(layer)
            prompt_region.positive = tag
            prompt_region.conditioning_strength = settings.llm_layout_default_strength
            prompt_region.conditioning_feather = settings.llm_layout_default_feather / 100
            prompt_region.color_hint_strength = settings.llm_layout_color_hint_strength
            prompt_region.hint_colors = list(region.hint_colors)
            prompt_region.full_strength_mask = True
        if target_layer is not None:
            _restore_active_layer(layers, target_layer)

    def _refresh_region_controls(self):
        _clear_layout(self.region_controls_layout)
        regions = _layout_squirrel_regions(self._model)
        if len(regions) == 0:
            self.region_controls_toggle.setVisible(False)
            self.region_controls_scroll.setVisible(False)
            return

        self.region_controls_toggle.setVisible(True)
        self.region_controls_scroll.setVisible(self.region_controls_toggle.isChecked())
        self.region_controls_toggle.setText(_("Region controls ({count})").format(count=len(regions)))
        self.region_controls_layout.addWidget(QLabel(_("Palette"), self), 0, 0)
        self.region_controls_layout.addWidget(QLabel(_("Region"), self), 0, 1)
        weight_header = QLabel(_("Weight"), self)
        weight_header.setToolTip(
            _("Per-region text prompt weight. Higher values make this region's prompt more forceful.")
        )
        feather_header = QLabel(_("Feather"), self)
        feather_header.setToolTip(
            _("Per-region regional prompt mask feather, as a percentage of this region's smaller dimension.")
        )
        color_header = QLabel(_("Color"), self)
        color_header.setToolTip(
            _("Per-region color hint strength. 0 disables color hints for this region.")
        )
        self.region_controls_layout.addWidget(weight_header, 0, 2)
        self.region_controls_layout.addWidget(feather_header, 0, 3)
        self.region_controls_layout.addWidget(color_header, 0, 4)

        for row, (layer, region) in enumerate(regions, start=1):
            palette = _palette_widget(_region_hint_colors(region), region, self._refresh_region_controls, self)
            palette.setToolTip(layer.name)

            prompt = QLineEdit(region.positive, self)
            prompt.setToolTip(_("Regional prompt text for this shape."))
            if warning := _layer_visibility_warning(layer):
                prompt.setToolTip(warning)
                prompt.setStyleSheet("border: 1px solid #d09040;")
            prompt.textChanged.connect(lambda text, r=region: setattr(r, "positive", text))

            weight = QDoubleSpinBox(self)
            weight.setDecimals(2)
            weight.setRange(0.0, 5.0)
            weight.setSingleStep(0.1)
            weight.setSuffix("x")
            weight.setValue(region.conditioning_strength)
            weight.setToolTip(weight_header.toolTip())
            weight.valueChanged.connect(
                lambda value, r=region: setattr(r, "conditioning_strength", value)
            )

            feather = QDoubleSpinBox(self)
            feather.setDecimals(0)
            feather.setRange(0.0, 200.0)
            feather.setSingleStep(5.0)
            feather.setSuffix("%")
            feather.setValue(region.conditioning_feather * 100)
            feather.setToolTip(feather_header.toolTip())
            feather.valueChanged.connect(
                lambda value, r=region: setattr(r, "conditioning_feather", value / 100)
            )

            color = QDoubleSpinBox(self)
            color.setDecimals(2)
            color.setRange(0.0, 1.0)
            color.setSingleStep(0.05)
            color.setValue(region.color_hint_strength)
            color.setToolTip(color_header.toolTip())
            color.valueChanged.connect(
                lambda value, r=region: setattr(r, "color_hint_strength", value)
            )

            self.region_controls_layout.addWidget(palette, row, 0)
            self.region_controls_layout.addWidget(prompt, row, 1)
            self.region_controls_layout.addWidget(weight, row, 2)
            self.region_controls_layout.addWidget(feather, row, 3)
            self.region_controls_layout.addWidget(color, row, 4)

    def _set_status(self, text: str, error: bool = False):
        self.status.setText(text)
        self.status.setStyleSheet("color: #d06060;" if error else "")


def _openai_layout_workflow(prompt: str, model: str):
    return {
        "1": {
            "class_type": "OpenAIChatNode",
            "inputs": {
                "prompt": prompt,
                "persist_context": False,
                "model": model,
            },
        },
        "2": {
            "class_type": "ETN_KritaSendText",
            "inputs": {
                "value": ["1", 0],
                "name": "Layout Squirrel",
                "type": "text",
            },
        },
    }


def _generation_target_layer(active: Layer):
    if _is_llm_layout_layer(active):
        active = active.parent_layer or active

    if active.type is LayerType.paint and not _is_llm_layout_layer(active):
        return active

    for layer in active.child_layers:
        if _is_llm_layout_layer(layer):
            continue
        if layer.type is LayerType.paint:
            return layer
        if target := _generation_target_layer(layer):
            return target
    return None


def _restore_active_layer(layers, target: Layer):
    target_id = target.id

    def restore():
        if layer := layers.find(target_id):
            layers.active = layer

    restore()
    QTimer.singleShot(0, restore)
    QTimer.singleShot(100, restore)
    QTimer.singleShot(500, restore)


def _layout_squirrel_regions(model: "DocumentModel"):
    result = []
    for region in model.regions:
        layer = region.first_layer
        if layer is not None and _is_llm_layout_layer(layer):
            result.append((layer, region))
    result.sort(key=lambda item: item[0].name)
    return result


def _clear_layout(layout: QGridLayout):
    while layout.count():
        item = layout.takeAt(0)
        if widget := item.widget():
            widget.deleteLater()


def _palette_widget(colors: list[str], region, refresh, parent: QWidget):
    widget = QWidget(parent)
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    for index, color in enumerate(colors[:3]):
        swatch = QPushButton("", widget)
        swatch.setFixedSize(14, 14)
        swatch.setToolTip(_("Click to change this region hint color."))
        swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #808080;")
        swatch.clicked.connect(
            lambda _checked=False, i=index, c=color: _choose_hint_color(region, i, c, refresh, widget)
        )
        layout.addWidget(swatch)
    if len(colors) < 3:
        add_color = QPushButton("+", widget)
        add_color.setFixedSize(14, 14)
        add_color.setToolTip(_("Add another hint color for this region."))
        add_color.clicked.connect(
            lambda _checked=False: _choose_hint_color(
                region, len(_region_hint_colors(region)), "#808080", refresh, widget
            )
        )
        layout.addWidget(add_color)
    layout.addStretch(1)
    widget.setLayout(layout)
    return widget


def _choose_hint_color(region, index: int, current: str, refresh, parent: QWidget):
    color = QColorDialog.getColor(QColor(current), parent, _("Choose hint color"))
    if not color.isValid():
        return

    colors = _region_hint_colors(region)
    if index < len(colors):
        colors[index] = color.name()
    else:
        colors.append(color.name())
    region.hint_colors = _dedupe_colors(colors)[:3]
    if index == 0:
        _recolor_region_shape(region, region.hint_colors[0])
    refresh()


def _region_hint_colors(region):
    colors = _dedupe_colors(region.hint_colors)
    if not colors:
        colors = ["#808080"]
    return colors


def _dedupe_colors(colors):
    result = []
    for color in colors[:3]:
        if _is_hex_color(color):
            color = color.lower()
            if color not in result:
                result.append(color)
    return result


def _recolor_region_shape(region, color: str):
    layer = region.first_layer
    if layer is None:
        return

    replacement = _replace_region_vector_layer(region, layer, color)
    if replacement is not None:
        return

    layer.refresh()


def _replace_region_vector_layer(region, layer: Layer, color: str):
    if layer.type is not LayerType.vector:
        return None
    parent = layer.parent_layer
    if parent is None:
        return None

    bounds = layer.compute_bounds()
    if bounds.area == 0:
        return None

    canvas = layer._manager.image_extent
    shape = _shape_from_layer_name(layer.name)
    svg = _bounds_to_svg(shape, bounds, color, canvas)
    replacement = layer._manager.create_vector(
        layer.name,
        svg,
        make_active=False,
        parent=parent,
        above=layer,
    )
    replacement.is_visible = layer.is_visible
    replacement.is_locked = layer.is_locked
    region.unlink(layer)
    region.link(replacement)
    layer.remove()
    return replacement


def _shape_from_layer_name(name: str):
    return "ellipse" if name.startswith("[ellipse]") else "rect"


def _bounds_to_svg(shape: str, bounds: Bounds, color: str, canvas: Extent):
    safe_color = escape(color, quote=True)
    common = (
        f'fill="{safe_color}" fill-opacity="0.72" '
        f'stroke="{safe_color}" stroke-opacity="1" stroke-width="3"'
    )
    if shape == "ellipse":
        element = (
            f'<ellipse cx="{bounds.x + bounds.width / 2:.3f}" '
            f'cy="{bounds.y + bounds.height / 2:.3f}" '
            f'rx="{bounds.width / 2:.3f}" ry="{bounds.height / 2:.3f}" {common}/>'
        )
    else:
        element = (
            f'<rect x="{bounds.x:.3f}" y="{bounds.y:.3f}" '
            f'width="{bounds.width:.3f}" height="{bounds.height:.3f}" {common}/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width}" height="{canvas.height}" '
        f'viewBox="0 0 {canvas.width} {canvas.height}">{element}</svg>'
    )


def _layout_visibility_warnings(tags, regions, canvas_width: int, canvas_height: int):
    canvas_area = max(1, canvas_width * canvas_height)
    names = []
    for region in regions:
        x, y, width, height = region.box
        near_edge = x <= 1 or y <= 1 or x + width >= canvas_width - 1 or y + height >= canvas_height - 1
        too_small = (width * height) / canvas_area < 0.005 or width < canvas_width * 0.03 or height < canvas_height * 0.03
        if near_edge and too_small:
            names.append(tags[region.tag_index])
    if not names:
        return ""
    return _("Warning: these regions may be mostly off-canvas or too small to see: {names}.").format(
        names=", ".join(names)
    )


def _layer_visibility_warning(layer: Layer):
    bounds = layer.compute_bounds()
    if bounds.area == 0:
        return _("This region layer has no visible pixels.")
    canvas = layer._manager.image_extent
    near_edge = (
        bounds.x <= 1
        or bounds.y <= 1
        or bounds.x + bounds.width >= canvas.width - 1
        or bounds.y + bounds.height >= canvas.height - 1
    )
    too_small = (
        bounds.area / max(1, canvas.width * canvas.height) < 0.005
        or bounds.width < canvas.width * 0.03
        or bounds.height < canvas.height * 0.03
    )
    if near_edge and too_small:
        return _("This region is very small and touches the canvas edge; it may be mostly off-canvas.")
    return ""


def _is_hex_color(color: object):
    if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
        return False
    try:
        int(color[1:], 16)
        return True
    except ValueError:
        return False


def _is_llm_layout_layer(layer: Layer):
    if layer.name.startswith(LAYOUT_REGION_GROUP_NAMES):
        return True
    parent = layer.parent_layer
    while parent is not None and not parent.is_root:
        if parent.name.startswith(LAYOUT_REGION_GROUP_NAMES):
            return True
        parent = parent.parent_layer
    return False
