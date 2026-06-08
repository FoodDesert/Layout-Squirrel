from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from PyQt5.QtCore import QMetaObject, QObject, Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import eventloop
from ..client import ClientEvent, TextOutput
from ..connection import ConnectionState
from ..layout_regions import (
    LayoutError,
    build_layout_prompt,
    parse_region_tags,
    region_to_svg,
    validate_layout_response,
)
from ..localization import translate as _
from ..root import root
from ..settings import settings
from ..layer import Layer, LayerType
from ..util import trim_text

if TYPE_CHECKING:
    from ..model import Model

LAYOUT_REGION_GROUP_NAMES = ("Layout Squirrel Regions", "LLM Layout Regions")


class LayoutRegionWidget(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._model: Model = root.active_model
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

        self.default_strength = QDoubleSpinBox(self)
        self.default_strength.setDecimals(2)
        self.default_strength.setRange(0.0, 5.0)
        self.default_strength.setSingleStep(0.1)
        self.default_strength.setValue(settings.llm_layout_default_strength)
        self.default_strength.setSuffix("x")
        self.default_strength.setToolTip(_("Default regional conditioning strength"))
        self.default_strength.valueChanged.connect(self._save_defaults)

        self.default_feather = QDoubleSpinBox(self)
        self.default_feather.setDecimals(0)
        self.default_feather.setRange(0.0, 200.0)
        self.default_feather.setSingleStep(5.0)
        self.default_feather.setValue(settings.llm_layout_default_feather)
        self.default_feather.setSuffix("%")
        self.default_feather.setToolTip(
            _("Default conditioning feather, as a percentage of each region's smaller dimension")
        )
        self.default_feather.valueChanged.connect(self._save_defaults)

        self.color_hint = QDoubleSpinBox(self)
        self.color_hint.setDecimals(2)
        self.color_hint.setRange(0.0, 1.0)
        self.color_hint.setSingleStep(0.05)
        self.color_hint.setValue(settings.llm_layout_color_hint_denoise)
        self.color_hint.setToolTip(
            _("Img2img denoise for the region-color hint; lower preserves the color layout more strongly, 1 disables it")
        )
        self.color_hint.valueChanged.connect(self._save_defaults)

        self.generate_button = QPushButton(_("Generate Layout"), self)
        self.generate_button.clicked.connect(self.generate_layout)

        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(2)
        controls.addWidget(QLabel(_("LLM"), self), 0, 0)
        controls.addWidget(QLabel(_("Prompt weight"), self), 0, 1)
        controls.addWidget(QLabel(_("Mask feather"), self), 0, 2)
        controls.addWidget(QLabel(_("Color denoise"), self), 0, 3)
        controls.addWidget(self.model_select, 1, 0)
        controls.addWidget(self.default_strength, 1, 1)
        controls.addWidget(self.default_feather, 1, 2)
        controls.addWidget(self.color_hint, 1, 3)
        controls.addWidget(self.generate_button, 1, 4)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel(_("Layout Squirrel"), self))
        layout.addWidget(self.description)
        layout.addWidget(self.tags)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        self.setLayout(layout)
        self._apply_base_prompt_default()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: "Model"):
        self._model = model
        self._apply_base_prompt_default()

    def _save_defaults(self, *ignored):
        settings.llm_layout_description = self.description.toPlainText()
        settings.llm_layout_tags = self.tags.text()
        settings.llm_layout_model = self.model_select.currentText()
        settings.llm_layout_default_strength = self.default_strength.value()
        settings.llm_layout_default_feather = self.default_feather.value()
        settings.llm_layout_color_hint_denoise = self.color_hint.value()
        settings.save()

    def _apply_base_prompt_default(self):
        prompt = self._model.regions.positive.strip()
        if prompt == "":
            self._model.regions.positive = settings.llm_layout_base_prompt
        elif prompt == "duo, white background, chibi":
            self._model.regions.positive = settings.llm_layout_base_prompt

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
            self._set_status(_("Created {count} editable layout regions.").format(count=len(regions)))
        except Exception as e:
            self._set_status(str(e), error=True)
        finally:
            self.generate_button.setEnabled(True)

    async def _request_llm_layout(self, prompt: str) -> str:
        connection = root.connection
        if connection.state is not ConnectionState.connected:
            raise LayoutError("Connect to a local ComfyUI server before generating a layout.")

        client = connection.client
        if not hasattr(client, "enqueue_raw_workflow"):
            raise LayoutError("The connected backend cannot run Comfy Partner/API node workflows.")

        workflow = _openai_layout_workflow(prompt, self.model_select.currentText())
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
            job_id = await client.enqueue_raw_workflow(workflow, front=True)
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
            name = f"{region.tag_index}: {trim_text(tag, 80)}"
            svg = region_to_svg(region, tag, canvas_width, canvas_height)
            layer = layers.create_vector(name, svg, make_active=False, parent=group)
            prompt_region = self._model.regions.add_region_for_layer(layer)
            prompt_region.positive = tag
            prompt_region.conditioning_strength = 1.0
            prompt_region.conditioning_feather = self.default_feather.value() / 100
            prompt_region.full_strength_mask = True
        if target_layer is not None:
            _restore_active_layer(layers, target_layer)

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


def _is_llm_layout_layer(layer: Layer):
    if layer.name.startswith(LAYOUT_REGION_GROUP_NAMES):
        return True
    parent = layer.parent_layer
    while parent is not None and not parent.is_root:
        if parent.name.startswith(LAYOUT_REGION_GROUP_NAMES):
            return True
        parent = parent.parent_layer
    return False
