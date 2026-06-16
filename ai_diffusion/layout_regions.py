from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class LayoutRegion:
    tag_index: int
    shape: str
    box: tuple[float, float, float, float]
    z_index: int
    display_color: str
    hint_colors: tuple[str, ...]


class LayoutError(ValueError):
    pass


_palette = [
    "#d55e00",
    "#0072b2",
    "#009e73",
    "#cc79a7",
    "#f0e442",
    "#56b4e9",
    "#e69f00",
    "#8a63d2",
]


def parse_region_tags(text: str) -> list[str]:
    try:
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace = ","
        lexer.whitespace_split = True
        lexer.commenters = ""
        tags = list(lexer)
    except ValueError as e:
        raise LayoutError(f"Could not parse region tags: {e}") from e
    tags = [tag.strip() for tag in tags]
    tags = [tag for tag in tags if tag]
    if not tags:
        raise LayoutError("Enter at least one comma-separated region tag.")
    return tags


def build_layout_prompt(
    layout_description: str,
    tags: list[str],
    canvas_width: int,
    canvas_height: int,
) -> str:
    tag_lines = "\n".join(f"{i}: {tag}" for i, tag in enumerate(tags))
    description = layout_description.strip() or "(none)"
    return f"""You are a spatial layout planner for a Krita regional prompting tool.

Return only valid JSON. Do not use markdown.

Canvas:
- width: {canvas_width}
- height: {canvas_height}
- origin: top-left
- coordinates: normalized numbers from 0.0 to 1.0

Optional layout description:
{description}

Required region tags:
{tag_lines}

Rules:
- Create exactly one region for every tag_index listed above.
- Never add regions for objects mentioned only in the layout description.
- Do not rewrite, combine, split, or invent tags.
- Use the layout description only for placement, size, depth, and grouping.
- Supported shape values are "rect" and "ellipse".
- box is [x_min, y_min, width, height] in normalized coordinates.
- z_index is visual depth: lower values are farther back, higher values are closer.
- hint_colors is an ordered list of 1 to 3 hex colors, from most prominent to least prominent, implied by the region tags.
- display_color must be the first hint_colors color and is used as the Krita guide fill color.
- Use explicit color tags when present: for "squirrel, pink fur", choose pink as the first hint color rather than averaging squirrel colors.
- If tags imply multiple important visible colors, include them in hint_colors. For example, red fur and brown clothes should use two colors.
- If the region tags do not include an explicit color, infer a plausible average color for the object or background.
- The hint colors may be used as weak generation color hints.
- Background regions such as sky, ground, water, wall, forest, or mountains should usually be large and far back.
- Related object parts may touch or overlap if that helps composition.

JSON schema:
{{
  "canvas": {{"width": {canvas_width}, "height": {canvas_height}}},
  "regions": [
    {{
      "tag_index": 0,
      "shape": "ellipse",
      "box": [0.1, 0.2, 0.35, 0.5],
      "z_index": 10,
      "display_color": "#d55e00",
      "hint_colors": ["#d55e00", "#8b5a2b"]
    }}
  ]
}}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise LayoutError("The LLM response did not contain a JSON object.")
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as e:
            raise LayoutError(f"The LLM response was not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise LayoutError("The LLM response must be a JSON object.")
    return data


def validate_layout_response(
    text: str,
    tags: list[str],
    canvas_width: int,
    canvas_height: int,
) -> list[LayoutRegion]:
    data = extract_json_object(text)
    regions = data.get("regions")
    if not isinstance(regions, list):
        raise LayoutError("The LLM response must contain a regions list.")

    seen: set[int] = set()
    result: list[LayoutRegion] = []
    for i, item in enumerate(regions):
        if not isinstance(item, dict):
            raise LayoutError(f"Region {i} must be an object.")
        tag_index = _require_int(item, "tag_index", f"Region {i}")
        if tag_index < 0 or tag_index >= len(tags):
            raise LayoutError(f"Region {i} has unknown tag_index {tag_index}.")
        if tag_index in seen:
            raise LayoutError(f"Tag index {tag_index} appears more than once.")
        seen.add(tag_index)

        shape = str(item.get("shape", "")).lower()
        if shape not in {"rect", "ellipse"}:
            raise LayoutError(f"Region {i} has unsupported shape '{shape}'.")

        box = _coerce_box(item.get("box"), canvas_width, canvas_height, f"Region {i}")
        z_index = _optional_int(item, "z_index", tag_index)
        hint_colors = _coerce_hint_colors(item.get("hint_colors"), item.get("display_color"), tag_index)
        color = hint_colors[0]
        result.append(LayoutRegion(tag_index, shape, box, z_index, color, hint_colors))

    missing = [i for i in range(len(tags)) if i not in seen]
    if missing:
        raise LayoutError(f"Missing regions for tag indices: {missing}.")
    return sorted(result, key=lambda r: (r.z_index, r.tag_index))


def region_to_svg(region: LayoutRegion, tag: str, canvas_width: int, canvas_height: int) -> str:
    x, y, width, height = region.box
    color = escape(region.display_color, quote=True)
    title = escape(tag, quote=False)
    common = (
        f'fill="{color}" fill-opacity="0.72" '
        f'stroke="{color}" stroke-opacity="1" stroke-width="3"'
    )
    if region.shape == "ellipse":
        shape = (
            f'<ellipse cx="{x + width / 2:.3f}" cy="{y + height / 2:.3f}" '
            f'rx="{width / 2:.3f}" ry="{height / 2:.3f}" {common}/>'
        )
    else:
        shape = f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" {common}/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">'
        f"<title>{title}</title>{shape}</svg>"
    )


def _require_int(item: dict[str, Any], key: str, context: str) -> int:
    value = item.get(key)
    if not isinstance(value, int):
        raise LayoutError(f"{context} must contain integer {key}.")
    return value


def _optional_int(item: dict[str, Any], key: str, default: int) -> int:
    value = item.get(key, default)
    return value if isinstance(value, int) else default


def _coerce_box(value: Any, canvas_width: int, canvas_height: int, context: str):
    if not isinstance(value, list) or len(value) != 4:
        raise LayoutError(f"{context} must contain box [x, y, width, height].")
    if not all(isinstance(v, (int, float)) for v in value):
        raise LayoutError(f"{context} box values must be numeric.")
    x, y, width, height = [float(v) for v in value]

    if max(abs(x), abs(y), abs(width), abs(height)) <= 1.5:
        x *= canvas_width
        width *= canvas_width
        y *= canvas_height
        height *= canvas_height

    x = max(0.0, min(x, canvas_width - 1))
    y = max(0.0, min(y, canvas_height - 1))
    width = max(1.0, min(width, canvas_width - x))
    height = max(1.0, min(height, canvas_height - y))
    return x, y, width, height


def _coerce_color(value: Any, tag_index: int) -> str:
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value.lower()
    return _palette[tag_index % len(_palette)]


def _coerce_hint_colors(value: Any, display_color: Any, tag_index: int) -> tuple[str, ...]:
    colors: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", item):
                color = item.lower()
                if color not in colors:
                    colors.append(color)
            if len(colors) >= 3:
                break
    if not colors:
        colors.append(_coerce_color(display_color, tag_index))
    return tuple(colors)
