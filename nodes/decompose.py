"""Node 3 — decompose sheet into physical plate regions (OpenCV + LLM mm labels)."""

from __future__ import annotations

from pathlib import Path

from context import SheetContext
from plate_detect import detect_plates, file_image_px
from schema import PLATE_DIMS_SCHEMA
from vision import call_text, call_vision, guess_mime, image_to_base64

PLATE_DIMS_PROMPT = """\
This label draft image is exactly {width}×{height} pixels.

Computer vision already detected these plate outlines (pixel coordinates):
{plate_list}

For each plate id, read ONLY written mm dimension numbers on the drawing that
state that plate's width and height. Return null when not clearly annotated.

Rules:
- STACKED ROWS with repeated height labels (e.g. "20") are one plate per row.
- COLUMN GUIDES inside one row do NOT create separate width dimensions.
- Do NOT invent or change bounding boxes.

{qc_hint}
"""


def _plate_list_text(plates: list[dict]) -> str:
    lines = []
    for plate in plates:
        bbox = plate.get("bbox_px") or []
        lines.append(f"  #{plate.get('id')}: bbox_px={bbox}")
    return "\n".join(lines) or "  (none)"


def _annotate_plate_dimensions(
    image_path: str | Path,
    ctx: SheetContext,
    plates: list[dict],
    *,
    qc_hint: str | None = None,
) -> None:
    if not plates:
        return

    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    w = ctx.image_px.get("width", "?")
    h = ctx.image_px.get("height", "?")
    hint_block = f"QC note: {qc_hint}" if qc_hint else ""
    prompt = PLATE_DIMS_PROMPT.format(
        width=w,
        height=h,
        plate_list=_plate_list_text(plates),
        qc_hint=hint_block,
    )
    result = call_vision(
        prompt,
        [(image_b64, mime)],
        schema=PLATE_DIMS_SCHEMA,
        label="plate-dims",
        max_tokens=4000,
    )
    if result.get("error") == "parse_failed":
        ctx.warnings.append("plate-dims: invalid JSON — mm sizes may be missing")
        return

    by_id = {row.get("id"): row for row in result.get("plates") or []}
    for plate in plates:
        row = by_id.get(plate.get("id"))
        if not row:
            continue
        if row.get("width_mm") is not None:
            plate["width_mm"] = row["width_mm"]
        if row.get("height_mm") is not None:
            plate["height_mm"] = row["height_mm"]


def run_decompose(
    image_path: str | Path,
    ctx: SheetContext,
    *,
    qc_hint: str | None = None,
) -> SheetContext:
    image_path = Path(image_path)
    ctx.image_px = file_image_px(image_path)

    plates, meta = detect_plates(image_path)
    if meta.get("error"):
        ctx.warnings.append(f"decompose: CV detection failed ({meta['error']})")
        return ctx

    if not plates:
        ctx.warnings.append("decompose: no plates detected")
        return ctx

    ctx.warnings.append(
        f"decompose: OpenCV detected {len(plates)} plates "
        f"({ctx.image_px['width']}×{ctx.image_px['height']}px)"
    )
    _annotate_plate_dimensions(image_path, ctx, plates, qc_hint=qc_hint)
    ctx.plate_regions = plates
    return ctx
