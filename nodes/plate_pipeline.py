"""Nodes 4-6 — per-plate transcribe, position, and size."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from context import SheetContext
from schema import POSITION_SCHEMA, SIZE_SCHEMA, TRANSCRIBE_SCHEMA
from vision import call_text, call_vision, crop_plate, pil_to_base64

MAX_PLATES = 60

TRANSCRIBE_PROMPT = """\
This crop is plate {plate_id} of {total} from a label draft.
Plate size: {width} mm wide × {height} mm tall.

Transcribe every text string on this plate, top-to-bottom order.
Side-by-side texts on different rows are SEPARATE lines (e.g. "DISABLE" and
"ENABLE" are two lines, not one).
Return each line with tight bbox_px [x1,y1,x2,y2] in THIS crop's pixels.
"""

SIZE_PROMPT = """\
Plate {plate_id} of {total}. Plate size: {width} mm wide × {height} mm tall.{slot_context}

Estimate CAPITAL letter height (size_mm) for each text line below.
Typical engraved labels: 3–8mm for body text, up to ~12mm for large titles.

Texts:
{texts}
"""

POSITION_PROMPT = """\
This crop is plate {plate_id} of {total}. Size: {width} mm × {height} mm.

Give the CENTER of each text: x_mm from left edge, y_mm from top edge.
x_mm in [0, {width}], y_mm in [0, {height}].

Texts (with sizes):
{texts}
"""


def _texts_block(lines: list[dict]) -> str:
    return "\n".join(
        f"  line {i + 1}: {ln.get('text', '')!r}"
        for i, ln in enumerate(lines)
    )


def _texts_with_sizes(lines: list[dict]) -> str:
    return "\n".join(
        f"  line {i + 1}: {ln.get('text', '')!r} (~{ln.get('size_mm', '?')}mm)"
        for i, ln in enumerate(lines)
    )


def _plate_context_line(region: dict, total: int) -> dict:
    return {
        "plate_id": region.get("id", "?"),
        "total": total,
        "width": region.get("width_mm", "?"),
        "height": region.get("height_mm", "?"),
    }


def _new_label_from_region(region: dict, ctx: SheetContext) -> dict:
    return {
        "label_number": region.get("id"),
        "width_mm": region.get("width_mm"),
        "height_mm": region.get("height_mm"),
        "quantity": None,
        "material": ctx.material_notes,
        "background_color": None,
        "text_color": None,
        "fixing": None,
        "notes": None,
        "bbox_px": region.get("bbox_px"),
        "lines": [],
        "holes": [],
    }


def _transcribe_plate(
    crop_b64: str, region: dict, total: int, warnings: list[str]
) -> list[dict]:
    ctx = _plate_context_line(region, total)
    prompt = TRANSCRIBE_PROMPT.format(**ctx)
    result = call_vision(
        prompt,
        [(crop_b64, "image/png")],
        schema=TRANSCRIBE_SCHEMA,
        label=f"transcribe#{ctx['plate_id']}",
        max_tokens=4000,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {ctx['plate_id']}: transcribe invalid JSON")
        return []
    lines = []
    for i, ln in enumerate(result.get("lines") or []):
        lines.append({
            "text": ln.get("text", ""),
            "x_mm": None,
            "y_mm": None,
            "size_mm": None,
            "alignment": None,
            "bold": None,
            "bbox_px": ln.get("bbox_px"),
        })
    return lines


def _size_plate(
    region: dict, lines: list[dict], total: int, warnings: list[str]
) -> None:
    ctx = _plate_context_line(region, total)
    slot_context = ""
    height = region.get("height_mm")
    n = len(lines)
    if n > 1 and isinstance(height, (int, float)):
        slot = float(height) / n
        slot_context = f" {n} stacked rows — ~{slot:.1f}mm per row."
    prompt = SIZE_PROMPT.format(
        **ctx,
        slot_context=slot_context,
        texts=_texts_block(lines),
    )
    result = call_text(
        prompt,
        schema=SIZE_SCHEMA,
        label=f"size#{ctx['plate_id']}",
        max_tokens=1500,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {ctx['plate_id']}: size invalid JSON")
        return
    by_line = {row.get("line"): row.get("size_mm") for row in result.get("lines") or []}
    for i, ln in enumerate(lines):
        if (i + 1) in by_line:
            ln["size_mm"] = by_line[i + 1]


def _position_plate(
    crop_b64: str, region: dict, lines: list[dict], total: int, warnings: list[str]
) -> None:
    ctx = _plate_context_line(region, total)
    prompt = POSITION_PROMPT.format(
        **ctx,
        texts=_texts_with_sizes(lines),
    )
    result = call_vision(
        prompt,
        [(crop_b64, "image/png")],
        schema=POSITION_SCHEMA,
        label=f"pos#{ctx['plate_id']}",
        max_tokens=3000,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {ctx['plate_id']}: position invalid JSON")
        return
    by_line = {
        row.get("line"): (row.get("x_mm"), row.get("y_mm"))
        for row in result.get("lines") or []
    }
    for i, ln in enumerate(lines):
        if (i + 1) in by_line:
            ln["x_mm"], ln["y_mm"] = by_line[i + 1]


def _process_one_plate(
    image: Image.Image,
    ctx: SheetContext,
    region: dict,
    total: int,
    *,
    transcribe: bool = True,
    position: bool = True,
    size: bool = True,
) -> dict:
    crop = crop_plate(
        image,
        region.get("bbox_px") or [0, 0, image.width, image.height],
        ctx.image_px,
    )
    crop_b64 = pil_to_base64(crop)

    existing = next(
        (lab for lab in ctx.labels if lab.get("label_number") == region.get("id")),
        None,
    )
    label = existing or _new_label_from_region(region, ctx)

    if transcribe:
        label["lines"] = _transcribe_plate(crop_b64, region, total, ctx.warnings)
    if size and label.get("lines"):
        _size_plate(region, label["lines"], total, ctx.warnings)
    if position and label.get("lines"):
        _position_plate(crop_b64, region, label["lines"], total, ctx.warnings)

    return label


def run_all_plates(
    image_path: str | Path,
    ctx: SheetContext,
    *,
    plate_ids: list[int] | None = None,
    transcribe: bool = True,
    position: bool = True,
    size: bool = True,
) -> SheetContext:
    regions = ctx.plate_regions[:MAX_PLATES]
    if plate_ids is not None:
        id_set = set(plate_ids)
        regions = [r for r in regions if r.get("id") in id_set]
    total = len(ctx.plate_regions)
    image = Image.open(image_path)

    labels_by_id = {lab.get("label_number"): lab for lab in ctx.labels}
    for region in regions:
        label = _process_one_plate(
            image,
            ctx,
            region,
            total,
            transcribe=transcribe,
            position=position,
            size=size,
        )
        labels_by_id[label["label_number"]] = label

    ctx.labels = [
        labels_by_id[r["id"]]
        for r in ctx.plate_regions[:MAX_PLATES]
        if r["id"] in labels_by_id
    ]
    return ctx
