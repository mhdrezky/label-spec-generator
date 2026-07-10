"""Node 3 — decompose sheet into physical plate regions (OpenCV + LLM mm labels)."""

from __future__ import annotations

from pathlib import Path

from context import SheetContext
from plate_detect import detect_plates, file_image_px
from plate_gate import evaluate_gate
from schema import DECOMPOSE_FRAC_SCHEMA, PLATE_COUNT_SCHEMA, PLATE_DIMS_SCHEMA
from vision import call_text, call_vision, guess_mime, image_to_base64

PLATE_COUNT_PROMPT = """\
Count the physical label PLATES to manufacture in this draft — each is a separate
rectangle cut as its own piece.
- Stacked horizontal rows, each with its own outline/height, are ONE plate per row.
- Vertical column guides INSIDE one row do NOT create separate plates.
Return only the integer plate_count.
"""

LLM_DECOMPOSE_PROMPT = """\
This label draft image is exactly {width}×{height} pixels.

Find every physical label PLATE to manufacture — one entry per rectangle that
would be cut as a separate piece.
- STACKED ROWS: each horizontal strip with its own outline is ONE plate per row.
- COLUMN GUIDES inside one row do NOT split it into separate plates.

For each plate return:
- id: 1, 2, 3... top-to-bottom then left-to-right
- bbox_frac: [x1, y1, x2, y2] as FRACTIONS of the image in [0, 1]
  (x = column/width, y = row/height; NEVER pixels)
- width_mm / height_mm: copy ONLY written mm numbers; null if not stated

{qc_hint}
"""

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


def _llm_plate_count(image_path: str | Path, ctx: SheetContext) -> int | None:
    """Cheap coarse plate-count yardstick (one vision call, integer only)."""
    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    result = call_vision(
        PLATE_COUNT_PROMPT,
        [(image_b64, mime)],
        schema=PLATE_COUNT_SCHEMA,
        label="plate-count",
        max_tokens=200,
    )
    if result.get("error") == "parse_failed":
        ctx.warnings.append("plate-count: invalid JSON — yardstick unavailable")
        return None
    count = result.get("plate_count")
    return count if isinstance(count, int) else None


def _llm_decompose(
    image_path: str | Path, ctx: SheetContext, *, qc_hint: str | None = None
) -> list[dict]:
    """LLM decompose fallback. Emits FRACTIONAL [0,1] coords only; code converts
    to file pixels and drops any out-of-range box — no absolute-pixel path."""
    w = ctx.image_px.get("width") or 0
    h = ctx.image_px.get("height") or 0
    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    hint_block = f"QC note: {qc_hint}" if qc_hint else ""
    prompt = LLM_DECOMPOSE_PROMPT.format(width=w, height=h, qc_hint=hint_block)
    result = call_vision(
        prompt,
        [(image_b64, mime)],
        schema=DECOMPOSE_FRAC_SCHEMA,
        label="decompose-llm",
        max_tokens=8000,
    )
    if result.get("error") == "parse_failed":
        ctx.warnings.append("decompose-llm: invalid JSON response")
        return []

    plates: list[dict] = []
    dropped = 0
    for raw in result.get("plates") or []:
        frac = raw.get("bbox_frac")
        if not (
            isinstance(frac, list)
            and len(frac) == 4
            and all(isinstance(v, (int, float)) for v in frac)
        ):
            dropped += 1
            continue
        if not all(0.0 <= v <= 1.0 for v in frac):
            dropped += 1
            continue
        x1, y1, x2, y2 = frac
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        plates.append({
            "id": len(plates) + 1,
            "bbox_px": [round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)],
            "width_mm": raw.get("width_mm"),
            "height_mm": raw.get("height_mm"),
        })
    if dropped:
        ctx.warnings.append(
            f"decompose-llm: dropped {dropped} plate(s) with invalid/out-of-range coords"
        )
    return plates


def run_decompose(
    image_path: str | Path,
    ctx: SheetContext,
    *,
    qc_hint: str | None = None,
    force_llm: bool = False,
) -> SheetContext:
    """Hybrid decompose. CV first; a measurable-without-GT gate (or a QC-driven
    ``force_llm``) routes to the fractional LLM fallback when CV is distrusted."""
    image_path = Path(image_path)
    ctx.image_px = file_image_px(image_path)

    plates, meta = detect_plates(image_path)
    if meta.get("error"):
        ctx.warnings.append(f"decompose: CV detection failed ({meta['error']})")
        plates = []

    # Compute the yardstick once; the gate and QC (Phase 3) reuse ctx.llm_count.
    if ctx.llm_count is None:
        ctx.llm_count = _llm_plate_count(image_path, ctx)

    gate = evaluate_gate(plates, ctx.image_px, ctx.llm_count)
    ctx.gate = gate

    if plates and gate["trust_cv"] and not force_llm:
        ctx.decompose_method = "opencv"
        ctx.warnings.append(
            f"decompose: OpenCV {len(plates)} plates trusted "
            f"(llm≈{ctx.llm_count}, {ctx.image_px['width']}×{ctx.image_px['height']}px)"
        )
        _annotate_plate_dimensions(image_path, ctx, plates, qc_hint=qc_hint)
        ctx.plate_regions = plates
        return ctx

    reason = "QC forced LLM" if force_llm else "; ".join(gate["reasons"]) or "no plates"
    ctx.decompose_method = "llm"
    ctx.warnings.append(f"decompose: CV distrusted ({reason}) → LLM fallback")
    llm_plates = _llm_decompose(image_path, ctx, qc_hint=qc_hint)
    if not llm_plates:
        ctx.warnings.append("decompose: LLM fallback produced no plates")
    ctx.plate_regions = llm_plates
    return ctx
