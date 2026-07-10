"""Node 7 — sheet-level quality check."""

from pathlib import Path

from context import SheetContext
from schema import SHEET_QC_SCHEMA
from vision import call_vision, guess_mime, image_to_base64

SHEET_QC_PROMPT = """\
Review this label draft and the extracted specification summary below.

Check:
- plate_count_ok: does the plate count match what you see?
- decompose_ok: are plates whole rows/outlines (not split per column)?
- verdict: "pass" if acceptable, "revise" only for CLEAR structural errors
- fix: "decompose" if wrong plate count/split; "position" or "size" if a few
  plates have wrong geometry; null when pass
- plate_ids: list of plate ids needing position/size fix (null if pass/decompose)
- notes: one short sentence or null

Spec summary:
{spec_summary}
"""


def _spec_summary(ctx: SheetContext) -> str:
    lines = [f"Plates: {len(ctx.labels)}"]
    for label in ctx.labels[:30]:
        texts = ", ".join(
            (ln.get("text") or "")[:20] for ln in (label.get("lines") or [])[:4]
        )
        w, h = label.get("width_mm"), label.get("height_mm")
        lines.append(
            f"  #{label.get('label_number')}: {w}x{h}mm, "
            f"{len(label.get('lines') or [])} lines — {texts}"
        )
    if len(ctx.labels) > 30:
        lines.append(f"  ... and {len(ctx.labels) - 30} more")
    return "\n".join(lines)


def run_sheet_qc(image_path: str | Path, ctx: SheetContext) -> SheetContext:
    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    prompt = SHEET_QC_PROMPT.format(spec_summary=_spec_summary(ctx))
    result = call_vision(
        prompt,
        [(image_b64, mime)],
        schema=SHEET_QC_SCHEMA,
        label="qc",
        max_tokens=1500,
    )
    if result.get("error") == "parse_failed":
        ctx.warnings.append("sheet QC: invalid JSON — accepting output")
        ctx.qc_result = {"verdict": "pass", "fix": None}
        return ctx

    ctx.qc_result = result
    if result.get("verdict") != "pass" and result.get("notes"):
        ctx.warnings.append(f"sheet QC: {result.get('notes')}")
    return ctx
