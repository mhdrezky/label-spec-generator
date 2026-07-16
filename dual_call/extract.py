"""Two-call label extraction: structure then content."""

from __future__ import annotations

from pathlib import Path

from plate_detect import file_image_px
from vision import call_vision, guess_mime, image_to_base64

from dual_call.cv_hint import detect_cv_plates, format_cv_hint, reconcile_plates
from dual_call.postprocess import merge_to_spec
from dual_call.prompts import CONTENT_PROMPT, STRUCTURE_PROMPT, format_plate_list
from dual_call.schema import CONTENT_SCHEMA, STRUCTURE_SCHEMA

CONTENT_MAX_TOKENS = 16000


def run_dual(image_path: str | Path) -> dict:
    """Run structure + content vision calls and merge to a label spec."""
    image_path = Path(image_path)
    warnings: list[str] = []
    image_px = file_image_px(image_path)
    w = image_px.get("width") or 0
    h = image_px.get("height") or 0
    if w <= 0 or h <= 0:
        raise ValueError(f"Could not read image dimensions: {image_path}")

    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)

    cv_plates, cv_meta = detect_cv_plates(image_path, image_px)
    cv_hint = format_cv_hint(cv_plates)

    structure_prompt = STRUCTURE_PROMPT.format(width=w, height=h, cv_hint=cv_hint)
    structure = call_vision(
        structure_prompt,
        [(image_b64, mime)],
        schema=STRUCTURE_SCHEMA,
        label="dual-structure",
        max_tokens=8000,
    )
    if structure.get("error") == "parse_failed":
        warnings.append("dual-structure: invalid JSON response")
        structure = {
            "draft_type": None,
            "material": None,
            "background_color": None,
            "text_color": None,
            "fixing": None,
            "default_size_mm": None,
            "line_sizes": [],
            "dimension_annotations": [],
            "plates": [],
        }

    reconciled_plates, gate = reconcile_plates(structure, cv_plates, image_px, warnings)
    structure = {**structure, "plates": reconciled_plates, "gate": gate, "cv_meta": cv_meta}

    plates_for_prompt = reconciled_plates
    content_prompt = CONTENT_PROMPT.format(
        width=w,
        height=h,
        plate_list=format_plate_list(plates_for_prompt),
    )
    content = call_vision(
        content_prompt,
        [(image_b64, mime)],
        schema=CONTENT_SCHEMA,
        label="dual-content",
        max_tokens=CONTENT_MAX_TOKENS,
    )
    if content.get("error") == "parse_failed":
        warnings.append("dual-content: invalid JSON response")
        content = {"plates": []}

    spec = merge_to_spec(structure, content, image_px, warnings)

    return {
        "structure": structure,
        "content": content,
        "spec": spec,
        "warnings": warnings,
        "image_px": image_px,
    }
