"""OpenCV plate hints for dual-call structure."""

from __future__ import annotations

from pathlib import Path

from plate_detect import detect_plates
from plate_gate import evaluate_gate


def _bbox_iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_cv_plates(image_path: str | Path, image_px: dict) -> tuple[list[dict], dict]:
    """Run deterministic OpenCV detect; return plates with bbox_frac + meta."""
    plates, meta = detect_plates(image_path)
    w = image_px.get("width") or 0
    h = image_px.get("height") or 0
    if w <= 0 or h <= 0:
        return [], meta

    out: list[dict] = []
    for plate in plates:
        bb = plate.get("bbox_px")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        out.append({
            "id": plate.get("id"),
            "bbox_px": [float(v) for v in bb],
            "bbox_frac": [bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h],
            "width_mm": plate.get("width_mm"),
            "height_mm": plate.get("height_mm"),
        })
    return out, meta


def format_cv_hint(cv_plates: list[dict]) -> str:
    if not cv_plates:
        return "  (none — no computer-vision regions detected)"
    lines = []
    for plate in cv_plates:
        frac = plate.get("bbox_frac") or []
        bb = plate.get("bbox_px") or []
        lines.append(f"  #{plate.get('id')}: bbox_frac={[round(v, 4) for v in frac]} bbox_px={bb}")
    return "\n".join(lines)


def _match_llm_mm(cv_plate: dict, llm_plates: list[dict], image_px: dict) -> tuple:
    """Copy width_mm/height_mm from best-overlap LLM plate, if any."""
    cv_bb = cv_plate.get("bbox_px")
    if not cv_bb:
        return cv_plate.get("width_mm"), cv_plate.get("height_mm")

    best_iou = 0.0
    best = None
    w = image_px.get("width") or 1
    h = image_px.get("height") or 1
    for llm in llm_plates:
        frac = llm.get("bbox_frac")
        if not (isinstance(frac, (list, tuple)) and len(frac) == 4):
            continue
        llm_bb = [frac[0] * w, frac[1] * h, frac[2] * w, frac[3] * h]
        iou = _bbox_iou(cv_bb, llm_bb)
        if iou > best_iou:
            best_iou = iou
            best = llm

    if best is None or best_iou < 0.05:
        return cv_plate.get("width_mm"), cv_plate.get("height_mm")
    return best.get("width_mm"), best.get("height_mm")


def reconcile_plates(
    structure: dict,
    cv_plates: list[dict],
    image_px: dict,
    warnings: list[str],
) -> tuple[list[dict], dict]:
    """Prefer CV outlines when the gate trusts them; merge mm labels from LLM."""
    llm_plates = structure.get("plates") or []
    gate = evaluate_gate(
        [{"bbox_px": p["bbox_px"]} for p in cv_plates],
        image_px,
        len(llm_plates),
    )

    record = {
        "trust_cv": gate.get("trust_cv"),
        "cv_count": gate.get("cv_count"),
        "llm_count": len(llm_plates),
        "reasons": gate.get("reasons") or [],
    }

    if not gate.get("trust_cv") or not cv_plates:
        warnings.append(
            "structure: using LLM plate outlines "
            f"(CV gate: cv={gate.get('cv_count')}, llm={len(llm_plates)}, "
            f"trust_cv={gate.get('trust_cv')})"
        )
        return llm_plates, record

    warnings.append(
        f"structure: using CV plate outlines ({len(cv_plates)} plates, gate trusted)"
    )
    merged: list[dict] = []
    for i, cv in enumerate(cv_plates, start=1):
        width_mm, height_mm = _match_llm_mm(cv, llm_plates, image_px)
        merged.append({
            "id": i,
            "bbox_frac": cv.get("bbox_frac"),
            "bbox_px": cv.get("bbox_px"),
            "width_mm": width_mm,
            "height_mm": height_mm,
        })
    return merged, record
