"""Deterministic plate outline detection from CAD-style drafts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 12
MIN_PLATE_AREA = 800
LINE_THRESHOLD_FRAC = 0.22
EDGE_MARGIN_PX = 10
INNER_PAD_PX = 2
# A plate outline spans at least this fraction of the drawing width. The old
# 0.45/0.55 two-branch gate (coupled to an absolute bh>=50/<=50 split) assumed
# every plate was ~half the sheet wide and killed CAD sheets whose plate column
# sits inside a margin (marshall plates measured 0.43w -> all dropped). Phase-1
# attrition confirmed the strip gate, not MIN_*, was the killer; this single
# relative gate is the de-overfit. Narrower grids (traffolyte/mla) that still
# fall below it are the Phase-2 gate's job to route to the LLM, not a place to
# add per-layout branches here.
STRIP_MIN_WIDTH_FRAC = 0.33
# Reject a box larger than this fraction of the sheet: it is almost always a
# FRAME enclosing the real plates (image003's lower grid bbox = 0.65, marshall's
# table border = 0.79), not a plate. But a single-plate draft's plate legitimately
# fills up to half the sheet (drawing.png = 0.50), which the old 0.35 cap wrongly
# dropped → CV returned 0 → LLM fallback mis-boxed it and lost the "Switchboard"
# line. 0.60 sits in the gap: keeps single plates (≤0.50), rejects frames (≥0.65).
MAX_PLATE_AREA_FRAC = 0.60


def _cluster_positions(indices: np.ndarray, min_gap: int = 4) -> list[int]:
    if len(indices) == 0:
        return []
    clusters: list[int] = []
    start = int(indices[0])
    prev = int(indices[0])
    for raw in indices[1:]:
        idx = int(raw)
        if idx - prev > min_gap:
            clusters.append((start + prev) // 2)
            start = idx
        prev = idx
    clusters.append((start + prev) // 2)
    return clusters


def _line_positions(projection: np.ndarray, threshold_frac: float) -> list[int]:
    if projection.size == 0 or projection.max() == 0:
        return []
    thresh = float(projection.max()) * threshold_frac
    indices = np.where(projection >= thresh)[0]
    return _cluster_positions(indices)


def _bbox(box: tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2, y2]


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _vertical_splits_in_band(
    binary: np.ndarray, y1: int, y2: int, x1: int, x2: int
) -> list[int]:
    band_h = y2 - y1
    if band_h < MIN_PLATE_HEIGHT:
        return []
    band = binary[y1:y2, x1:x2]
    if band.size == 0:
        return []
    ver_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(3, band_h * 4 // 5))
    )
    ver = cv2.morphologyEx(band, cv2.MORPH_OPEN, ver_k)
    positions = _line_positions(ver.sum(axis=0), LINE_THRESHOLD_FRAC)
    return [
        x1 + pos
        for pos in positions
        if x1 + EDGE_MARGIN_PX < x1 + pos < x2 - EDGE_MARGIN_PX
    ]


def _split_top_band(
    binary: np.ndarray, box: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    splits = _vertical_splits_in_band(binary, y1, y2, x1, x2)
    if len(splits) < 2:
        return [box]

    xs = sorted(set([x1 + INNER_PAD_PX] + splits + [x2 - INNER_PAD_PX]))
    cols: list[tuple[int, int, int, int]] = []
    for i in range(len(xs) - 1):
        left, right = xs[i], xs[i + 1]
        if right - left < MIN_PLATE_WIDTH:
            continue
        cols.append((
            left,
            y1 + INNER_PAD_PX,
            right,
            y2 - INNER_PAD_PX,
        ))
    return cols or [box]


def _find_contour_boxes(binary: np.ndarray, w: int, h: int) -> list[tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < MIN_PLATE_WIDTH or bh < MIN_PLATE_HEIGHT:
            continue
        if bw * bh < MIN_PLATE_AREA:
            continue
        if bw > w * 0.98 and bh > h * 0.98:
            continue
        raw.append((x, y, x + bw, y + bh))

    if not raw:
        return []

    raw.sort(key=lambda b: _area(b), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in raw:
        if _area(box) > MAX_PLATE_AREA_FRAC * w * h:
            continue
        kept.append(box)
    return kept


def detect_plates(image_path: str | Path) -> tuple[list[dict], dict]:
    """Detect plate rectangles in real image pixel coordinates."""
    img = cv2.imread(str(image_path))
    if img is None:
        return [], {"error": "cannot_read_image"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    boxes = _find_contour_boxes(binary, w, h)
    plates: list[tuple[int, int, int, int]] = []

    for box in boxes:
        bw = box[2] - box[0]
        if bw <= w * STRIP_MIN_WIDTH_FRAC:
            continue
        # Always attempt a column split; _split_top_band returns [box] unchanged
        # when it finds fewer than two vertical guides, so short strips without
        # guides stay a single plate (no bh threshold needed).
        plates.extend(_split_top_band(binary, box))

    plates.sort(key=lambda b: (b[1], b[0]))

    result: list[dict] = []
    for idx, box in enumerate(plates, start=1):
        result.append({
            "id": idx,
            "bbox_px": _bbox(box),
            "width_mm": None,
            "height_mm": None,
        })

    return result, {"size": (w, h), "method": "opencv", "count": len(result)}


def file_image_px(image_path: str | Path) -> dict[str, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"width": 0, "height": 0}
    h, w = img.shape[:2]
    return {"width": w, "height": h}
