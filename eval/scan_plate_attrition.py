"""Phase 1 Step 0 — per-stage attrition probe for OpenCV plate detection.

    python eval/scan_plate_attrition.py

Reports, for every image in eval/images/, how many candidate boxes survive each
filtering stage of plate_detect.detect_plates. Purpose: localize WHICH knob kills
CAD sheets (marshall/brn94/traffolyte → 0) before changing any threshold, so the
Phase 1 decision rule is not confounded. Read-only: does not modify plate_detect.

Stages:
  raw    contours after morphology close
  min    survive MIN_PLATE_WIDTH/HEIGHT/AREA (+ full-sheet >0.98 reject)
  cap    survive area-cap 0.35*w*h  (== _find_contour_boxes output)
  strip  survive strip-width branch (bw>0.45/0.55*w) -> kept / dropped
  final  after _split_top_band  (== detect_plates output count)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plate_detect as pd  # noqa: E402
from plate_detect import (  # noqa: E402
    MIN_PLATE_AREA,
    MIN_PLATE_HEIGHT,
    MIN_PLATE_WIDTH,
    _split_top_band,
    detect_plates,
)

EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

CAD_COHORT = {
    "marshall-ave-st-leonards-msb-1-pdf-p1",
    "marshall-ave-st-leonards-msb-1-pdf-p2",
    "marshall-ave-st-leonards-msb-1-pdf-p3",
    "marshall-ave-st-leonards-msb-1-pdf-p4",
    "marshall-ave-st-leonards-msb-1-pdf-p5",
    "marshall-ave-st-leonards-msb-1-pdf-p6",
    "marshall-ave-st-leonards-msb-1-pdf-p7",
    "brn94-pdf",
    "traffolyte-pdf",
    "mla-white-red-pdf",
}


def _binary(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    return binary, w, h


def _stage_counts(image_path: Path) -> dict:
    binary, w, h = _binary(image_path)
    if binary is None:
        return {"error": "cannot_read"}

    # raw contours after morphology close (mirrors _find_contour_boxes head)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    raw = len(contours)

    after_min = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < MIN_PLATE_WIDTH or bh < MIN_PLATE_HEIGHT:
            continue
        if bw * bh < MIN_PLATE_AREA:
            continue
        if bw > w * 0.98 and bh > h * 0.98:
            continue
        after_min.append((x, y, x + bw, y + bh))

    after_cap = [b for b in after_min if pd._area(b) <= 0.35 * w * h]

    # strip-width branch (detect_plates main loop)
    strip_kept, strip_dropped = 0, 0
    dropped_wfrac = []
    plates = []
    for box in after_cap:
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        if bw > w * 0.55 and bh >= 50:
            strip_kept += 1
            plates.extend(_split_top_band(binary, box))
        elif bw > w * 0.45 and bh <= 50:
            strip_kept += 1
            plates.append(box)
        else:
            strip_dropped += 1
            dropped_wfrac.append(bw / w)

    final = len(detect_plates(image_path)[0])

    return {
        "wh": (w, h),
        "raw": raw,
        "min": len(after_min),
        "cap": len(after_cap),
        "strip_kept": strip_kept,
        "strip_dropped": strip_dropped,
        "final": final,
        "dropped_wfrac_max": max(dropped_wfrac) if dropped_wfrac else 0.0,
    }


def main() -> None:
    images = sorted(
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    header = (
        f"{'image':<44} {'WxH':<11} {'raw':>5} {'min':>4} {'cap':>4} "
        f"{'keep':>4} {'drop':>4} {'final':>5} {'dropWmax':>8}  cohort"
    )
    print(header)
    print("-" * len(header))
    for image in images:
        c = _stage_counts(image)
        tag = "CAD" if image.stem in CAD_COHORT else ""
        if c.get("error"):
            print(f"{image.name:<44} ERROR {c['error']}")
            continue
        w, h = c["wh"]
        print(
            f"{image.name:<44} {f'{w}x{h}':<11} {c['raw']:>5} {c['min']:>4} "
            f"{c['cap']:>4} {c['strip_kept']:>4} {c['strip_dropped']:>4} "
            f"{c['final']:>5} {c['dropped_wfrac_max']:>8.2f}  {tag}"
        )


if __name__ == "__main__":
    main()
