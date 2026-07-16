"""Regression benchmark against hand-verified ground truths.

    python benchmark/run_benchmark.py [--cached]

For every image in sample-data/ with a ground truth in
benchmark/expected/<name>.json, runs the dual-call pipeline and scores output.
Predictions are saved to benchmark/predictions/.

Scores per image:
- labels:    predicted label count == expected label count
- text:      % of expected text lines found (normalized exact match)
- position:  % of expected non-null mm values within +/-2mm (after measure)
- null:      % of expected-null fields still null before measure step
"""

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_call.extract import run_dual  # noqa: E402
from llm_cache import set_run_cache_dir  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent
IMAGES_DIR = PROJECT_ROOT / "sample-data"
EXPECTED_DIR = BENCHMARK_DIR / "expected"
PREDICTIONS_DIR = BENCHMARK_DIR / "predictions"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
POSITION_TOLERANCE_MM = 2.0
LINE_NUMERIC_FIELDS = ("x_mm", "y_mm", "size_mm")
LABEL_NUMERIC_FIELDS = ("width_mm", "height_mm")


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").upper()).strip()


def label_texts(label: dict) -> set[str]:
    return {norm_text(ln.get("text", "")) for ln in label.get("lines") or []}


def pair_labels(expected: list[dict], predicted: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    remaining = list(predicted)
    for exp in expected:
        exp_texts = label_texts(exp)
        best, best_score = None, -1
        for pred in remaining:
            score = len(exp_texts & label_texts(pred))
            if score > best_score:
                best, best_score = pred, score
        if best is not None:
            pairs.append((exp, best))
            remaining.remove(best)
    return pairs


def score_image(expected: dict, raw: dict, measured: dict) -> dict:
    exp_labels = expected.get("labels") or []
    cal_labels = measured.get("labels") or []
    raw_labels = raw.get("labels") or []

    text_total = text_hit = 0
    pos_total = pos_hit = 0
    null_total = null_hit = 0

    def position_check(exp_val, pred_val):
        nonlocal pos_total, pos_hit
        pos_total += 1
        if (
            isinstance(pred_val, (int, float))
            and abs(exp_val - pred_val) <= POSITION_TOLERANCE_MM
        ):
            pos_hit += 1

    def null_check(pred_val):
        nonlocal null_total, null_hit
        null_total += 1
        null_hit += pred_val is None

    for exp, pred in pair_labels(exp_labels, cal_labels):
        for field in LABEL_NUMERIC_FIELDS:
            exp_val = exp.get(field)
            if isinstance(exp_val, (int, float)):
                position_check(exp_val, pred.get(field))

        pred_lines = {norm_text(ln.get("text", "")): ln for ln in pred.get("lines") or []}
        for exp_line in exp.get("lines") or []:
            text_total += 1
            pred_line = pred_lines.get(norm_text(exp_line.get("text", "")))
            if pred_line is None:
                continue
            text_hit += 1
            for field in LINE_NUMERIC_FIELDS:
                exp_val = exp_line.get(field)
                if isinstance(exp_val, (int, float)):
                    position_check(exp_val, pred_line.get(field))

    for exp, pred in pair_labels(exp_labels, raw_labels):
        for field in LABEL_NUMERIC_FIELDS:
            if exp.get(field) is None:
                null_check(pred.get(field))
        pred_lines = {norm_text(ln.get("text", "")): ln for ln in pred.get("lines") or []}
        for exp_line in exp.get("lines") or []:
            pred_line = pred_lines.get(norm_text(exp_line.get("text", "")))
            if pred_line is None:
                continue
            for field in LINE_NUMERIC_FIELDS:
                if exp_line.get(field) is None:
                    null_check(pred_line.get(field))

    def pct(hit: int, total: int) -> str:
        return f"{100 * hit / total:.0f}% ({hit}/{total})" if total else "n/a"

    return {
        "labels": f"{len(cal_labels)}/{len(exp_labels)}"
        + ("" if len(cal_labels) == len(exp_labels) else " MISMATCH"),
        "text": pct(text_hit, text_total),
        "position": pct(pos_hit, pos_total),
        "null": pct(null_hit, null_total),
    }


def main() -> None:
    use_cached = "--cached" in sys.argv
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    images = sorted(
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, dict]] = []
    skipped: list[str] = []

    for image in images:
        expected_path = EXPECTED_DIR / f"{image.stem}.json"
        if not expected_path.exists():
            skipped.append(image.name)
            continue

        prediction_path = PREDICTIONS_DIR / f"{image.stem}.json"
        raw_path = PREDICTIONS_DIR / f"{image.stem}.raw.json"

        if use_cached and prediction_path.exists():
            measured = json.loads(prediction_path.read_text(encoding="utf-8"))
            raw = (
                json.loads(raw_path.read_text(encoding="utf-8"))
                if raw_path.exists()
                else measured
            )
        else:
            print(f"Running extract on {image.name}...")
            cache_dir = PREDICTIONS_DIR / "llm" / image.stem
            set_run_cache_dir(cache_dir)
            result = run_dual(image)
            measured = copy.deepcopy(result["spec"])
            raw_labels = []
            for lab in measured.get("labels") or []:
                raw_lab = copy.deepcopy(lab)
                for ln in raw_lab.get("lines") or []:
                    ln.pop("measured_fields", None)
                raw_labels.append(raw_lab)
            raw = {
                "unit": measured.get("unit"),
                "image_px": measured.get("image_px"),
                "dimension_annotations": measured.get("dimension_annotations"),
                "labels": raw_labels,
            }
            prediction_path.write_text(
                json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raw_path.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        score = score_image(expected, raw, measured)
        score["method"] = "dual"
        rows.append((image.name, score))

    if skipped:
        print(
            "Skipped (no expected/<name>.json yet): " + ", ".join(skipped),
            file=sys.stderr,
        )

    if not rows:
        print(
            "No image has a ground truth yet. Create benchmark/expected/<name>.json",
            file=sys.stderr,
        )
        sys.exit(1)

    name_width = max(len(name) for name, _ in rows)
    header = (
        f"{'image'.ljust(name_width)}  {'method':<8} {'labels':<14} {'text':<14} "
        f"{'position':<14} {'null':<14}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, score in rows:
        print(
            f"{name.ljust(name_width)}  {score['method']:<8} {score['labels']:<14} "
            f"{score['text']:<14} {score['position']:<14} {score['null']:<14}"
        )


if __name__ == "__main__":
    main()
