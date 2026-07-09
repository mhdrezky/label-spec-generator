"""Anti-overfitting eval harness.

    python eval/run_eval.py [--cached]

For every image in eval/images/ with a hand-verified ground truth in
eval/expected/<name>.json (same schema as the RAW extraction — annotated
values only, null where the draft has no annotation), this runs the vision
extraction and scores it. Predictions are saved to eval/predictions/.

Rule of the game: every prompt/schema change must be scored against ALL
images here, never against a single one — that is how the previous pipeline
ended up overfitted.

Scores per image:
- labels:    predicted label count == expected label count
- text:      % of expected text lines found (normalized exact match)
- position:  % of expected non-null mm values within +/-2mm (incl. w/h),
             scored AFTER bbox calibration (calibrate.py) — measures the
             full geometry pipeline
- null:      % of expected-null numeric fields also null in the RAW model
             output, BEFORE calibration fills them (catches invented numbers)

--cached reuses eval/predictions/*.json (raw model output) instead of
calling the API.
"""

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrate import calibrate  # noqa: E402
from extract import extract_specs  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
EXPECTED_DIR = EVAL_DIR / "expected"
PREDICTIONS_DIR = EVAL_DIR / "predictions"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
POSITION_TOLERANCE_MM = 2.0
LINE_NUMERIC_FIELDS = ("x_mm", "y_mm", "size_mm")
LABEL_NUMERIC_FIELDS = ("width_mm", "height_mm")


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").upper()).strip()


def label_texts(label: dict) -> set[str]:
    return {norm_text(ln.get("text", "")) for ln in label.get("lines") or []}


def pair_labels(expected: list[dict], predicted: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy pairing by shared line texts."""
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


def score_image(expected: dict, raw: dict, calibrated: dict) -> dict:
    """Position accuracy uses the calibrated spec; null precision uses the
    raw model output (before calibration fills unannotated fields)."""
    exp_labels = expected.get("labels") or []
    cal_labels = calibrated.get("labels") or []
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
        if use_cached and prediction_path.exists():
            predicted = json.loads(prediction_path.read_text(encoding="utf-8"))
        else:
            print(f"Extracting {image.name}...")
            predicted, schema_errors = extract_specs(image)
            for error in schema_errors:
                print(f"  schema warning: {error}", file=sys.stderr)
            prediction_path.write_text(
                json.dumps(predicted, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        calibrated = copy.deepcopy(predicted)
        calibrate(calibrated, [])
        rows.append((image.name, score_image(expected, predicted, calibrated)))

    if skipped:
        print(
            "Skipped (no expected/<name>.json yet): " + ", ".join(skipped),
            file=sys.stderr,
        )

    if not rows:
        print(
            "No image has a ground truth yet. Create eval/expected/<name>.json "
            "(run the pipeline once, then hand-correct the raw extraction).",
            file=sys.stderr,
        )
        sys.exit(1)

    name_width = max(len(name) for name, _ in rows)
    header = f"{'image'.ljust(name_width)}  {'labels':<14} {'text':<14} {'position':<14} {'null':<14}"
    print()
    print(header)
    print("-" * len(header))
    for name, score in rows:
        print(
            f"{name.ljust(name_width)}  {score['labels']:<14} {score['text']:<14} "
            f"{score['position']:<14} {score['null']:<14}"
        )


if __name__ == "__main__":
    main()
