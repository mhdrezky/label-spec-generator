"""Side-by-side eval: designer pipeline vs dual_call per layout cohort.

    python eval/run_compare.py [--cached]

Uses eval/cohorts.json for cohort labels. Scores with score_image from run_eval.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_call.extract import run_dual  # noqa: E402
from eval.run_eval import (  # noqa: E402
    EVAL_DIR,
    EXPECTED_DIR,
    IMAGES_DIR,
    score_image,
)
from llm_cache import set_run_cache_dir  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

COHORTS_PATH = EVAL_DIR / "cohorts.json"
PREDICTIONS_DIR = EVAL_DIR / "predictions"
DUAL_PREDICTIONS_DIR = EVAL_DIR / "predictions-dual"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DUAL_DIR = Path(__file__).resolve().parent.parent / "results-dual"


def load_cohorts() -> dict[str, str]:
    if not COHORTS_PATH.is_file():
        return {}
    return json.loads(COHORTS_PATH.read_text(encoding="utf-8"))


def latest_specs_in(root: Path, image_stem: str) -> Path | None:
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/specs.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        src = Path(data.get("source_image") or "").name
        if src == f"{image_stem}.png" or src.startswith(image_stem):
            return path
        if image_stem in str(data.get("source_image") or ""):
            return path
    return None




def baseline_from_results(image_stem: str) -> dict | None:
    path = latest_specs_in(RESULTS_DIR, image_stem)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    labels = data.get("labels") or []
    return {
        "labels": str(len(labels)),
        "warnings": len(data.get("warnings") or []),
        "path": str(path.parent.name),
    }


def run_or_load_pipeline(image: Path, use_cached: bool) -> tuple[dict, dict]:
    prediction_path = PREDICTIONS_DIR / f"{image.stem}.json"
    raw_path = PREDICTIONS_DIR / f"{image.stem}.raw.json"

    if use_cached and prediction_path.exists():
        measured = json.loads(prediction_path.read_text(encoding="utf-8"))
        raw = (
            json.loads(raw_path.read_text(encoding="utf-8"))
            if raw_path.exists()
            else measured
        )
        return raw, measured

    cache_dir = PREDICTIONS_DIR / "llm" / image.stem
    set_run_cache_dir(cache_dir)
    ctx = run_pipeline(image)
    measured = ctx.to_spec_dict()
    raw = {
        "unit": ctx.unit,
        "image_px": ctx.image_px,
        "dimension_annotations": ctx.dimension_annotations,
        "labels": ctx.pre_measure_labels or ctx.labels,
    }
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    prediction_path.write_text(
        json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return raw, measured


def run_or_load_dual(image: Path, use_cached: bool) -> tuple[dict, dict]:
    prediction_path = DUAL_PREDICTIONS_DIR / f"{image.stem}.json"
    raw_path = DUAL_PREDICTIONS_DIR / f"{image.stem}.raw.json"

    if use_cached and prediction_path.exists():
        measured = json.loads(prediction_path.read_text(encoding="utf-8"))
        raw = (
            json.loads(raw_path.read_text(encoding="utf-8"))
            if raw_path.exists()
            else measured
        )
        return raw, measured

    cache_dir = DUAL_PREDICTIONS_DIR / "llm" / image.stem
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

    DUAL_PREDICTIONS_DIR.mkdir(exist_ok=True)
    prediction_path.write_text(
        json.dumps(measured, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return raw, measured


def main() -> None:
    use_cached = "--cached" in sys.argv
    cohorts = load_cohorts()
    if not cohorts:
        print(f"No cohorts in {COHORTS_PATH}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, dict, dict, dict | None, bool]] = []
    skipped: list[str] = []

    for image_name, cohort in sorted(cohorts.items(), key=lambda x: (x[1], x[0])):
        image = IMAGES_DIR / image_name
        if not image.is_file():
            skipped.append(f"{image_name} (missing file)")
            continue
        expected_path = EXPECTED_DIR / f"{image.stem}.json"
        pipe_raw, pipe_meas = run_or_load_pipeline(image, use_cached)
        dual_raw, dual_meas = run_or_load_dual(image, use_cached)
        baseline = baseline_from_results(image.stem)

        if expected_path.exists():
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            pipe_score = score_image(expected, pipe_raw, pipe_meas)
            dual_score = score_image(expected, dual_raw, dual_meas)
            rows.append((cohort, image_name, pipe_score, dual_score, baseline, True))
        else:
            pipe_score = {
                "labels": f"{len(pipe_meas.get('labels') or [])}",
                "text": "n/a",
                "position": "n/a",
                "null": "n/a",
            }
            dual_score = {
                "labels": f"{len(dual_meas.get('labels') or [])}",
                "text": "n/a",
                "position": "n/a",
                "null": "n/a",
            }
            rows.append((cohort, image_name, pipe_score, dual_score, baseline, False))
            skipped.append(f"{image_name} (no GT - qualitative only)")

    if skipped:
        print("Skipped: " + ", ".join(skipped), file=sys.stderr)

    if not rows:
        print("No cohort images scored.", file=sys.stderr)
        sys.exit(1)

    name_width = max(len(name) for _, name, _, _, _, _ in rows)
    cohort_width = max(len(c) for c, _, _, _, _, _ in rows)
    header = (
        f"{'cohort'.ljust(cohort_width)}  {'image'.ljust(name_width)}  "
        f"{'pipe labels':<14} {'dual labels':<14} {'pipe text':<14} {'dual text':<14} "
        f"{'pipe pos':<14} {'dual pos':<14}"
    )
    print()
    print(header)
    print("-" * len(header))
    for cohort, name, pipe, dual, baseline, _has_gt in rows:
        print(
            f"{cohort.ljust(cohort_width)}  {name.ljust(name_width)}  "
            f"{pipe['labels']:<14} {dual['labels']:<14} "
            f"{pipe['text']:<14} {dual['text']:<14} "
            f"{pipe['position']:<14} {dual['position']:<14}"
        )
        if baseline:
            print(
                f"{'':>{cohort_width + name_width + 4}}  baseline results/{baseline['path']}: "
                f"{baseline['labels']} labels, {baseline['warnings']} warnings"
            )

    by_cohort: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for cohort, _, pipe, dual, _, _ in rows:
        by_cohort[cohort].append((pipe, dual))

    print()
    print("Per-cohort summary (images in cohort):")
    for cohort in sorted(by_cohort):
        pairs = by_cohort[cohort]
        print(f"  {cohort}: {len(pairs)} image(s)")


if __name__ == "__main__":
    main()
