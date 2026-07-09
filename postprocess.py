"""Deterministic post-processing of the raw extraction.

Everything that used to be crammed into the LLM prompt as layout rules
lives here as plain Python:

1. normalize      — sequential label numbers, lines sorted by y_mm.
2. correct_sequence_outliers — if annotated y positions form an arithmetic
   sequence (e.g. 24, 52, 80, ...), misread values are snapped back and
   reported.
3. calibrate (calibrate.py) — pixel bboxes + annotated dims → px/mm scale;
   fills unannotated positions/sizes with MEASURED values (recorded in
   ``measured_fields``) and repairs misread plate dimensions.
4. resolve_layout — the label editor needs concrete numbers on every line;
   anything still missing after measurement is filled with deterministic
   defaults, recorded in ``computed_fields``.
5. physical_checks — sanity warnings (text outside plate, size vs height...).

Value provenance, most to least trusted: annotated (plain value) >
measured (bbox-derived) > computed (layout default).

When ``stage_dir`` is set, a deep-copied specs snapshot is written after each
major stage under that directory (for eval / debugging).

Returns the spec dict with a top-level ``warnings`` list.
"""

import copy
import json
import os

from calibrate import calibrate, flag_suspect_plates

SEQUENCE_MIN_POINTS = 4
SEQUENCE_TOLERANCE_MM = 2.0
POSITION_STEP_MM = 0.5
MIN_TEXT_SIZE_MM = 2.0

# Ordered stage names written under results/<ts>/stages/ when stage_dir is set.
STAGE_NAMES = (
    "01_extract",
    "02_normalize",
    "03_calibrate",
    "04_flagged",
    "05_refine",
    "06_resolve",
)


def _round_step(value: float) -> float:
    return round(value / POSITION_STEP_MM) * POSITION_STEP_MM


def _label_name(label: dict) -> str:
    num = label.get("label_number", "?")
    first = ""
    lines = label.get("lines") or []
    if lines and lines[0].get("text"):
        first = f" ({lines[0]['text'][:20]})"
    return f"label #{num}{first}"


def _sort_lines_by_y(label: dict) -> None:
    lines = label.get("lines") or []
    annotated = [ln for ln in lines if isinstance(ln.get("y_mm"), (int, float))]
    if len(annotated) == len(lines) and len(lines) > 1:
        # stable sort: same-row texts keep their left-to-right order
        lines.sort(key=lambda ln: ln["y_mm"])


def _snapshot_stage(stage_dir: str | None, name: str, spec: dict, warnings: list[str]) -> None:
    """Write a deep-copied specs snapshot for one pipeline stage."""
    if not stage_dir:
        return
    os.makedirs(stage_dir, exist_ok=True)
    payload = copy.deepcopy(spec)
    payload["warnings"] = list(warnings)
    payload["pipeline_stage"] = name
    path = os.path.join(stage_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize(spec: dict) -> None:
    labels = spec.get("labels") or []
    for i, label in enumerate(labels, start=1):
        label["label_number"] = i
        label["lines"] = label.get("lines") or []
        _sort_lines_by_y(label)
        label.setdefault("holes", [])


def drop_impossible_values(spec: dict, warnings: list[str]) -> None:
    """The model sometimes copies a nearby annotation into the wrong field
    (e.g. x_mm=24 on a 20mm-wide plate). A coordinate outside the plate is
    physically impossible — drop it to null so calibration/resolver refill it."""
    for label in spec.get("labels") or []:
        name = _label_name(label)
        width = label.get("width_mm")
        height = label.get("height_mm")
        for line in label.get("lines") or []:
            for field, dim in (("x_mm", width), ("y_mm", height)):
                value = line.get(field)
                if (
                    isinstance(value, (int, float))
                    and isinstance(dim, (int, float))
                    and not 0 <= value <= dim
                ):
                    warnings.append(
                        f"{name}: '{line.get('text', '')}' {field}={value} is "
                        f"outside the plate ({dim}mm) — dropped, will be refilled"
                    )
                    line[field] = None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def correct_sequence_outliers(spec: dict, warnings: list[str]) -> None:
    for label in spec.get("labels") or []:
        lines = label.get("lines") or []
        ys = [ln.get("y_mm") for ln in lines]
        if len(lines) < SEQUENCE_MIN_POINTS or not all(
            isinstance(y, (int, float)) for y in ys
        ):
            continue

        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        step = _median(diffs)
        if step <= 0:
            continue

        intercept = _median([y - i * step for i, y in enumerate(ys)])
        expected = [intercept + i * step for i in range(len(ys))]
        outliers = [
            i for i, y in enumerate(ys)
            if abs(y - expected[i]) > SEQUENCE_TOLERANCE_MM
        ]

        # Only trust the sequence when it explains the clear majority.
        if not outliers or len(outliers) > len(ys) // 3:
            continue

        for i in outliers:
            fixed = _round_step(expected[i])
            warnings.append(
                f"{_label_name(label)}: line '{lines[i].get('text', '')}' "
                f"y_mm={ys[i]} breaks the {step:g}mm sequence — corrected to {fixed:g}"
            )
            lines[i]["y_mm"] = fixed
            computed = lines[i].setdefault("computed_fields", [])
            if "y_mm" not in computed:
                computed.append("y_mm")


def _snap_equal_row_positions(label: dict) -> None:
    """Numbered-column plates: many short tokens in equal rows → slot centers."""
    lines = label.get("lines") or []
    n = len(lines)
    height = label.get("height_mm")
    if not (isinstance(height, (int, float)) and n >= 6):
        return
    if not all(len((ln.get("text") or "").strip()) <= 4 for ln in lines):
        return
    for i, line in enumerate(lines):
        line["y_mm"] = _round_step(float(height) * (2 * i + 1) / (2 * n))
        computed = line.setdefault("computed_fields", [])
        if "y_mm" not in computed:
            computed.append("y_mm")


def _default_size(height: float, line_count: int) -> float:
    if line_count <= 1:
        size = height * 0.55
    else:
        size = (height / line_count) * 0.55
    return max(MIN_TEXT_SIZE_MM, _round_step(size))


def resolve_layout(spec: dict, warnings: list[str]) -> None:
    for label in spec.get("labels") or []:
        width = label.get("width_mm")
        height = label.get("height_mm")
        lines = label.get("lines") or []
        n = len(lines)

        if label.get("quantity") is None:
            label["quantity"] = 1
            warnings.append(
                f"{_label_name(label)}: quantity not stated — defaulted to 1"
            )

        for i, line in enumerate(lines):
            computed = line.setdefault("computed_fields", [])

            if line.get("size_mm") is None:
                if isinstance(height, (int, float)):
                    line["size_mm"] = _default_size(float(height), n)
                    computed.append("size_mm")
                else:
                    warnings.append(
                        f"{_label_name(label)}: line '{line.get('text', '')}' "
                        "has no size_mm and plate height is unknown"
                    )

            if line.get("y_mm") is None:
                if isinstance(height, (int, float)):
                    # y is the CENTER of the text (editor anchor convention)
                    slot_center = float(height) * (2 * i + 1) / (2 * n)
                    line["y_mm"] = _round_step(slot_center)
                    computed.append("y_mm")
                else:
                    warnings.append(
                        f"{_label_name(label)}: line '{line.get('text', '')}' "
                        "has no y_mm and plate height is unknown"
                    )

            if line.get("x_mm") is None:
                if isinstance(width, (int, float)):
                    line["x_mm"] = _round_step(float(width) / 2)
                    computed.append("x_mm")
                    if line.get("alignment") is None:
                        line["alignment"] = "center"
                        computed.append("alignment")
                else:
                    warnings.append(
                        f"{_label_name(label)}: line '{line.get('text', '')}' "
                        "has no x_mm and plate width is unknown"
                    )


def physical_checks(spec: dict, warnings: list[str]) -> None:
    for label in spec.get("labels") or []:
        name = _label_name(label)
        width = label.get("width_mm")
        height = label.get("height_mm")

        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            warnings.append(f"{name}: plate dimensions not annotated in the draft")

        for line in label.get("lines") or []:
            text = line.get("text", "")
            if not text.strip():
                warnings.append(f"{name}: empty text line")

            size = line.get("size_mm")
            y = line.get("y_mm")
            x = line.get("x_mm")

            # x_mm / y_mm are the CENTER of the text (editor anchor)
            if isinstance(height, (int, float)):
                if isinstance(size, (int, float)) and size >= height:
                    warnings.append(
                        f"{name}: '{text}' size_mm={size} >= plate height {height}"
                    )
                if isinstance(y, (int, float)):
                    half = (size / 2) if isinstance(size, (int, float)) else 0
                    if y > height or y < 0:
                        warnings.append(
                            f"{name}: '{text}' y_mm={y} outside plate height {height}"
                        )
                    elif y + half > height + 0.5 or y - half < -0.5:
                        warnings.append(
                            f"{name}: '{text}' (center {y}mm, size {size}mm) "
                            f"extends past the plate edge (height {height})"
                        )
            if (
                isinstance(width, (int, float))
                and isinstance(x, (int, float))
                and (x > width or x < 0)
            ):
                warnings.append(
                    f"{name}: '{text}' x_mm={x} outside plate width {width}"
                )


def postprocess(spec: dict, refiner=None, stage_dir: str | None = None) -> dict:
    """``refiner(spec, warnings)`` — optional pass-2 hook (see layered.refine_with_layers),
    invoked only when the sanity check flags physically impossible geometry.

    ``stage_dir`` — if set, write a specs snapshot after each major stage.
    """
    warnings: list[str] = []
    _snapshot_stage(stage_dir, "01_extract", spec, warnings)

    normalize(spec)
    drop_impossible_values(spec, warnings)
    correct_sequence_outliers(spec, warnings)
    _snapshot_stage(stage_dir, "02_normalize", spec, warnings)

    calibrate(spec, warnings)

    for label in spec.get("labels") or []:
        _snap_equal_row_positions(label)
    _snapshot_stage(stage_dir, "03_calibrate", spec, warnings)

    flagged = flag_suspect_plates(spec, warnings)
    _snapshot_stage(stage_dir, "04_flagged", spec, warnings)

    if refiner is not None and flagged:
        refiner(spec, warnings)
        # refined values replace the suspect ones; re-apply the guards
        drop_impossible_values(spec, warnings)
    _snapshot_stage(stage_dir, "05_refine", spec, warnings)

    for label in spec.get("labels") or []:
        _sort_lines_by_y(label)
    resolve_layout(spec, warnings)
    physical_checks(spec, warnings)
    spec["total_labels"] = len(spec.get("labels") or [])
    spec["warnings"] = warnings
    _snapshot_stage(stage_dir, "06_resolve", spec, warnings)
    return spec
