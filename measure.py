"""Deterministic px→mm measurement — no heuristic repair."""

from __future__ import annotations

POSITION_STEP_MM = 0.5
MIN_TEXT_SIZE_MM = 1.0
EDGE_MATCH_TOLERANCE = 0.15
EDGE_CLAMP_EPS_MM = 0.5
COORD_SLACK = 0.05
SCALE_MISMATCH_TOLERANCE = 0.20
MAX_TEXT_HEIGHT_FRACTION = 0.4
SINGLE_LINE_MAX_TEXT_FRACTION = 0.85
SLOT_SIZE_FRACTION = 0.65
SHORT_TEXT_SIZE_FRACTION = 0.55
SHORT_TEXT_MAX_CHARS = 3
MIN_PLATE_DIM_MM = 3.0


def _round_step(value: float) -> float:
    return round(value / POSITION_STEP_MM) * POSITION_STEP_MM


def _clamp_center(value: float, dim: float) -> float:
    if dim <= 2 * EDGE_CLAMP_EPS_MM:
        return _round_step(dim / 2)
    return _round_step(min(max(value, EDGE_CLAMP_EPS_MM), dim - EDGE_CLAMP_EPS_MM))


def _valid_bbox(bbox) -> bool:
    return (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _label_name(label: dict) -> str:
    num = label.get("label_number", "?")
    lines = label.get("lines") or []
    first = f" ({lines[0]['text'][:20]})" if lines and lines[0].get("text") else ""
    return f"label #{num}{first}"


def _sort_lines_by_y(label: dict) -> None:
    lines = label.get("lines") or []
    annotated = [ln for ln in lines if isinstance(ln.get("y_mm"), (int, float))]
    if len(annotated) == len(lines) and len(lines) > 1:
        lines.sort(key=lambda ln: ln["y_mm"])


def normalize(spec: dict) -> None:
    labels = spec.get("labels") or []
    for i, label in enumerate(labels, start=1):
        label["label_number"] = i
        label["lines"] = label.get("lines") or []
        _sort_lines_by_y(label)
        label.setdefault("holes", [])


def drop_impossible_values(spec: dict, warnings: list[str]) -> None:
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
                        f"{name}: '{line.get('text', '')}' {field}={value} "
                        f"outside plate ({dim}mm) — dropped"
                    )
                    line[field] = None


def _match_dimension_to_edge(
    label: dict, field: str, edge: tuple[float, float], spec: dict, warnings: list[str]
) -> bool:
    """Fill null plate dimension when a dimension line brackets this edge."""
    wanted_axis = "horizontal" if field == "width_mm" else "vertical"
    e0, e1 = edge
    edge_len = e1 - e0
    if edge_len <= 0:
        return False
    tol = max(8.0, EDGE_MATCH_TOLERANCE * edge_len)

    best = None
    for dim in spec.get("dimension_annotations") or []:
        if dim.get("axis") != wanted_axis:
            continue
        value = dim.get("value_mm")
        span = dim.get("span_px")
        if not (_is_num(value) and value >= MIN_PLATE_DIM_MM):
            continue
        if not (isinstance(span, (list, tuple)) and len(span) == 2
                and all(_is_num(v) for v in span)):
            continue
        s0, s1 = sorted(span)
        err = abs(s0 - e0) + abs(s1 - e1)
        if abs(s0 - e0) <= tol and abs(s1 - e1) <= tol and (best is None or err < best[0]):
            best = (err, value)

    if best is None:
        return False

    value = best[1]
    if label.get(field) is not None:
        return False

    label[field] = float(value)
    measured = label.setdefault("measured_fields", [])
    if field not in measured:
        measured.append(field)
    warnings.append(
        f"{_label_name(label)}: {field} filled {value:g}mm from dimension line"
    )
    return True


def _estimate_size_from_bbox(
    text_h_mm: float, slot_h: float | None, line_count: int, text: str
) -> float:
    if line_count > 1 and slot_h and slot_h > 0:
        cap = slot_h * SLOT_SIZE_FRACTION
        if len((text or "").strip()) <= SHORT_TEXT_MAX_CHARS:
            cap = min(cap, slot_h * SHORT_TEXT_SIZE_FRACTION)
        return min(text_h_mm, cap)
    return text_h_mm


def _measure_lines(label: dict, warnings: list[str]) -> None:
    bbox = label.get("bbox_px")
    if not _valid_bbox(bbox):
        return
    width, height = label.get("width_mm"), label.get("height_mm")
    sx = (bbox[2] - bbox[0]) / width if _is_num(width) and width > 0 else None
    sy = (bbox[3] - bbox[1]) / height if _is_num(height) and height > 0 else None
    name = _label_name(label)
    lines_list = label.get("lines") or []
    line_count = len(lines_list)
    slot_h = (height / line_count) if _is_num(height) and line_count > 0 else None
    max_text_frac = (
        SINGLE_LINE_MAX_TEXT_FRACTION if line_count == 1 else MAX_TEXT_HEIGHT_FRACTION
    )

    for line in lines_list:
        lb = line.get("bbox_px")
        if not _valid_bbox(lb):
            continue
        measured = line.setdefault("measured_fields", [])

        if sx and not _is_num(line.get("x_mm")):
            cx = ((lb[0] + lb[2]) / 2 - bbox[0]) / sx
            if -COORD_SLACK * width <= cx <= (1 + COORD_SLACK) * width:
                line["x_mm"] = _clamp_center(cx, width)
                measured.append("x_mm")

        if sy:
            if not _is_num(line.get("y_mm")):
                cy = ((lb[1] + lb[3]) / 2 - bbox[1]) / sy
                if -COORD_SLACK * height <= cy <= (1 + COORD_SLACK) * height:
                    line["y_mm"] = _clamp_center(cy, height)
                    measured.append("y_mm")

            if not _is_num(line.get("size_mm")):
                text_h = (lb[3] - lb[1]) / sy
                text_h = _estimate_size_from_bbox(
                    text_h, slot_h, line_count, line.get("text", "")
                )
                if 0 < text_h <= max_text_frac * height:
                    line["size_mm"] = max(MIN_TEXT_SIZE_MM, _round_step(text_h))
                    measured.append("size_mm")


def physical_checks(spec: dict, warnings: list[str]) -> None:
    for label in spec.get("labels") or []:
        name = _label_name(label)
        width = label.get("width_mm")
        height = label.get("height_mm")

        if not _is_num(width) or not _is_num(height):
            warnings.append(f"{name}: plate dimensions missing")

        if label.get("quantity") is None:
            label["quantity"] = 1

        for line in label.get("lines") or []:
            text = line.get("text", "")
            if not text.strip():
                warnings.append(f"{name}: empty text line")

            size = line.get("size_mm")
            y = line.get("y_mm")
            x = line.get("x_mm")

            if _is_num(height):
                if _is_num(size) and size >= height:
                    warnings.append(
                        f"{name}: '{text}' size_mm={size} >= plate height {height}"
                    )
                if _is_num(y):
                    half = (size / 2) if _is_num(size) else 0
                    if y > height or y < 0:
                        warnings.append(
                            f"{name}: '{text}' y_mm={y} outside plate height {height}"
                        )
                    elif y + half > height + 0.5 or y - half < -0.5:
                        warnings.append(
                            f"{name}: '{text}' (center {y}mm, size {size}mm) "
                            f"extends past plate edge (height {height})"
                        )
            if _is_num(width) and _is_num(x) and (x > width or x < 0):
                warnings.append(
                    f"{name}: '{text}' x_mm={x} outside plate width {width}"
                )


def run_measure(spec: dict, warnings: list[str]) -> None:
    """Fill null plate dims from dimension edges; measure line bbox gaps."""
    labels = spec.get("labels") or []
    for label in labels:
        bbox = label.get("bbox_px")
        if not _valid_bbox(bbox):
            if not _is_num(label.get("width_mm")) or not _is_num(label.get("height_mm")):
                warnings.append(
                    f"{_label_name(label)}: no plate bbox — dimensions cannot be measured"
                )
            continue
        if not _is_num(label.get("width_mm")):
            _match_dimension_to_edge(
                label, "width_mm", (bbox[0], bbox[2]), spec, warnings
            )
        if not _is_num(label.get("height_mm")):
            _match_dimension_to_edge(
                label, "height_mm", (bbox[1], bbox[3]), spec, warnings
            )
        _measure_lines(label, warnings)

    normalize(spec)
    drop_impossible_values(spec, warnings)
    physical_checks(spec, warnings)
