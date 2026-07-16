"""Per-plate resolution: content-complete vs derived (bbox/dimension calc)."""

from __future__ import annotations

COORD_SLACK = 0.05
POSITION_STEP_MM = 0.5
ROW_CLUSTER_FRAC = 0.18
X_OVERLAP_MERGE = 0.35
SINGLE_ROW_TOP_FRAC = 0.42
SINGLE_ROW_MAX_BBOX_HEIGHT_FRAC = 0.45
CONTENT_Y_HINT_MIN_FRAC = 0.6
CONTENT_Y_ROW_TOL_MM = 2.0
CHAR_WIDTH_FRAC = 0.55 / 0.66  # match editor: cap height -> em box -> glyph width
SINGLE_LINE_WIDTH_MARGIN_MM = 2.0
WIDE_PLATE_ASPECT = 4.0
HEADER_ROW_MAX_FRAC = 0.38
COLUMN_PAIR_X_FRACS = (0.33, 0.67)
X_SAME_COLUMN_TOL_MM = 2.0
COLUMN_SPAN_MIN_FRAC = 0.72
COLUMN_EDGE_MARGIN_FRAC = 0.04
PAIR_BAND_COL_FRAC = 0.58
PAIR_BAND_MAX_BLOCK_FRAC = 0.90

RESOLUTION_CONTENT = "content"
RESOLUTION_DERIVED = "derived"


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_bbox_px(bbox) -> bool:
    return (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _round_step(value: float) -> float:
    return round(value / POSITION_STEP_MM) * POSITION_STEP_MM


def _clamp_center(value: float, dim: float) -> float:
    eps = 0.5
    if dim <= 2 * eps:
        return _round_step(dim / 2)
    return _round_step(min(max(value, eps), dim - eps))


def _x_overlap_ratio(a: list, b: list) -> float:
    if not (_valid_bbox_px(a) and _valid_bbox_px(b)):
        return 0.0
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    inter = max(0.0, right - left)
    if inter <= 0:
        return 0.0
    narrow = min(a[2] - a[0], b[2] - b[0])
    return inter / narrow if narrow > 0 else 0.0


def _position_from_bbox(
    line_bbox: list,
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
) -> tuple[float | None, float | None]:
    if not (_valid_bbox_px(line_bbox) and _valid_bbox_px(plate_bbox)):
        return None, None
    if not (_is_num(width_mm) and width_mm > 0 and _is_num(height_mm) and height_mm > 0):
        return None, None
    sx = (plate_bbox[2] - plate_bbox[0]) / width_mm
    sy = (plate_bbox[3] - plate_bbox[1]) / height_mm
    if sx <= 0 or sy <= 0:
        return None, None
    cx = ((line_bbox[0] + line_bbox[2]) / 2 - plate_bbox[0]) / sx
    cy = ((line_bbox[1] + line_bbox[3]) / 2 - plate_bbox[1]) / sy
    x_mm = None
    y_mm = None
    if -COORD_SLACK * width_mm <= cx <= (1 + COORD_SLACK) * width_mm:
        x_mm = _clamp_center(cx, width_mm)
    if -COORD_SLACK * height_mm <= cy <= (1 + COORD_SLACK) * height_mm:
        y_mm = _clamp_center(cy, height_mm)
    return x_mm, y_mm


def _llm_position_in_bounds(
    x_mm,
    y_mm,
    width_mm,
    height_mm,
) -> tuple[float | None, float | None]:
    out_x = None
    out_y = None
    if _is_num(x_mm) and _is_num(width_mm) and width_mm > 0:
        if -COORD_SLACK * width_mm <= float(x_mm) <= (1 + COORD_SLACK) * width_mm:
            out_x = _clamp_center(float(x_mm), width_mm)
    if _is_num(y_mm) and _is_num(height_mm) and height_mm > 0:
        if -COORD_SLACK * height_mm <= float(y_mm) <= (1 + COORD_SLACK) * height_mm:
            out_y = _clamp_center(float(y_mm), height_mm)
    return out_x, out_y


def _capture_content_hints(lines: list[dict]) -> None:
    for ln in lines:
        if _is_num(ln.get("x_mm")):
            ln["_content_x"] = float(ln["x_mm"])
        if _is_num(ln.get("y_mm")):
            ln["_content_y"] = float(ln["y_mm"])
        if _is_num(ln.get("size_mm")):
            ln["_content_size"] = float(ln["size_mm"])


def _scale_hints(
    lines: list[dict],
    content_w,
    content_h,
    final_w,
    final_h,
) -> None:
    sx = float(final_w) / float(content_w) if _is_num(content_w) and content_w > 0 else 1.0
    sy = float(final_h) / float(content_h) if _is_num(content_h) and content_h > 0 else 1.0
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return
    for ln in lines:
        if _is_num(ln.get("_content_x")):
            ln["_content_x"] = float(ln["_content_x"]) * sx
        if _is_num(ln.get("_content_y")):
            ln["_content_y"] = float(ln["_content_y"]) * sy


def _dims_scale_consistent(bbox_px: list, width_mm, height_mm, tolerance: float = 0.25) -> bool:
    if not (_valid_bbox_px(bbox_px) and _is_num(width_mm) and _is_num(height_mm)):
        return False
    if width_mm <= 0 or height_mm <= 0:
        return False
    w_px = bbox_px[2] - bbox_px[0]
    h_px = bbox_px[3] - bbox_px[1]
    if w_px <= 0 or h_px <= 0:
        return False
    sx = w_px / float(width_mm)
    sy = h_px / float(height_mm)
    if sx <= 0 or sy <= 0:
        return False
    return abs(sx - sy) / max(sx, sy) <= tolerance


def _lines_have_complete_content(
    lines: list[dict],
    width_mm,
    height_mm,
) -> bool:
    if not lines or not (_is_num(width_mm) and _is_num(height_mm)):
        return False
    for ln in lines:
        text = (ln.get("text") or "").strip()
        if not text:
            return False
        if not all(_is_num(ln.get(k)) for k in ("x_mm", "y_mm", "size_mm")):
            return False
        if not (0 <= float(ln["x_mm"]) <= float(width_mm)):
            return False
        if not (0 <= float(ln["y_mm"]) <= float(height_mm)):
            return False
    return True


def _dims_corrected(content_w, content_h, final_w, final_h, tolerance: float = 0.08) -> bool:
    if not all(_is_num(v) for v in (content_w, content_h, final_w, final_h)):
        return False
    if content_w <= 0 or content_h <= 0 or final_w <= 0 or final_h <= 0:
        return False
    return (
        abs(float(content_w) - float(final_w)) / float(final_w) > tolerance
        or abs(float(content_h) - float(final_h)) / float(final_h) > tolerance
    )


def _plate_aspect_ratio(width_mm, height_mm) -> float:
    if not (_is_num(width_mm) and _is_num(height_mm) and height_mm > 0):
        return 0.0
    return float(width_mm) / float(height_mm)


def _is_wide_strip_plate(width_mm, height_mm) -> bool:
    return _plate_aspect_ratio(width_mm, height_mm) >= WIDE_PLATE_ASPECT


def resolve_plate_mode(
    plate_lines: list[dict],
    bbox_px: list,
    final_w,
    final_h,
    content_w,
    content_h,
    *,
    spec_table: bool,
    table_cell: bool,
) -> str:
    """Choose content-complete vs derived resolution for one plate."""
    if not plate_lines:
        return RESOLUTION_DERIVED

    _capture_content_hints(plate_lines)
    cw = content_w if _is_num(content_w) else final_w
    ch = content_h if _is_num(content_h) else final_h

    if spec_table and table_cell and _lines_have_complete_content(plate_lines, cw, ch):
        return RESOLUTION_CONTENT

    if _dims_corrected(cw, ch, final_w, final_h):
        return RESOLUTION_DERIVED

    wide = _is_wide_strip_plate(final_w, final_h)

    if len(plate_lines) > 1:
        if not wide and _lines_have_complete_content(plate_lines, cw, ch):
            return RESOLUTION_CONTENT
        return RESOLUTION_DERIVED

    if _lines_have_complete_content(plate_lines, cw, ch) and _dims_scale_consistent(
        bbox_px, cw, ch
    ):
        return RESOLUTION_CONTENT

    if _lines_have_complete_content(plate_lines, final_w, final_h):
        return RESOLUTION_CONTENT

    return RESOLUTION_DERIVED


def _cluster_lines_by_content_y(lines: list[dict], height_mm: float) -> list[list[dict]]:
    tol = max(CONTENT_Y_ROW_TOL_MM, float(height_mm) * 0.12)
    ordered = sorted(
        [(float(ln["_content_y"]), ln) for ln in lines if _is_num(ln.get("_content_y"))],
        key=lambda item: item[0],
    )
    if not ordered:
        return [[ln] for ln in lines]

    clusters: list[list[dict]] = [[ordered[0][1]]]
    centers = [ordered[0][0]]
    for y_val, ln in ordered[1:]:
        if y_val - centers[-1] <= tol:
            clusters[-1].append(ln)
            centers[-1] = sum(float(x["_content_y"]) for x in clusters[-1]) / len(clusters[-1])
        else:
            clusters.append([ln])
            centers.append(y_val)
    return clusters


def _line_cy_px(line: dict) -> float | None:
    bbox = line.get("bbox_px")
    if not _valid_bbox_px(bbox):
        return None
    return (bbox[1] + bbox[3]) / 2.0


def _cluster_lines_by_bbox(lines: list[dict], plate_bbox: list) -> list[list[dict]]:
    ph = plate_bbox[3] - plate_bbox[1]
    if ph <= 0:
        return [[ln] for ln in lines]
    tol = max(4.0, ROW_CLUSTER_FRAC * ph)

    ordered: list[tuple[float, dict]] = []
    for ln in lines:
        cy = _line_cy_px(ln)
        if cy is None:
            continue
        ordered.append((cy, ln))
    if not ordered:
        return [[ln] for ln in lines]
    ordered.sort(key=lambda item: item[0])

    clusters: list[list[dict]] = [[ordered[0][1]]]
    centers: list[float] = [ordered[0][0]]
    for cy, ln in ordered[1:]:
        if cy - centers[-1] <= tol:
            clusters[-1].append(ln)
            centers[-1] = sum(_line_cy_px(x) or 0 for x in clusters[-1]) / len(clusters[-1])
        else:
            clusters.append([ln])
            centers.append(cy)
    return clusters


def _cluster_lines_by_row(
    lines: list[dict],
    plate_bbox: list,
    height_mm: float,
) -> list[list[dict]]:
    hinted = [ln for ln in lines if _is_num(ln.get("_content_y"))]
    if len(hinted) >= max(1, len(lines)) * CONTENT_Y_HINT_MIN_FRAC:
        return _cluster_lines_by_content_y(lines, height_mm)
    return _cluster_lines_by_bbox(lines, plate_bbox)


def _cluster_x_span_frac(cluster: list[dict], plate_bbox: list) -> float:
    bboxes = [ln.get("bbox_px") for ln in cluster if _valid_bbox_px(ln.get("bbox_px"))]
    if not bboxes or not _valid_bbox_px(plate_bbox):
        return 0.0
    left = min(b[0] for b in bboxes)
    right = max(b[2] for b in bboxes)
    pw = plate_bbox[2] - plate_bbox[0]
    return (right - left) / pw if pw > 0 else 0.0


def _merge_column_stack_cluster(
    cluster: list[dict],
    prev: list[dict],
    plate_bbox: list | None = None,
    *,
    merging_index: int = 1,
) -> bool:
    """True when every line in cluster stacks under one line in prev (same x column)."""
    if not cluster or not prev:
        return False
    wide_row = (
        plate_bbox is not None
        and _cluster_x_span_frac(cluster, plate_bbox) >= 0.55
        and _cluster_x_span_frac(prev, plate_bbox) >= 0.55
    )
    if wide_row and merging_index == 1:
        return False
    for ln in cluster:
        bbox = ln.get("bbox_px") or []
        matches = [
            p
            for p in prev
            if _x_overlap_ratio(bbox, p.get("bbox_px") or []) >= X_OVERLAP_MERGE
        ]
        if len(matches) != 1:
            return False
    return True


def _merge_row_clusters(
    clusters: list[list[dict]],
    plate_bbox: list | None = None,
) -> list[list[dict]]:
    if len(clusters) <= 1:
        return clusters
    merged = [clusters[0]]
    for i, cluster in enumerate(clusters[1:], start=1):
        prev = merged[-1]
        if _merge_column_stack_cluster(
            cluster, prev, plate_bbox, merging_index=i
        ):
            prev.extend(cluster)
            continue
        if len(cluster) == 1 and len(prev) >= 2:
            orphan = cluster[0]
            ob = orphan.get("bbox_px") or []
            if any(
                _x_overlap_ratio(ob, ln.get("bbox_px") or []) >= X_OVERLAP_MERGE
                for ln in prev
            ):
                prev.extend(cluster)
                continue
        merged.append(cluster)
    return merged


def _dominant_content_y(content_ys: list[float], height_mm: float) -> float | None:
    if not content_ys:
        return None
    tol = max(CONTENT_Y_ROW_TOL_MM, float(height_mm) * 0.12)
    ordered = sorted(content_ys)
    buckets: list[list[float]] = [[ordered[0]]]
    for y_val in ordered[1:]:
        if y_val - buckets[-1][-1] <= tol:
            buckets[-1].append(y_val)
        else:
            buckets.append([y_val])
    buckets.sort(key=lambda bucket: (-len(bucket), sum(bucket) / len(bucket)))
    chosen = buckets[0]
    if len(buckets) > 1 and len(buckets[0]) == len(buckets[1]):
        chosen = min(buckets[:2], key=lambda bucket: sum(bucket) / len(bucket))
    return _round_step(sum(chosen) / len(chosen))


def _snap_cluster_y_mm(
    cluster: list[dict],
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
) -> float | None:
    content_ys = [
        float(ln["_content_y"])
        for ln in cluster
        if _is_num(ln.get("_content_y"))
    ]
    tol = max(CONTENT_Y_ROW_TOL_MM, float(height_mm) * 0.12)
    if content_ys:
        if max(content_ys) - min(content_ys) <= tol:
            return _dominant_content_y(content_ys, height_mm)
        dominant = _dominant_content_y(content_ys, height_mm)
        if dominant is not None:
            return dominant

    ys: list[float] = []
    for ln in cluster:
        bbox = ln.get("bbox_px")
        if not _valid_bbox_px(bbox):
            if _is_num(ln.get("y_mm")):
                ys.append(float(ln["y_mm"]))
            continue
        _, y_mm = _position_from_bbox(bbox, plate_bbox, width_mm, height_mm)
        if _is_num(y_mm):
            ys.append(float(y_mm))
    if not ys:
        return None
    ys.sort()
    return _round_step(ys[len(ys) // 2])


def _line_x_mm(
    ln: dict,
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
) -> float | None:
    bbox = ln.get("bbox_px")
    if _valid_bbox_px(bbox):
        x_mm, _ = _position_from_bbox(bbox, plate_bbox, width_mm, height_mm)
        if _is_num(x_mm):
            return x_mm
    if _is_num(ln.get("x_mm")):
        return _clamp_center(float(ln["x_mm"]), width_mm)
    return None


def _assign_cluster_y_values(
    cluster: list[dict],
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
) -> list[float]:
    """One shared y per visual row (never split column stacks onto extra rows)."""
    y_mm = _snap_cluster_y_mm(cluster, plate_bbox, width_mm, height_mm)
    if not _is_num(y_mm):
        return []
    for ln in cluster:
        ln["y_mm"] = y_mm
    return [float(y_mm)]


def _body_lines_for_header(
    header: dict,
    body: list[dict],
    assigned: set[int],
) -> list[dict]:
    hb = header.get("bbox_px") or []
    col_lines = [
        ln
        for ln in body
        if id(ln) not in assigned
        and _x_overlap_ratio(ln.get("bbox_px") or [], hb) >= X_OVERLAP_MERGE * 0.75
    ]
    for ln in col_lines:
        assigned.add(id(ln))
    return col_lines


def _equal_column_slot_xs(n: int, width_mm: float) -> list[float]:
    """Even column centers across plate width (layout container, not LLM coords)."""
    if n <= 0:
        return []
    margin = float(width_mm) * COLUMN_EDGE_MARGIN_FRAC
    usable = float(width_mm) - 2 * margin
    return [_round_step(margin + usable * (i + 0.5) / n) for i in range(n)]


def _block_bounds_from_headers(
    header_xs: list[float],
    width_mm: float,
) -> list[tuple[float, float]]:
    """Column block edges at header midpoints; last block runs to plate right margin."""
    n = len(header_xs)
    margin = float(width_mm) * COLUMN_EDGE_MARGIN_FRAC
    bounds: list[tuple[float, float]] = []
    for i in range(n):
        left = margin if i == 0 else (header_xs[i - 1] + header_xs[i]) / 2.0
        right = (
            float(width_mm) - margin
            if i == n - 1
            else (header_xs[i] + header_xs[i + 1]) / 2.0
        )
        bounds.append((left, right))
    return bounds


def _pair_band_bounds(
    header_x: float,
    block_left: float,
    block_right: float,
    width_mm: float,
    n_columns: int,
) -> tuple[float, float]:
    """Tight pair band centered on header — avoids stretching LOW/HIGH to plate edge."""
    block_w = block_right - block_left
    if block_w <= 0 or n_columns <= 0:
        return block_left, block_right
    avg_col = float(width_mm) / n_columns
    pair_w = min(block_w * PAIR_BAND_MAX_BLOCK_FRAC, avg_col * PAIR_BAND_COL_FRAC)
    pair_w = max(pair_w, float(width_mm) * 0.08)
    half = pair_w / 2.0
    left = max(block_left, float(header_x) - half)
    right = min(block_right, float(header_x) + half)
    if right - left < pair_w * 0.75:
        left = max(0.5, float(header_x) - half)
        right = min(float(width_mm) - 0.5, float(header_x) + half)
    return left, right


def _place_lines_in_column_band(
    col_lines: list[dict],
    left: float,
    right: float,
    width_mm: float,
    *,
    center_x: float | None = None,
) -> None:
    col_w = right - left
    if col_w <= 0:
        return
    if len(col_lines) == 1:
        x_mm = center_x if _is_num(center_x) else (left + right) / 2.0
        col_lines[0]["x_mm"] = _clamp_center(float(x_mm), width_mm)
        return
    col_lines.sort(
        key=lambda ln: (
            float(ln["_content_y"]) if _is_num(ln.get("_content_y")) else 0.0,
            _line_cy_px(ln) or 0.0,
            ln.get("text") or "",
        )
    )
    fracs = COLUMN_PAIR_X_FRACS
    if len(col_lines) > 2:
        step = 1.0 / (len(col_lines) + 1)
        fracs = tuple(step * (i + 1) for i in range(len(col_lines)))
    for ln, frac in zip(col_lines, fracs):
        ln["x_mm"] = _clamp_center(left + col_w * float(frac), width_mm)


def _layout_column_blocks(
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
    clusters: list[list[dict]],
    warnings: list[str],
    label_number,
) -> None:
    """Layout row-2 text inside column blocks; header x/y stay unchanged."""
    if len(clusters) < 2 or float(width_mm) / float(height_mm) < WIDE_PLATE_ASPECT:
        return

    headers = sorted(
        clusters[0],
        key=lambda ln: _line_x_mm(ln, plate_bbox, width_mm, height_mm) or ln.get("x_mm") or 0,
    )
    n = len(headers)
    if n < 2:
        return

    header_xs: list[float] = []
    for header in headers:
        x_mm = header.get("x_mm")
        if not _is_num(x_mm):
            x_mm = _line_x_mm(header, plate_bbox, width_mm, height_mm)
        if not _is_num(x_mm):
            return
        header_xs.append(float(x_mm))

    body = [ln for cluster in clusters[1:] for ln in cluster]
    assigned: set[int] = set()
    groups: list[tuple[dict, list[dict]]] = []
    for header in headers:
        groups.append((header, _body_lines_for_header(header, body, assigned)))

    for orphan in body:
        if id(orphan) in assigned:
            continue
        ox = _line_x_mm(orphan, plate_bbox, width_mm, height_mm) or orphan.get("x_mm") or 0
        nearest = min(
            range(n),
            key=lambda i: abs(float(ox) - header_xs[i]),
        )
        groups[nearest][1].append(orphan)
        assigned.add(id(orphan))

    bounds = _block_bounds_from_headers(header_xs, width_mm)
    tol = max(X_SAME_COLUMN_TOL_MM, float(width_mm) * 0.015)
    needs_expand = max(header_xs) < float(width_mm) * COLUMN_SPAN_MIN_FRAC
    clustered_max = max(header_xs)

    if needs_expand:
        slot_xs = _equal_column_slot_xs(n, width_mm)
        for header, x_mm in zip(headers, slot_xs):
            header["x_mm"] = x_mm
        header_xs = slot_xs
        bounds = _block_bounds_from_headers(header_xs, width_mm)

    for i, ((header, col_body), (left, right)) in enumerate(zip(groups, bounds)):
        if not col_body:
            continue
        if len(col_body) >= 2:
            xs = [
                _line_x_mm(ln, plate_bbox, width_mm, height_mm) or ln.get("x_mm")
                for ln in col_body
            ]
            if all(_is_num(x) for x in xs) and max(xs) - min(xs) > tol:
                for ln, x_mm in zip(col_body, xs):
                    if _is_num(x_mm):
                        ln["x_mm"] = float(x_mm)
                continue
        header_x = float(header.get("x_mm") or header_xs[i])
        pair_left, pair_right = _pair_band_bounds(
            header_x, left, right, width_mm, n
        )
        _place_lines_in_column_band(
            col_body,
            pair_left,
            pair_right,
            width_mm,
            center_x=header.get("x_mm"),
        )

    if needs_expand:
        num = label_number if label_number is not None else "?"
        warnings.append(
            f"plate #{num}: equal {n} column slot(s) across {width_mm:g}mm "
            f"(was clustered to {clustered_max:g}mm max before layout)"
        )


def _layout_single_row_columns(
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
    cluster: list[dict],
    warnings: list[str],
    label_number,
) -> None:
    """Spread one-row multi-column labels when LLM x positions cluster left."""
    if float(width_mm) / float(height_mm) < WIDE_PLATE_ASPECT:
        return
    n = len(cluster)
    if n < 3:
        return

    ordered = sorted(
        cluster,
        key=lambda ln: _line_x_mm(ln, plate_bbox, width_mm, height_mm) or ln.get("x_mm") or 0,
    )
    xs: list[float] = []
    for ln in ordered:
        x_mm = ln.get("x_mm")
        if not _is_num(x_mm):
            x_mm = _line_x_mm(ln, plate_bbox, width_mm, height_mm)
        if not _is_num(x_mm):
            return
        xs.append(float(x_mm))

    if max(xs) >= float(width_mm) * COLUMN_SPAN_MIN_FRAC:
        return

    slot_xs = _equal_column_slot_xs(n, float(width_mm))
    for ln, x_mm in zip(ordered, slot_xs):
        ln["x_mm"] = x_mm

    num = label_number if label_number is not None else "?"
    warnings.append(
        f"plate #{num}: equal {n} column slot(s) across {width_mm:g}mm "
        f"(rightmost was {max(xs):g}mm before layout)"
    )


def _estimate_text_width_mm(text: str, size_mm: float) -> float:
    return max(1, len(text.strip())) * float(size_mm) * CHAR_WIDTH_FRAC


def fit_single_line_plate(label: dict, warnings: list[str]) -> None:
    """Shrink oversized single-line text so it fits the plate width."""
    lines = label.get("lines") or []
    if len(lines) != 1:
        return
    width_mm = label.get("width_mm")
    height_mm = label.get("height_mm")
    if not (_is_num(width_mm) and width_mm > 0):
        return

    ln = lines[0]
    text = (ln.get("text") or "").strip()
    size_mm = ln.get("size_mm")
    if not text or not _is_num(size_mm) or size_mm <= 0:
        return

    avail = float(width_mm) - 2 * SINGLE_LINE_WIDTH_MARGIN_MM
    est = _estimate_text_width_mm(text, float(size_mm))
    if est > avail:
        fitted = _round_step(float(size_mm) * (avail / est))
        if fitted > 0 and fitted < float(size_mm):
            num = label.get("label_number", "?")
            warnings.append(
                f"plate #{num}: text {text!r} too wide at {size_mm:g}mm "
                f"— reduced to {fitted:g}mm"
            )
            ln["size_mm"] = fitted

    if _is_num(height_mm) and height_mm > 0 and _is_num(ln.get("size_mm")):
        if float(height_mm) >= float(ln["size_mm"]) * 2.2:
            ln["y_mm"] = _round_step(float(height_mm) / 2.0)


def refine_line_y_positions(label: dict, warnings: list[str]) -> None:
    """Cluster rows; merge column stacks; center single-row plates when needed."""
    lines = label.get("lines") or []
    plate_bbox = label.get("bbox_px") or []
    width_mm = label.get("width_mm")
    height_mm = label.get("height_mm")
    if (
        len(lines) <= 1
        or not _valid_bbox_px(plate_bbox)
        or not (_is_num(width_mm) and width_mm > 0)
        or not (_is_num(height_mm) and height_mm > 0)
    ):
        return

    clusters = _cluster_lines_by_row(lines, plate_bbox, float(height_mm))
    wide = _is_wide_strip_plate(width_mm, height_mm)
    before = len(clusters)
    if wide:
        clusters = _merge_row_clusters(clusters, plate_bbox)
    if wide and len(clusters) < before:
        num = label.get("label_number", "?")
        warnings.append(
            f"plate #{num}: merged {before - len(clusters)} extra row(s) "
            f"into adjacent row (column stack)"
        )

    row_ys: list[float] = []
    for cluster in clusters:
        row_ys.extend(
            _assign_cluster_y_values(
                cluster, plate_bbox, float(width_mm), float(height_mm)
            )
        )

    if wide:
        if len(clusters) >= 2:
            _layout_column_blocks(
                plate_bbox,
                float(width_mm),
                float(height_mm),
                clusters,
                warnings,
                label.get("label_number"),
            )
        elif len(clusters) == 1:
            _layout_single_row_columns(
                plate_bbox,
                float(width_mm),
                float(height_mm),
                clusters[0],
                warnings,
                label.get("label_number"),
            )

    if not wide or len(clusters) != 1 or not row_ys:
        return

    ph = plate_bbox[3] - plate_bbox[1]
    compact = all(
        _valid_bbox_px(ln.get("bbox_px"))
        and (ln["bbox_px"][3] - ln["bbox_px"][1]) / ph <= SINGLE_ROW_MAX_BBOX_HEIGHT_FRAC
        for ln in lines
    )
    if not compact:
        return

    max_y = max(row_ys)
    sizes = [ln.get("size_mm") for ln in lines if _is_num(ln.get("size_mm"))]
    text_h = max(sizes) if sizes else None
    if not (_is_num(text_h) and text_h > 0):
        return
    if max_y > height_mm * SINGLE_ROW_TOP_FRAC:
        return
    if height_mm < text_h * 2.2:
        return

    centered = _round_step(float(height_mm) / 2.0)
    num = label.get("label_number", "?")
    if abs(max_y - centered) > POSITION_STEP_MM:
        warnings.append(
            f"plate #{num}: single text row hugging top (y={max_y:g}mm) "
            f"— centered at {centered:g}mm"
        )
        for ln in lines:
            ln["y_mm"] = centered


def build_content_complete_lines(
    plate_lines: list[dict],
    final_w: float,
    final_h: float,
    content_w,
    content_h,
) -> list[dict]:
    """Use LLM mm values, rescale when plate dims were corrected."""
    _capture_content_hints(plate_lines)
    _scale_hints(plate_lines, content_w, content_h, final_w, final_h)
    lines_out: list[dict] = []
    for ln in plate_lines:
        x_mm = ln.get("_content_x") if _is_num(ln.get("_content_x")) else ln.get("x_mm")
        y_mm = ln.get("_content_y") if _is_num(ln.get("_content_y")) else ln.get("y_mm")
        size_mm = ln.get("_content_size") if _is_num(ln.get("_content_size")) else ln.get("size_mm")
        if _is_num(x_mm):
            x_mm = _clamp_center(float(x_mm), final_w)
        if _is_num(y_mm):
            y_mm = _clamp_center(float(y_mm), final_h)
        lines_out.append({
            "text": ln.get("text", ""),
            "x_mm": x_mm,
            "y_mm": y_mm,
            "size_mm": float(size_mm) if _is_num(size_mm) else None,
            "alignment": None,
            "bold": None,
            "bbox_px": ln.get("bbox_px"),
        })
    return lines_out


def build_derived_lines(
    plate_lines: list[dict],
    plate: dict,
    structure: dict,
    stated_by_line: dict[int, float],
    *,
    table_cell: bool,
    warnings: list[str],
    pid: int,
) -> list[dict]:
    """Position from bbox; optional table-cell LLM fallback for x/y."""
    from dual_call.postprocess import _resolve_size_mm

    _capture_content_hints(plate_lines)
    cw = plate.get("_content_w")
    ch = plate.get("_content_h")
    fw = plate.get("width_mm")
    fh = plate.get("height_mm")
    if _is_num(cw) and _is_num(ch) and _is_num(fw) and _is_num(fh):
        _scale_hints(plate_lines, cw, ch, fw, fh)

    lines_out: list[dict] = []
    plate_bbox = plate.get("bbox_px") or []
    for i, ln in enumerate(plate_lines, start=1):
        text = ln.get("text", "")
        line_bbox = ln.get("bbox_px")
        size_mm = (
            float(ln["_content_size"])
            if table_cell and _is_num(ln.get("_content_size"))
            else _resolve_size_mm(
                i, ln, structure, stated_by_line, line_bbox, plate_bbox, fh
            )
        )

        bbox_x, bbox_y = _position_from_bbox(line_bbox or [], plate_bbox, fw, fh)
        llm_x, llm_y = _llm_position_in_bounds(ln.get("x_mm"), ln.get("y_mm"), fw, fh)
        if table_cell:
            x_mm = llm_x if llm_x is not None else bbox_x
            y_mm = llm_y if llm_y is not None else bbox_y
        else:
            x_mm = bbox_x if bbox_x is not None else llm_x
            y_mm = bbox_y if bbox_y is not None else llm_y

        if x_mm is None and _is_num(ln.get("x_mm")):
            warnings.append(
                f"plate #{pid} '{text}': x_mm={ln.get('x_mm')} outside plate - null"
            )
        if y_mm is None and _is_num(ln.get("y_mm")):
            warnings.append(
                f"plate #{pid} '{text}': y_mm={ln.get('y_mm')} outside plate - null"
            )

        lines_out.append({
            "text": text,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "size_mm": size_mm,
            "alignment": None,
            "bold": None,
            "bbox_px": line_bbox,
            "_content_y": ln.get("_content_y"),
        })
    return lines_out
