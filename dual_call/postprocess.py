"""Merge structure + content into label spec; frac to px; tiered resolvers."""

from __future__ import annotations

from measure import run_measure

COORD_SLACK = 0.05
POSITION_STEP_MM = 0.5
ROW_CLUSTER_FRAC = 0.18
X_OVERLAP_MERGE = 0.35
SINGLE_ROW_TOP_FRAC = 0.42
SINGLE_ROW_MAX_BBOX_HEIGHT_FRAC = 0.45
MERGED_LINE_MIN_HEIGHT_FRAC = 0.45
MERGED_HEADER_ROW_FRAC = 0.42
WIDE_PLATE_ASPECT = 4.0


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_frac_bbox(frac) -> bool:
    if not (isinstance(frac, (list, tuple)) and len(frac) == 4):
        return False
    if not all(isinstance(v, (int, float)) for v in frac):
        return False
    if not all(0.0 <= v <= 1.0 for v in frac):
        return False
    x1, y1, x2, y2 = frac
    return x2 > x1 and y2 > y1


def _valid_bbox_px(bbox) -> bool:
    return (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def frac_to_bbox_px(frac: list, image_px: dict) -> list[int] | None:
    if not _valid_frac_bbox(frac):
        return None
    w = image_px.get("width") or 0
    h = image_px.get("height") or 0
    if w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = frac
    return [round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)]


def frac_span_to_px(span_frac: list, image_px: dict, axis: str) -> list[float] | None:
    if not (
        isinstance(span_frac, (list, tuple))
        and len(span_frac) == 2
        and all(isinstance(v, (int, float)) for v in span_frac)
        and 0.0 <= span_frac[0] <= 1.0
        and 0.0 <= span_frac[1] <= 1.0
    ):
        return None
    w = image_px.get("width") or 0
    h = image_px.get("height") or 0
    if axis == "horizontal":
        if w <= 0:
            return None
        return [span_frac[0] * w, span_frac[1] * w]
    if axis == "vertical":
        if h <= 0:
            return None
        return [span_frac[0] * h, span_frac[1] * h]
    return None


def _round_step(value: float) -> float:
    return round(value / POSITION_STEP_MM) * POSITION_STEP_MM


def _clamp_center(value: float, dim: float) -> float:
    eps = 0.5
    if dim <= 2 * eps:
        return _round_step(dim / 2)
    return _round_step(min(max(value, eps), dim - eps))


def sort_lines_by_y(label: dict) -> None:
    lines = label.get("lines") or []
    if len(lines) <= 1:
        return

    def sort_key(ln: dict) -> tuple:
        y_mm = ln.get("y_mm")
        if _is_num(y_mm):
            return (0, y_mm)
        bbox = ln.get("bbox_px")
        if _valid_bbox_px(bbox):
            return (1, bbox[1])
        return (2, 0)

    lines.sort(key=sort_key)


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


def _line_cy_px(line: dict, plate_bbox: list) -> float | None:
    bbox = line.get("bbox_px")
    if not _valid_bbox_px(bbox):
        return None
    return (bbox[1] + bbox[3]) / 2.0


def _cluster_lines_by_row(
    lines: list[dict], plate_bbox: list
) -> list[list[dict]]:
    ph = plate_bbox[3] - plate_bbox[1]
    if ph <= 0:
        return [[ln] for ln in lines]
    tol = max(4.0, ROW_CLUSTER_FRAC * ph)

    ordered: list[tuple[float, dict]] = []
    for ln in lines:
        cy = _line_cy_px(ln, plate_bbox)
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
            centers[-1] = sum(_line_cy_px(x, plate_bbox) or 0 for x in clusters[-1]) / len(
                clusters[-1]
            )
        else:
            clusters.append([ln])
            centers.append(cy)
    return clusters


def _merge_orphan_row_clusters(clusters: list[list[dict]]) -> list[list[dict]]:
    if len(clusters) <= 1:
        return clusters
    merged = [clusters[0]]
    for cluster in clusters[1:]:
        if len(cluster) == 1 and len(merged[-1]) >= 2:
            orphan = cluster[0]
            ob = orphan.get("bbox_px") or []
            if any(_x_overlap_ratio(ob, ln.get("bbox_px") or []) >= X_OVERLAP_MERGE for ln in merged[-1]):
                merged[-1].extend(cluster)
                continue
        merged.append(cluster)
    return merged


def _snap_cluster_y_mm(
    cluster: list[dict],
    plate_bbox: list,
    width_mm: float,
    height_mm: float,
) -> float | None:
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


def _lines_have_compact_bbox_height(lines: list[dict], plate_bbox: list) -> bool:
    ph = plate_bbox[3] - plate_bbox[1]
    if ph <= 0:
        return False
    for ln in lines:
        bbox = ln.get("bbox_px")
        if not _valid_bbox_px(bbox):
            return False
        if (bbox[3] - bbox[1]) / ph > SINGLE_ROW_MAX_BBOX_HEIGHT_FRAC:
            return False
    return True


def _plate_aspect_ratio(
    plate_bbox: list,
    width_mm=None,
    height_mm=None,
) -> float:
    if _is_num(width_mm) and _is_num(height_mm) and height_mm > 0:
        return float(width_mm) / float(height_mm)
    if _valid_bbox_px(plate_bbox):
        w = plate_bbox[2] - plate_bbox[0]
        h = plate_bbox[3] - plate_bbox[1]
        return w / h if h > 0 else 0.0
    return 0.0


def _looks_merged_multi_word_line(
    line: dict,
    plate_bbox: list,
    width_mm=None,
    height_mm=None,
    line_count=None,
) -> bool:
    """Wide strip plates only: split 3+ word column blobs, not title plates."""
    if line_count == 1:
        return False
    parts = (line.get("text") or "").strip().split()
    if len(parts) < 3:
        return False
    if _plate_aspect_ratio(plate_bbox, width_mm, height_mm) < WIDE_PLATE_ASPECT:
        return False
    bbox = line.get("bbox_px")
    if not (_valid_bbox_px(bbox) and _valid_bbox_px(plate_bbox)):
        return False
    ph = plate_bbox[3] - plate_bbox[1]
    if ph <= 0:
        return False
    bh = bbox[3] - bbox[1]
    return bh / ph >= MERGED_LINE_MIN_HEIGHT_FRAC


def _split_merged_line(line: dict) -> list[dict]:
    """Expand one multi-word line into separate entries with estimated sub-bboxes."""
    text = (line.get("text") or "").strip()
    parts = text.split()
    bbox = line.get("bbox_px")
    if len(parts) < 2 or not _valid_bbox_px(bbox):
        return [line]

    x0, y0, x1, y1 = (float(v) for v in bbox)
    bh = y1 - y0
    bw = x1 - x0
    base = {
        k: v
        for k, v in line.items()
        if k not in ("text", "bbox_px", "x_mm", "y_mm")
    }

    def entry(word: str, bb: list[float]) -> dict:
        return {
            **base,
            "text": word,
            "bbox_px": [round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3])],
            "x_mm": None,
            "y_mm": None,
        }

    if len(parts) == 3 and bh > 0:
        split_y = y0 + bh * MERGED_HEADER_ROW_FRAC
        row2_y0 = y0 + bh * 0.52
        half_w = bw / 2
        return [
            entry(parts[0], [x0, y0, x1, split_y]),
            entry(parts[1], [x0, row2_y0, x0 + half_w, y1]),
            entry(parts[2], [x0 + half_w, row2_y0, x1, y1]),
        ]

    slice_w = bw / len(parts)
    return [
        entry(
            word,
            [x0 + i * slice_w, y0, x0 + (i + 1) * slice_w, y1],
        )
        for i, word in enumerate(parts)
    ]


def _expand_merged_content_lines(
    lines: list[dict],
    plate_bbox: list,
    warnings: list[str],
    plate_id: int,
    width_mm=None,
    height_mm=None,
    line_count=None,
) -> list[dict]:
    """Split LLM lines that merged multiple words into one tall column blob."""
    expanded: list[dict] = []
    split_count = 0
    for ln in lines:
        if _looks_merged_multi_word_line(
            ln, plate_bbox, width_mm, height_mm, line_count
        ):
            pieces = _split_merged_line(ln)
            if len(pieces) > 1:
                split_count += 1
            expanded.extend(pieces)
        else:
            expanded.append(ln)
    if split_count:
        warnings.append(
            f"plate #{plate_id}: split {split_count} merged multi-word line(s) "
            f"into separate text entries"
        )
    return expanded


def _content_plate_meta(content: dict) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    for plate in content.get("plates") or []:
        pid = plate.get("id")
        if isinstance(pid, int):
            meta[pid] = {
                "line_count": plate.get("line_count"),
                "width_mm": plate.get("width_mm"),
                "height_mm": plate.get("height_mm"),
            }
    return meta


def _refine_line_y_positions(label: dict, warnings: list[str]) -> None:
    """Cluster rows from bbox, merge column orphans, center single-row plates."""
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

    clusters = _cluster_lines_by_row(lines, plate_bbox)
    before = len(clusters)
    clusters = _merge_orphan_row_clusters(clusters)
    if len(clusters) < before:
        num = label.get("label_number", "?")
        warnings.append(
            f"plate #{num}: merged {before - len(clusters)} orphan row(s) "
            f"into adjacent row (side-by-side text)"
        )

    row_ys: list[float] = []
    for cluster in clusters:
        y_mm = _snap_cluster_y_mm(cluster, plate_bbox, float(width_mm), float(height_mm))
        if not _is_num(y_mm):
            continue
        row_ys.append(y_mm)
        for ln in cluster:
            ln["y_mm"] = y_mm

    if len(clusters) != 1 or not row_ys:
        return
    if not _lines_have_compact_bbox_height(lines, plate_bbox):
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


def _stated_size_by_line(structure: dict) -> dict[int, float]:
    return {
        s["line"]: s["size_mm"]
        for s in structure.get("line_sizes") or []
        if isinstance(s.get("line"), int) and _is_num(s.get("size_mm"))
    }


def _size_from_bbox(
    line_bbox: list | None,
    plate_bbox: list | None,
    height_mm,
) -> float | None:
    if not (_valid_bbox_px(line_bbox) and _valid_bbox_px(plate_bbox)):
        return None
    if not (_is_num(height_mm) and height_mm > 0):
        return None
    ph = plate_bbox[3] - plate_bbox[1]
    if ph <= 0:
        return None
    sy = ph / height_mm
    text_h = (line_bbox[3] - line_bbox[1]) / sy
    return float(text_h) if text_h > 0 else None


def _resolve_size_mm(
    line_index: int,
    line: dict,
    structure: dict,
    stated_by_line: dict[int, float],
    line_bbox: list | None,
    plate_bbox: list | None,
    height_mm,
) -> float | None:
    if line_index in stated_by_line:
        return float(stated_by_line[line_index])
    default = structure.get("default_size_mm")
    if _is_num(default):
        return float(default)
    llm_size = line.get("size_mm")
    if _is_num(llm_size):
        return float(llm_size)
    bbox_size = _size_from_bbox(line_bbox, plate_bbox, height_mm)
    if _is_num(bbox_size):
        return float(bbox_size)
    return None



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
    x_mm, y_mm, width_mm, height_mm
) -> tuple[float | None, float | None]:
    if not (_is_num(width_mm) and _is_num(height_mm)):
        return None, None
    out_x = x_mm if _is_num(x_mm) and 0 <= x_mm <= width_mm else None
    out_y = y_mm if _is_num(y_mm) and 0 <= y_mm <= height_mm else None
    return out_x, out_y


def _line_center_in_plate(line_bbox, plate_bbox, slack_frac: float = 0.08) -> bool:
    """Keep only lines whose center falls inside this plate (layout-agnostic filter)."""
    if not (_valid_bbox_px(line_bbox) and _valid_bbox_px(plate_bbox)):
        return False
    cx = (line_bbox[0] + line_bbox[2]) / 2
    cy = (line_bbox[1] + line_bbox[3]) / 2
    pw = plate_bbox[2] - plate_bbox[0]
    ph = plate_bbox[3] - plate_bbox[1]
    pad_x = pw * slack_frac
    pad_y = ph * slack_frac
    return (
        plate_bbox[0] - pad_x <= cx <= plate_bbox[2] + pad_x
        and plate_bbox[1] - pad_y <= cy <= plate_bbox[3] + pad_y
    )


def _best_tiled_dim_sum_for_edge(
    edge: tuple[float, float],
    dimension_annotations: list[dict],
    axis: str,
    bbox_px: list | None = None,
    height_mm=None,
) -> float | None:
    """Find dimension segments that tile a plate edge; prefer scale-consistent chains."""
    e0, e1 = edge
    edge_len = e1 - e0
    if edge_len <= 0:
        return None
    tol = max(8.0, 0.04 * edge_len)

    segments: list[tuple[float, float, float]] = []
    for dim in dimension_annotations:
        if dim.get("axis") != axis:
            continue
        value = dim.get("value_mm")
        span = dim.get("span_px")
        if not (
            _is_num(value)
            and value > 0
            and isinstance(span, (list, tuple))
            and len(span) == 2
            and span[1] > span[0]
        ):
            continue
        s0, s1 = float(span[0]), float(span[1])
        if s1 < e0 - tol or s0 > e1 + tol:
            continue
        segments.append((s0, s1, float(value)))

    if not segments:
        return None
    segments.sort(key=lambda s: s[0])

    sy = None
    if _valid_bbox_px(bbox_px) and _is_num(height_mm) and height_mm > 0:
        sy = (bbox_px[3] - bbox_px[1]) / float(height_mm)

    best_sum: float | None = None
    best_score = float("-inf")

    def score_chain(chain: list[tuple[float, float, float]]) -> float:
        total_mm = sum(s[2] for s in chain)
        if total_mm <= 0:
            return float("-inf")
        small_pen = sum(1 for s in chain if s[2] < 20)
        sx = edge_len / total_mm
        if sy and sy > 0 and sx > 0:
            scale_pen = abs(sx - sy) / sy
        else:
            scale_pen = 0.25
        return -(scale_pen + 0.08 * small_pen + 0.02 * len(chain))

    def try_chain(chain: list[tuple[float, float, float]]) -> None:
        nonlocal best_sum, best_score
        if not chain:
            return
        cover_start = chain[0][0]
        cover_end = chain[-1][1]
        if abs(cover_start - e0) > tol or abs(cover_end - e1) > tol:
            return
        for i in range(len(chain) - 1):
            if chain[i + 1][0] - chain[i][1] > tol:
                return
        total = sum(s[2] for s in chain)
        sc = score_chain(chain)
        if sc > best_score:
            best_score = sc
            best_sum = total

    def dfs(cursor: float, chain: list[tuple[float, float, float]], start_idx: int) -> None:
        if abs(cursor - e1) <= tol:
            try_chain(chain)
            return
        if cursor > e1 + tol or len(chain) >= 10:
            return
        for i in range(start_idx, len(segments)):
            s0, s1, val = segments[i]
            if s1 <= cursor + tol:
                continue
            if s0 - cursor > tol:
                continue
            dfs(max(cursor, s1), chain + [(s0, s1, val)], i + 1)

    for i, seg in enumerate(segments):
        s0, s1, _ = seg
        if abs(s0 - e0) > tol:
            continue
        dfs(s1, [seg], i + 1)

    for seg in segments:
        try_chain([seg])

    return best_sum


def _resolve_plate_width_mm(
    bbox_px: list,
    dimension_annotations: list[dict],
    width_mm,
    height_mm,
    spec_stub: dict,
    warnings: list[str],
    label_num: int,
) -> float | None:
    """Pick plate width: tiled dimension sum beats single-segment / column hints."""
    if not _valid_bbox_px(bbox_px):
        return width_mm if _is_num(width_mm) else None

    edge = (bbox_px[0], bbox_px[2])
    tiled = _best_tiled_dim_sum_for_edge(
        edge, dimension_annotations, "horizontal", bbox_px, height_mm
    )

    tmp = {
        "label_number": label_num,
        "width_mm": width_mm if _is_num(width_mm) else None,
        "height_mm": height_mm if _is_num(height_mm) else None,
        "bbox_px": bbox_px,
        "lines": [],
    }
    if not _is_num(tmp.get("width_mm")):
        from measure import _match_dimension_to_edge

        _match_dimension_to_edge(tmp, "width_mm", edge, spec_stub, warnings)

    candidate = tmp.get("width_mm") if _is_num(tmp.get("width_mm")) else width_mm
    if _is_num(tiled):
        if not _is_num(candidate) or candidate < tiled * 0.85:
            if _is_num(candidate) and candidate < tiled * 0.85:
                warnings.append(
                    f"plate #{label_num}: width_mm={candidate:g} looks like a column "
                    f"segment — using tiled dimension sum {tiled:g}mm"
                )
            candidate = tiled

    inferred_w, _ = _infer_plate_mm_from_bbox(
        bbox_px,
        dimension_annotations,
        candidate if _is_num(candidate) else None,
        height_mm if _is_num(height_mm) else None,
    )
    out_w = inferred_w if _is_num(inferred_w) else candidate
    out_w, _ = _coerce_plate_dims(
        bbox_px, out_w, height_mm, warnings, label_num, tiled_width_mm=tiled
    )
    return out_w


def _coerce_plate_dims(
    bbox_px: list,
    width_mm,
    height_mm,
    warnings: list[str],
    label_num: int,
    tiled_width_mm: float | None = None,
) -> tuple[float | None, float | None]:
    """Reject column-width mistaken for plate width; recover from bbox scale."""
    if not (_valid_bbox_px(bbox_px) and _is_num(height_mm) and height_mm > 0):
        return width_mm, height_mm
    w_px = bbox_px[2] - bbox_px[0]
    h_px = bbox_px[3] - bbox_px[1]
    if h_px <= 0 or w_px <= 0:
        return width_mm, height_mm
    sy = h_px / float(height_mm)
    out_w = width_mm if _is_num(width_mm) else None
    if _is_num(out_w) and out_w > 0:
        sx = w_px / out_w
        if abs(sx - sy) / sy > 0.45:
            warnings.append(
                f"plate #{label_num}: width_mm={out_w} inconsistent with bbox scale "
                f"(sx={sx:.2f} sy={sy:.2f}) — re-inferring from outline"
            )
            out_w = None
    if not _is_num(out_w):
        if _is_num(tiled_width_mm) and tiled_width_mm > 0:
            out_w = float(tiled_width_mm)
        elif sy > 0:
            out_w = w_px / sy
    return out_w, height_mm


def _median_px_per_mm(dimension_annotations: list[dict], axis: str) -> float | None:
    scales: list[float] = []
    for dim in dimension_annotations:
        if dim.get("axis") != axis:
            continue
        span = dim.get("span_px")
        value = dim.get("value_mm")
        if not (
            isinstance(span, (list, tuple))
            and len(span) == 2
            and _is_num(value)
            and value > 0
            and span[1] > span[0]
        ):
            continue
        scales.append((span[1] - span[0]) / float(value))
    if not scales:
        return None
    scales.sort()
    return scales[len(scales) // 2]


def _infer_plate_mm_from_bbox(
    bbox_px: list,
    dimension_annotations: list[dict],
    width_mm,
    height_mm,
) -> tuple[float | None, float | None]:
    """Fill null plate mm from bbox + dimension-line px/mm scale (layout-agnostic)."""
    if not _valid_bbox_px(bbox_px):
        return width_mm, height_mm
    out_w = width_mm if _is_num(width_mm) else None
    out_h = height_mm if _is_num(height_mm) else None
    w_px = bbox_px[2] - bbox_px[0]
    h_px = bbox_px[3] - bbox_px[1]
    if out_h is None:
        ppm = _median_px_per_mm(dimension_annotations, "vertical")
        if ppm and ppm > 0:
            out_h = h_px / ppm
    if out_w is None:
        ppm = _median_px_per_mm(dimension_annotations, "horizontal")
        if ppm and ppm > 0:
            out_w = w_px / ppm
    return out_w, out_h


def _lines_by_content_plate(content: dict, image_px: dict) -> dict[int, list[dict]]:
    """Keep content lines grouped by plate id from the LLM content pass."""
    by_id: dict[int, list[dict]] = {}
    for content_plate in content.get("plates") or []:
        pid = content_plate.get("id")
        if not isinstance(pid, int):
            continue
        rows: list[dict] = []
        for ln in content_plate.get("lines") or []:
            bbox = frac_to_bbox_px(ln.get("bbox_frac"), image_px)
            rows.append({**ln, "bbox_px": bbox})
        by_id[pid] = rows
    return by_id


def _merge_content_lines(content: dict, image_px: dict) -> tuple[list[dict], dict[int, dict]]:
    """Collect lines and optional per-plate mm from the content pass."""
    seen: set[tuple] = set()
    lines: list[dict] = []
    plate_mm: dict[int, dict] = {}
    for content_plate in content.get("plates") or []:
        pid = content_plate.get("id")
        if isinstance(pid, int):
            plate_mm[pid] = {
                "width_mm": content_plate.get("width_mm"),
                "height_mm": content_plate.get("height_mm"),
            }
        for ln in content_plate.get("lines") or []:
            bbox = frac_to_bbox_px(ln.get("bbox_frac"), image_px)
            key = (ln.get("text", ""), tuple(bbox or ()))
            if key in seen:
                continue
            seen.add(key)
            lines.append({**ln, "bbox_px": bbox})
    return lines, plate_mm


def _assign_lines_to_plates(all_lines: list[dict], plates: list[dict]) -> dict[int, list[dict]]:
    """Assign each line to the smallest plate whose bbox contains its center."""
    by_id: dict[int, list[dict]] = {p["id"]: [] for p in plates}
    orphans = 0
    for ln in all_lines:
        bbox = ln.get("bbox_px")
        best_id = None
        best_area = None
        for plate in plates:
            pb = plate.get("bbox_px") or []
            if not _line_center_in_plate(bbox, pb):
                continue
            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            if best_area is None or area < best_area:
                best_area = area
                best_id = plate["id"]
        if best_id is not None:
            by_id[best_id].append(ln)
        else:
            orphans += 1
    if orphans:
        by_id["_orphans"] = [{"orphan_count": orphans}]
    return by_id


def _normalize_structure_plates(structure: dict, image_px: dict, warnings: list[str]) -> list[dict]:
    plates: list[dict] = []
    dropped = 0
    gate = structure.get("gate") or {}
    expected = gate.get("cv_count") if gate.get("trust_cv") else None
    for raw in structure.get("plates") or []:
        frac = raw.get("bbox_frac")
        bbox_px = raw.get("bbox_px")
        if not _valid_bbox_px(bbox_px):
            bbox_px = frac_to_bbox_px(frac, image_px) if frac is not None else None
        if bbox_px is None:
            dropped += 1
            continue
        if frac is None and _valid_bbox_px(bbox_px):
            w = image_px.get("width") or 1
            h = image_px.get("height") or 1
            frac = [bbox_px[0] / w, bbox_px[1] / h, bbox_px[2] / w, bbox_px[3] / h]
        plates.append({
            "id": raw.get("id") if isinstance(raw.get("id"), int) else len(plates) + 1,
            "bbox_frac": frac,
            "bbox_px": bbox_px,
            "width_mm": raw.get("width_mm"),
            "height_mm": raw.get("height_mm"),
        })
    if dropped:
        warnings.append(f"structure: dropped {dropped} plate(s) with invalid bbox")
    if isinstance(expected, int) and expected > 0 and len(plates) != expected:
        warnings.append(
            f"structure: plate count {len(plates)} != CV count {expected} — trimming to CV"
        )
        plates = plates[:expected]
        for i, plate in enumerate(plates, start=1):
            plate["id"] = i
    return plates


def merge_to_spec(
    structure: dict,
    content: dict,
    image_px: dict,
    warnings: list[str],
) -> dict:
    plates = _normalize_structure_plates(structure, image_px, warnings)
    content_by_plate = _lines_by_content_plate(content, image_px)
    content_meta = _content_plate_meta(content)
    all_content_lines, content_plate_mm = _merge_content_lines(content, image_px)
    lines_by_plate = _assign_lines_to_plates(all_content_lines, plates)
    orphan_info = lines_by_plate.pop("_orphans", None)
    if orphan_info:
        count = orphan_info[0].get("orphan_count", 0)
        if count:
            warnings.append(f"content: {count} line(s) could not be assigned to any plate bbox")
    stated_by_line = _stated_size_by_line(structure)

    dimension_annotations = []
    for dim in structure.get("dimension_annotations") or []:
        span_px = frac_span_to_px(
            dim.get("span_frac") or [],
            image_px,
            dim.get("axis") or "",
        )
        if span_px is None:
            warnings.append(
                f"structure: dropped dimension {dim.get('value_mm')}mm (bad span_frac)"
            )
            continue
        dimension_annotations.append({
            "value_mm": dim.get("value_mm"),
            "axis": dim.get("axis"),
            "span_px": span_px,
        })

    spec_stub = {"dimension_annotations": dimension_annotations}

    labels: list[dict] = []
    for plate in plates:
        pid = plate["id"]
        mm_hint = content_plate_mm.get(pid) or {}
        width_mm = plate.get("width_mm")
        height_mm = plate.get("height_mm")
        if not _is_num(height_mm) and _is_num(mm_hint.get("height_mm")):
            height_mm = mm_hint["height_mm"]

        bbox = plate.get("bbox_px") or []
        if _valid_bbox_px(bbox) and not _is_num(height_mm):
            from measure import _match_dimension_to_edge

            tmp_h = {"label_number": pid, "height_mm": None, "lines": []}
            _match_dimension_to_edge(
                tmp_h, "height_mm", (bbox[1], bbox[3]), spec_stub, warnings
            )
            if _is_num(tmp_h.get("height_mm")):
                height_mm = tmp_h["height_mm"]

        if not _is_num(width_mm) and _is_num(mm_hint.get("width_mm")):
            width_mm = mm_hint["width_mm"]

        width_mm = _resolve_plate_width_mm(
            bbox,
            dimension_annotations,
            width_mm,
            height_mm,
            spec_stub,
            warnings,
            pid,
        )
        if not _is_num(height_mm):
            _, height_mm = _infer_plate_mm_from_bbox(
                bbox, dimension_annotations, width_mm, None
            )
        plate = {**plate, "width_mm": width_mm, "height_mm": height_mm}

        plate_lines = content_by_plate.get(pid) or lines_by_plate.get(pid) or []
        meta = content_meta.get(pid) or {}
        line_count = meta.get("line_count")
        if isinstance(line_count, int) and line_count != len(plate_lines):
            warnings.append(
                f"plate #{pid}: content line_count={line_count} "
                f"but got {len(plate_lines)} line(s)"
            )
        plate_lines = _expand_merged_content_lines(
            plate_lines,
            plate.get("bbox_px") or [],
            warnings,
            pid,
            plate.get("width_mm"),
            plate.get("height_mm"),
            line_count if isinstance(line_count, int) else None,
        )
        lines_out: list[dict] = []
        plate_bbox = plate.get("bbox_px") or []
        for i, ln in enumerate(plate_lines, start=1):
            text = ln.get("text", "")
            line_bbox = ln.get("bbox_px")
            size_mm = _resolve_size_mm(i, ln, structure, stated_by_line, line_bbox, plate.get("bbox_px"), plate.get("height_mm"))

            bbox_x, bbox_y = _position_from_bbox(
                line_bbox or [],
                plate.get("bbox_px") or [],
                plate.get("width_mm"),
                plate.get("height_mm"),
            )
            llm_x, llm_y = _llm_position_in_bounds(
                ln.get("x_mm"),
                ln.get("y_mm"),
                plate.get("width_mm"),
                plate.get("height_mm"),
            )
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
            })

        label = {
            "label_number": pid,
            "width_mm": plate.get("width_mm"),
            "height_mm": plate.get("height_mm"),
            "quantity": 1,
            "material": structure.get("material"),
            "background_color": structure.get("background_color"),
            "text_color": structure.get("text_color"),
            "fixing": structure.get("fixing"),
            "notes": None,
            "bbox_px": plate.get("bbox_px"),
            "lines": lines_out,
            "holes": [],
        }
        sort_lines_by_y(label)
        _refine_line_y_positions(label, warnings)
        labels.append(label)

    spec = {
        "unit": "mm",
        "image_px": image_px,
        "dimension_annotations": dimension_annotations,
        "labels": labels,
        "total_labels": len(labels),
        "warnings": list(warnings),
    }
    run_measure(spec, warnings)
    spec["warnings"] = list(warnings)
    return spec
