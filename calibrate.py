"""Pixel-to-mm calibration from grounded bounding boxes.

The vision model is good at localizing things in pixels (grounding) and bad
at estimating millimeters, so extraction returns ``bbox_px`` for every plate
and text line and this module does the metric work deterministically.

Client drafts are often NOT drawn to scale (schematic CAD sheets, hand
sketches), so calibration is primarily PER PLATE: a text position inside a
plate is its fractional position within the plate bbox multiplied by that
plate's annotated dimension. This only assumes the draft keeps proportions
roughly right *within* one plate — far safer than assuming a uniform global
scale across the whole sheet.

Scale reference, in priority order:
1. Dimension lines in the drawing (``dimension_annotations``): px span / value.
   These are explicit measurements and are ANCHORED — trusted enough to
   repair a conflicting plate dimension on their own (this fixes the common
   failure where the model states a cell's width as the whole row's width).
2. Fallback: plate bbox vs the model's own plate-width guess, which repairs
   only when the sheet is internally consistent (else one summary warning).

Both are kept per-axis (horizontal / vertical separate — sheets are often
stretched differently per axis) and used to derive dimensions for plates
with no annotation at all.

Guards against sloppy boxes: a measured line coordinate that lands outside
its plate is discarded (the box grabbed a neighbouring annotation), and a
measured text height taller than 60% of the plate is discarded (the box
covered a row band, not the glyphs). Discarded measurements stay null for
the layout resolver to fill.

Measured values are recorded per line/label in ``measured_fields``;
annotated values always win over measurement.
"""

SCALE_MISMATCH_TOLERANCE = 0.20     # dim vs global-scale conflict threshold
SCALE_OUTLIER_FRACTION = 0.25       # >25% outlier samples = not drawn to scale
EDGE_MATCH_TOLERANCE = 0.15         # dimension span vs plate edge match window
COORD_SLACK = 0.05                  # measured coord may exceed plate by 5%
EDGE_CLAMP_EPS_MM = 0.5             # keep measured centers strictly inside plate
EQUAL_BBOX_FRAC = 0.08              # bbox spans within 8% → same drawn column/row
TABLE_EQUAL_BBOX_SHARE = 0.75       # fraction of plates that must share that span
MIN_TABLE_PLATES = 4                # need enough plates to call it a table axis
MIN_TABLE_DISTINCT_DIMS = 2         # …with at least this many distinct annotated sizes
MAX_TEXT_HEIGHT_FRACTION = 0.4      # taller = bbox is a row band, not glyphs (multi-line)
SINGLE_LINE_MAX_TEXT_FRACTION = 0.85  # one line may fill most of a short plate
SLOT_SIZE_FRACTION = 0.65           # cap height vs equal row slot on stacked plates
SHORT_TEXT_SIZE_FRACTION = 0.55     # numbers / codes in a stacked cell
SHORT_TEXT_MAX_CHARS = 3
SUSPECT_SIZE_FRACTION = 0.45        # final size above this flags the plate
CHAR_WIDTH_RATIO = 0.6              # rough text width estimate per char
MIN_PLATE_DIM_MM = 3.0
POSITION_STEP_MM = 0.5
MIN_TEXT_SIZE_MM = 1.0


def _round_step(value: float) -> float:
    return round(value / POSITION_STEP_MM) * POSITION_STEP_MM


def _clamp_center(value: float, dim: float) -> float:
    """Clamp a text-center coordinate into the plate, never onto the edge.

    Clamping onto exactly 0 / ``dim`` used to trip ``flag_suspect_plates``
    (``y >= height``) and send an otherwise-good baseline into refine.
    """
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


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _label_name(label: dict) -> str:
    num = label.get("label_number", "?")
    lines = label.get("lines") or []
    first = f" ({lines[0]['text'][:20]})" if lines and lines[0].get("text") else ""
    return f"label #{num}{first}"


class _AxisScale:
    """Median px/mm for one axis + a consistency verdict.

    ``anchored`` means the samples came from explicit dimension lines in the
    drawing (authoritative), not from the model's own plate-width guesses.
    An anchored scale is trusted enough to REPAIR a conflicting plate
    dimension even without corroborating plate-width agreement."""

    def __init__(self, samples: list[float], anchored: bool = False):
        self.scale: float | None = None
        self.consistent = False
        self.anchored = anchored
        min_samples = 1 if anchored else 3
        if samples:
            median = _median(samples)
            if median > 0:
                self.scale = median
                outliers = sum(
                    1 for s in samples
                    if abs(s - median) / median > SCALE_MISMATCH_TOLERANCE
                )
                self.consistent = (
                    len(samples) >= min_samples
                    and outliers / len(samples) <= SCALE_OUTLIER_FRACTION
                )


def _dimension_scales(spec: dict) -> tuple["_AxisScale", "_AxisScale"]:
    """Scale from explicit dimension lines: |span_px| / value_mm per axis."""
    sx_samples: list[float] = []
    sy_samples: list[float] = []
    for dim in spec.get("dimension_annotations") or []:
        value = dim.get("value_mm")
        span = dim.get("span_px")
        if not (_is_num(value) and value >= MIN_PLATE_DIM_MM):
            continue
        if not (isinstance(span, (list, tuple)) and len(span) == 2
                and all(_is_num(v) for v in span)):
            continue
        length = abs(span[1] - span[0])
        if length <= 0:
            continue
        (sx_samples if dim.get("axis") == "horizontal" else sy_samples).append(
            length / value
        )
    return _AxisScale(sx_samples, anchored=True), _AxisScale(sy_samples, anchored=True)


def _plate_scales(labels: list[dict]) -> tuple["_AxisScale", "_AxisScale"]:
    sx_samples: list[float] = []
    sy_samples: list[float] = []
    for label in labels:
        bbox = label.get("bbox_px")
        if not _valid_bbox(bbox):
            continue
        width, height = label.get("width_mm"), label.get("height_mm")
        if _is_num(width) and width >= MIN_PLATE_DIM_MM:
            sx_samples.append((bbox[2] - bbox[0]) / width)
        if _is_num(height) and height >= MIN_PLATE_DIM_MM:
            sy_samples.append((bbox[3] - bbox[1]) / height)
    return _AxisScale(sx_samples), _AxisScale(sy_samples)


def _tabular_axis(labels: list[dict], axis: str) -> bool:
    """True when plates share nearly equal drawn bbox spans but annotated sizes differ.

    Spec sheets (SIZE column in a table) draw every row the same pixel width
    even when real plate widths are 150 / 130 / 80. A global px/mm from those
    bboxes then "repairs" the correct annotations down to the majority size.
    """
    pairs: list[tuple[float, float]] = []
    for label in labels:
        bbox = label.get("bbox_px")
        if not _valid_bbox(bbox):
            continue
        if axis == "x":
            span = bbox[2] - bbox[0]
            dim = label.get("width_mm")
        else:
            span = bbox[3] - bbox[1]
            dim = label.get("height_mm")
        if _is_num(dim) and dim >= MIN_PLATE_DIM_MM and span > 0:
            pairs.append((float(span), float(dim)))
    if len(pairs) < MIN_TABLE_PLATES:
        return False
    spans = [s for s, _ in pairs]
    med_span = _median(spans)
    if med_span <= 0:
        return False
    equal = [
        (s, d) for s, d in pairs
        if abs(s - med_span) / med_span <= EQUAL_BBOX_FRAC
    ]
    if len(equal) / len(pairs) < TABLE_EQUAL_BBOX_SHARE:
        return False
    distinct = {round(d, 1) for _, d in equal}
    return len(distinct) >= MIN_TABLE_DISTINCT_DIMS


def _axis_scales(spec: dict) -> tuple["_AxisScale", "_AxisScale"]:
    """Prefer dimension-line anchors per axis; fall back to plate-width scale.

    Plate-derived scales that look "consistent" only because a table drew
    equal columns are marked inconsistent so annotated sizes are not repaired.
    """
    labels = spec.get("labels") or []
    dx, dy = _dimension_scales(spec)
    px, py = _plate_scales(labels)
    ax = dx if dx.scale is not None else px
    ay = dy if dy.scale is not None else py
    if not ax.anchored and _tabular_axis(labels, "x"):
        ax.consistent = False
    if not ay.anchored and _tabular_axis(labels, "y"):
        ay.consistent = False
    return ax, ay


def _match_dimension_to_edge(
    label: dict, field: str, edge: tuple[float, float], spec: dict, warnings: list[str]
) -> bool:
    """If a dimension line brackets this plate edge in pixels, its value IS
    that plate dimension (high confidence). Returns True if it set the field.

    This is the safe way to use dimension lines: it only acts when the line
    literally spans the edge, so cumulative position-markers or stray numbers
    (as on the KISO sheet) never touch a plate's size."""
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
    stated = label.get(field)
    measured = label.setdefault("measured_fields", [])
    if _is_num(stated) and abs(stated - value) / max(value, 1e-9) <= SCALE_MISMATCH_TOLERANCE:
        return True  # already agrees, nothing to change
    if _is_num(stated):
        warnings.append(
            f"{_label_name(label)}: {field}={stated:g} corrected to {value:g} "
            "from the dimension line bracketing this plate"
        )
    label[field] = float(value)
    if field not in measured:
        measured.append(field)
    return True


def _fill_dim_from_scale(
    label: dict, field: str, px: float, axis: "_AxisScale", warnings: list[str]
) -> bool:
    """Fill an unannotated plate dimension from the global scale, or repair a
    stated one only when a plate-derived scale is internally consistent.

    Annotated sizes always win on schematic / tabular axes (``axis.consistent``
    False) — e.g. a SIZE column listing 150×70 next to an 80-wide majority.
    Returns True when a stated value disagreed on a schematic sheet."""
    stated = label.get(field)
    if axis.scale is None:
        return False
    implied = px / axis.scale
    measured = label.setdefault("measured_fields", [])

    if not _is_num(stated):
        label[field] = max(MIN_PLATE_DIM_MM, _round_step(implied))
        measured.append(field)
        warnings.append(
            f"{_label_name(label)}: {field} not annotated — measured "
            f"{label[field]:g}mm from the drawing"
        )
        return False

    if abs(stated - implied) / max(implied, 1e-9) <= SCALE_MISMATCH_TOLERANCE:
        return False

    # No dimension line bracketed this edge. A plate-derived scale may repair
    # only when the sheet is internally consistent; tabular equal-bbox axes
    # and bare dimension-derived scales are NOT trusted to overwrite SIZE text.
    if axis.consistent and not axis.anchored:
        fixed = max(MIN_PLATE_DIM_MM, _round_step(implied))
        warnings.append(
            f"{_label_name(label)}: {field}={stated:g} conflicts with measured "
            f"{implied:.1f}mm — replaced with {fixed:g} (likely misread dimension)"
        )
        label[field] = fixed
        measured.append(field)
        return False

    return True  # keep annotation, count for schematic summary


def _estimate_size_from_bbox(
    text_h_mm: float, slot_h: float | None, line_count: int, text: str
) -> float:
    """Turn a tight bbox height into cap-letter height mm.

    Multi-line plates: bbox often spans the whole row cell, so cap the estimate
    to a fraction of the per-line slot. Short tokens (e.g. "73") are smaller."""
    if line_count > 1 and slot_h and slot_h > 0:
        cap = slot_h * SLOT_SIZE_FRACTION
        if len((text or "").strip()) <= SHORT_TEXT_MAX_CHARS:
            cap = min(cap, slot_h * SHORT_TEXT_SIZE_FRACTION)
        return min(text_h_mm, cap)
    return text_h_mm


def _measure_lines(label: dict, sheet_to_scale: bool, warnings: list[str]) -> None:
    """Positions/sizes from fractional position inside THIS plate's bbox,
    scaled by this plate's own dimensions (center-anchored, like the editor)."""
    bbox = label["bbox_px"]
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
            # else: box grabbed something outside the plate — leave null

        if sy:
            if not _is_num(line.get("y_mm")):
                cy = ((lb[1] + lb[3]) / 2 - bbox[1]) / sy
                if -COORD_SLACK * height <= cy <= (1 + COORD_SLACK) * height:
                    line["y_mm"] = _clamp_center(cy, height)
                    measured.append("y_mm")

            text_h = (lb[3] - lb[1]) / sy
            text_h = _estimate_size_from_bbox(
                text_h, slot_h, line_count, line.get("text", "")
            )
            glyph_box = 0 < text_h <= max_text_frac * height
            if not _is_num(line.get("size_mm")):
                if glyph_box:
                    line["size_mm"] = max(MIN_TEXT_SIZE_MM, _round_step(text_h))
                    measured.append("size_mm")
            elif (
                sheet_to_scale
                and glyph_box
                and abs(line["size_mm"] - text_h) / text_h > 0.8
            ):
                warnings.append(
                    f"{name}: '{line.get('text', '')}' annotated size "
                    f"{line['size_mm']:g}mm vs drawn ~{text_h:.1f}mm — check "
                    "the letter-size note"
                )


def _line_extent(line: dict) -> tuple[float, float] | None:
    """Rough horizontal extent (x1, x2) of a center-anchored text line."""
    x, size = line.get("x_mm"), line.get("size_mm")
    if not (_is_num(x) and _is_num(size)):
        return None
    half = len(line.get("text") or "") * size * CHAR_WIDTH_RATIO / 2
    return (x - half, x + half)


def flag_suspect_plates(spec: dict, warnings: list[str]) -> list[dict]:
    """Deterministic sanity check on each plate's final line geometry.

    A plate is flagged ``needs_refinement`` when its values — whatever their
    source — cannot be physically right: text taller than ~half the plate,
    two lines at the same position, or estimated text boxes overlapping.
    Flagged plates are candidates for the selective pass-2 refinement."""
    flagged: list[dict] = []
    for label in spec.get("labels") or []:
        height = label.get("height_mm")
        lines = label.get("lines") or []
        reasons: set[str] = set()

        if _is_num(height):
            line_count = len(lines)
            size_limit_frac = 0.85 if line_count == 1 else SUSPECT_SIZE_FRACTION
            for line in lines:
                size = line.get("size_mm")
                if _is_num(size) and size > size_limit_frac * height:
                    reasons.add("text taller than ~half the plate")
                y = line.get("y_mm")
                if _is_num(y):
                    # Centers only — glyph extent past the edge is common for
                    # near-edge rows and is warned in physical_checks, not a
                    # refine trigger (avoids false positives from clamp-to-edge).
                    if y < -EDGE_CLAMP_EPS_MM or y > height + EDGE_CLAMP_EPS_MM:
                        reasons.add("text centered on or past the plate edge")
        width = label.get("width_mm")
        if _is_num(width):
            for line in lines:
                x = line.get("x_mm")
                if _is_num(x) and (
                    x < -EDGE_CLAMP_EPS_MM or x > width + EDGE_CLAMP_EPS_MM
                ):
                    reasons.add("text centered on or past the plate edge")

        seen: set[tuple] = set()
        for line in lines:
            x, y = line.get("x_mm"), line.get("y_mm")
            if _is_num(x) and _is_num(y):
                if (x, y) in seen:
                    reasons.add("multiple texts at the same position")
                seen.add((x, y))

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                ya, yb = a.get("y_mm"), b.get("y_mm")
                sa, sb = a.get("size_mm"), b.get("size_mm")
                if not all(_is_num(v) for v in (ya, yb, sa, sb)):
                    continue
                if abs(ya - yb) >= (sa + sb) / 2:
                    continue  # different rows
                ea, eb = _line_extent(a), _line_extent(b)
                if ea and eb and ea[0] < eb[1] and eb[0] < ea[1]:
                    reasons.add("texts overlap each other")

        if reasons:
            label["needs_refinement"] = True
            flagged.append(label)
            warnings.append(
                f"{_label_name(label)}: geometry looks wrong "
                f"({'; '.join(sorted(reasons))}) — flagged for refinement"
            )
    return flagged


def calibrate(spec: dict, warnings: list[str]) -> None:
    """Fill/repair mm geometry from pixel bboxes. Mutates ``spec`` in place."""
    labels = spec.get("labels") or []
    tabular_x = _tabular_axis(labels, "x")
    tabular_y = _tabular_axis(labels, "y")
    ax, ay = _axis_scales(spec)
    sheet_to_scale = ax.consistent and ay.consistent
    if ax.scale is not None or ay.scale is not None:
        spec["px_per_mm"] = {
            "x": round(ax.scale, 3) if ax.scale else None,
            "y": round(ay.scale, 3) if ay.scale else None,
            "sheet_to_scale": sheet_to_scale,
            "anchored": ax.anchored or ay.anchored,
            "tabular_x": tabular_x,
            "tabular_y": tabular_y,
        }
    if tabular_x or tabular_y:
        axes = []
        if tabular_x:
            axes.append("widths")
        if tabular_y:
            axes.append("heights")
        warnings.append(
            "plate "
            + "/".join(axes)
            + " look tabular (equal drawn spans, varying annotated sizes) — "
            "keeping annotated dimensions"
        )

    schematic_conflicts = 0
    for label in labels:
        bbox = label.get("bbox_px")
        if not _valid_bbox(bbox):
            if any(not _is_num(label.get(f)) for f in ("width_mm", "height_mm")):
                warnings.append(
                    f"{_label_name(label)}: no plate bbox — missing dimensions "
                    "cannot be measured"
                )
            continue
        # 1) high-confidence: a dimension line bracketing this plate edge
        if not _match_dimension_to_edge(
            label, "width_mm", (bbox[0], bbox[2]), spec, warnings
        ):
            # 2) fallback: global scale (fill) / consistent plate-scale (repair)
            schematic_conflicts += _fill_dim_from_scale(
                label, "width_mm", bbox[2] - bbox[0], ax, warnings
            )
        if not _match_dimension_to_edge(
            label, "height_mm", (bbox[1], bbox[3]), spec, warnings
        ):
            schematic_conflicts += _fill_dim_from_scale(
                label, "height_mm", bbox[3] - bbox[1], ay, warnings
            )
        _measure_lines(label, sheet_to_scale, warnings)

    if schematic_conflicts:
        warnings.append(
            f"sheet does not appear to be drawn to scale ({schematic_conflicts} "
            "dimension(s) disagree with drawn sizes) — annotated values kept"
        )
