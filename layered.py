"""Per-crop geometry refinement for flagged plates (hybrid pipeline).

extract.py gives the decomposition + text; postprocess.py/calibrate is the
geometry baseline. When flag_suspect_plates marks a plate's baseline geometry
as physically impossible, ``refine_with_layers`` re-measures just that plate
from its crop, in designer-style passes — each stage its own schema-constrained
vision call on the single-plate crop:

  size     capital-letter height of each text vs the plate.
  position center x/y of each text, matching the drawing.
  review   lenient checklist against a PIL render of the computed layout;
           routes the worst issue back to size/position, capped so it never
           loops.

Text and decomposition are authoritative from extract.py and are never re-read
here. Only the offending plates pay for the extra vision calls.
"""

from PIL import Image

from schema import POSITION_SCHEMA, REVIEW_SCHEMA, SIZE_SCHEMA
from vision import call_vision, crop_plate, pil_to_base64
from plate_render import render_plate

MAX_REVIEW_ATTEMPTS = 2      # per plate, then accept with a warning
MAX_PLATES = 60              # runaway guard

SIZE_PROMPT = """\
This image is a crop of ONE label plate. Its real size is {width} mm wide by
{height} mm tall.
For each text below, estimate the height of its CAPITAL letters in millimetres,
judged against the plate's known height (a line filling half a 20mm plate is
~10mm; a small sub-label is smaller). Headings and sub-labels should differ.
Return size_mm for each, matched by its line index.

Texts:
{texts}
"""

POSITION_PROMPT = """\
This image is a crop of ONE label plate. Its real size is {width} mm wide by
{height} mm tall.
For each text, give the position of the CENTER of the text as the distance
from the plate's LEFT edge (x_mm) and from its TOP edge (y_mm), in millimetres,
matching where the text actually sits in the image. Return x_mm and y_mm for
each, matched by its line index.
x_mm must be between 0 and {width}; y_mm between 0 and {height} (they are
positions INSIDE this plate, not pixels).

Texts (with their estimated sizes):
{texts}
"""

REVIEW_PROMPT = """\
Two images follow:
(1) the ORIGINAL crop of a label plate from the client's drawing.
(2) a simplified RENDER of the layout currently computed for it — plain black
    text on a white plate inside a grey margin. It is intentionally schematic:
    no fonts, colours, or styling. Text may spill into the grey margin when it
    is larger than the plate; that is fine to see, judge it as "too big".

The plate is {width} mm x {height} mm. Judge only LAYOUT correspondence, and be
LENIENT — rough agreement passes, it need not be precise:
- plate_size_ok: do the plate proportions roughly match?
- text_count_ok: does the render have the same text items / words as the crop?
- text_size_ok: are the relative text sizes in the right ballpark?
- spacing_ok: are the texts in roughly the same rows / positions?
- verdict: "pass" if all four are acceptable. Only answer "revise" for a
  CLEAR mismatch (a text hugely wrong in size, or a text in the clearly wrong
  place). Minor differences pass.
- fix: when revising, the SINGLE most impactful stage — "size" (sizes clearly
  off) or "position" (clearly wrong placement); null when verdict is "pass".
  (Text is fixed, so never route to detect.)
- notes: one short sentence on what is off, or null.
"""


def _texts_block(texts: list[dict]) -> str:
    return "\n".join(f"  line {t.get('line', i + 1)}: {t.get('text', '')!r}"
                     for i, t in enumerate(texts))


def _texts_with_sizes(lines: list[dict]) -> str:
    return "\n".join(
        f"  line {ln.get('line', i + 1)}: {ln.get('text', '')!r} "
        f"(~{ln.get('size_mm', '?')}mm)"
        for i, ln in enumerate(lines)
    )


def stage_size(crop_b64: str, plate: dict, texts: list[dict], warnings: list[str]) -> dict:
    prompt = SIZE_PROMPT.format(
        width=plate.get("width_mm", "?"), height=plate.get("height_mm", "?"),
        texts=_texts_block(texts),
    )
    result = call_vision(
        prompt, [(crop_b64, "image/png")],
        schema=SIZE_SCHEMA, label=f"size#{plate.get('plate_id', '?')}", max_tokens=3000,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {plate.get('plate_id')}: size stage invalid JSON")
        return {"lines": []}
    return result


def stage_position(
    crop_b64: str, plate: dict, lines: list[dict], warnings: list[str]
) -> dict:
    prompt = POSITION_PROMPT.format(
        width=plate.get("width_mm", "?"), height=plate.get("height_mm", "?"),
        texts=_texts_with_sizes(lines),
    )
    result = call_vision(
        prompt, [(crop_b64, "image/png")],
        schema=POSITION_SCHEMA, label=f"pos#{plate.get('plate_id', '?')}", max_tokens=3000,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {plate.get('plate_id')}: position stage invalid JSON")
        return {"lines": []}
    return result


def stage_review(
    crop_b64: str, render_b64: str, plate: dict, warnings: list[str]
) -> dict:
    prompt = REVIEW_PROMPT.format(
        width=plate.get("width_mm", "?"), height=plate.get("height_mm", "?")
    )
    result = call_vision(
        prompt, [(crop_b64, "image/png"), (render_b64, "image/png")],
        schema=REVIEW_SCHEMA, label=f"review#{plate.get('plate_id', '?')}", max_tokens=1500,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {plate.get('plate_id')}: review invalid JSON — accepting")
        return {"verdict": "pass", "fix": None}
    return result


def _assemble_lines(texts: list[dict]) -> list[dict]:
    return [{"line": t.get("line", i + 1), "text": t.get("text", ""),
             "size_mm": None, "x_mm": None, "y_mm": None}
            for i, t in enumerate(texts)]


def _measure_size(lines: list[dict], crop_b64, plate, warnings) -> None:
    sizes = stage_size(crop_b64, plate, lines, warnings)
    size_by = {s.get("line"): s.get("size_mm") for s in sizes.get("lines") or []}
    for ln in lines:
        if ln["line"] in size_by:
            ln["size_mm"] = size_by[ln["line"]]


def _measure_position(lines: list[dict], crop_b64, plate, warnings) -> None:
    positions = stage_position(crop_b64, plate, lines, warnings)
    pos_by = {p.get("line"): (p.get("x_mm"), p.get("y_mm"))
              for p in positions.get("lines") or []}
    for ln in lines:
        if ln["line"] in pos_by:
            ln["x_mm"], ln["y_mm"] = pos_by[ln["line"]]


def _process_plate(
    image: Image.Image, image_px: dict | None, plate: dict, warnings: list[str]
) -> list[dict]:
    """Re-measure one plate's geometry from its crop, then review-and-revise
    (capped). Text is authoritative — review only re-runs size/position."""
    pid = plate.get("plate_id", "?")
    crop = crop_plate(image, plate.get("bbox_px") or [0, 0, image.width, image.height],
                      image_px)
    crop_b64 = pil_to_base64(crop)
    texts = plate.get("texts") or []

    lines = _assemble_lines(texts)
    _measure_size(lines, crop_b64, plate, warnings)
    _measure_position(lines, crop_b64, plate, warnings)

    for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
        render = render_plate({**plate, "lines": lines}, crop.width, crop.height)
        review = stage_review(crop_b64, pil_to_base64(render), plate, warnings)
        if review.get("verdict") == "pass":
            break
        fix = review.get("fix")
        note = review.get("notes") or "geometry off"
        warnings.append(f"plate {pid}: review revise ({fix}) — {note} [attempt {attempt}]")
        if fix == "size":
            _measure_size(lines, crop_b64, plate, warnings)
        elif fix == "position":
            _measure_position(lines, crop_b64, plate, warnings)
        else:
            # "detect"/null — text is fixed; the mismatch is placement at worst
            _measure_position(lines, crop_b64, plate, warnings)
    else:
        warnings.append(f"plate {pid}: still not clean after {MAX_REVIEW_ATTEMPTS} "
                        "review attempts — accepted as-is")
    return lines


def _label_to_plate(label: dict) -> dict:
    """Adapt an extract.py label into the plate shape the per-crop stages
    consume. Only text content is carried over — geometry is re-measured."""
    lines = label.get("lines") or []
    return {
        "plate_id": label.get("label_number"),
        "bbox_px": label.get("bbox_px"),
        "width_mm": label.get("width_mm"),
        "height_mm": label.get("height_mm"),
        "texts": [
            {"line": i + 1, "text": ln.get("text", "")}
            for i, ln in enumerate(lines)
        ],
    }


def _apply_measured(label: dict, refined: list[dict]) -> None:
    """Overwrite a label's line geometry with per-crop measured values
    (matched positionally — refined lines came from this label's texts)."""
    for orig, new in zip(label.get("lines") or [], refined):
        measured = [f for f in ("x_mm", "y_mm", "size_mm")
                    if isinstance(new.get(f), (int, float))]
        for field in ("x_mm", "y_mm", "size_mm"):
            orig[field] = new.get(field)
        orig["measured_fields"] = measured
        orig.pop("computed_fields", None)


def refine_with_layers(spec: dict, image_path: str, warnings: list[str]) -> int:
    """Re-measure only the plates flag_suspect_plates marked
    ``needs_refinement``. Returns the number of plates refined."""
    flagged = [lab for lab in spec.get("labels") or []
               if lab.get("needs_refinement")]
    if not flagged:
        return 0

    image_px = spec.get("image_px")
    image = Image.open(image_path)
    count = 0
    for label in flagged[:MAX_PLATES]:
        plate = _label_to_plate(label)
        if not plate.get("texts") or not plate.get("bbox_px"):
            continue
        refined = _process_plate(image, image_px, plate, warnings)
        _apply_measured(label, refined)
        count += 1
    return count
