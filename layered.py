"""Layered vision pipeline — works the way a designer does, in passes.

Each stage is its own schema-constrained vision call:

  1. detect     (full image)  plates + their text content. NO layout math.
                              Each plate is then CROPPED so later stages get
                              clean single-plate context. Most crucial stage.
  2. size        (per crop)    capital-letter height of each text vs the plate.
  3. position    (per crop)    center x/y of each text, matching the drawing.
  4. review      (per crop +   checklist against a RENDER of the computed
                  render)      layout; routes the ONE worst issue back to the
                              stage that owns it, capped so it never loops.

Orchestration is plain Python: a per-plate state machine with a hard
iteration cap. Only the offending plate is revised, never the whole sheet.
Vision produces the numbers; the deterministic postprocess (main.py) is the
final reconciliation net.
"""

from PIL import Image

from schema import (
    DETECT_SCHEMA,
    POSITION_SCHEMA,
    REDETECT_SCHEMA,
    REVIEW_SCHEMA,
    SIZE_SCHEMA,
)
from vision import call_vision, crop_plate, guess_mime, image_to_base64, pil_to_base64
from plate_render import render_plate

MAX_REVIEW_ATTEMPTS = 2      # per plate, then accept with a warning
MAX_PLATES = 60              # runaway guard

DETECT_PROMPT = """\
You are reading a client's draft of label plates to manufacture. Your ONLY job
here is to identify the PHYSICAL PLATES and their text content. Do not compute
any text layout, sizes, or positions yet.

A plate is one rectangle that is cut as a single piece. Count carefully:
- A row of separately outlined rectangles, each with its own full border, is
  SEVERAL plates.
- ONE outlined rectangle containing thin internal guide lines that merely
  align text is a SINGLE plate — those guides are not cut lines. A continuous
  dimension chain along an edge (e.g. 25, 100, 100, 25) spans segments of ONE
  plate, not several plates.
- Ignore the form's title block (Date, Client, Project, Quote/Job No,
  Drawn by, Checked) — it is not a plate.

For each plate provide:
- bbox_px: [x1, y1, x2, y2] of the plate's own outline rectangle ONLY —
  exclude dimension arrows, and exclude any note/legend text drawn OUTSIDE the
  rectangle.
- width_mm / height_mm: from annotated dimension lines only; null if not
  annotated (never guess a number).
- quantity: from a marking like "3x" / "QTY 2"; null if not stated.
- material / background_color / text_color / fixing: from spec notes near the
  drawing (a sheet-wide note applies to every plate it refers to); null if
  absent.
- notes: any other free-form remark (e.g. a "Letter Size: 15mm" instruction);
  null if none.
- texts: ONLY the text that is printed INSIDE the plate's outline rectangle —
  the words that get engraved on the plate itself. Top-to-bottom, each with a
  "line" index (1,2,3...). Texts separated by a clear gap or a guide line are
  SEPARATE entries (e.g. two words side by side are two lines, never joined).
  Transcribe exactly.
  Do NOT include: dimension numbers; and do NOT include spec/legend lines that
  sit OUTSIDE the plate rectangle such as "Letter Size: ...", "Material: ...",
  "Background Colour: ...", "Text Colour: ...", "Fixing: ..." — those describe
  the plate, they are not engraved on it, so they belong in the fields above,
  never in texts.
- image_px: the pixel width/height of the image you see (anchors bbox_px).

Getting the plate boundaries, the count, and "engraved text vs surrounding
note" right is the most important thing.
"""

REDETECT_PROMPT = """\
This image is a crop of ONE label plate. Re-read it carefully.
- width_mm / height_mm: from annotated dimensions only; null if none.
- texts: ONLY the text engraved INSIDE the plate rectangle, top-to-bottom,
  each with a "line" index and exact transcription. Separate items (gap or
  guide line between them) are separate lines. Do NOT include dimension
  numbers, or spec/legend notes like "Letter Size / Material / Background
  Colour / Text Colour / Fixing" that sit outside or beside the plate.
"""

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
  CLEAR mismatch (missing or extra words, a text hugely wrong in size, or a
  text in the clearly wrong place). Minor differences pass.
- fix: when revising, the SINGLE most impactful stage — "detect" (wrong or
  missing words / wrong plate), "size" (sizes clearly off), "position"
  (clearly wrong placement); null when verdict is "pass".
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


def stage_detect(image_path: str, warnings: list[str]) -> dict:
    result = call_vision(
        DETECT_PROMPT,
        [(image_to_base64(image_path), guess_mime(image_path))],
        schema=DETECT_SCHEMA, label="detect", max_tokens=12000,
    )
    if result.get("error") == "parse_failed":
        warnings.append("detect stage returned invalid JSON")
        return {"unit": "mm", "image_px": None, "plates": []}
    return result


def stage_redetect(crop_b64: str, warnings: list[str], plate_id) -> dict:
    result = call_vision(
        REDETECT_PROMPT, [(crop_b64, "image/png")],
        schema=REDETECT_SCHEMA, label=f"redetect#{plate_id}", max_tokens=4000,
    )
    if result.get("error") == "parse_failed":
        warnings.append(f"plate {plate_id}: re-detect returned invalid JSON")
        return {}
    return result


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


def _assemble_lines(texts: list[dict], sizes: dict, positions: dict) -> list[dict]:
    """Merge stage outputs into line dicts keyed by the detect line index."""
    size_by = {s.get("line"): s.get("size_mm") for s in sizes.get("lines") or []}
    pos_by = {p.get("line"): (p.get("x_mm"), p.get("y_mm"))
              for p in positions.get("lines") or []}
    lines = []
    for i, t in enumerate(texts):
        key = t.get("line", i + 1)
        x, y = pos_by.get(key, (None, None))
        lines.append({
            "line": key, "text": t.get("text", ""),
            "size_mm": size_by.get(key), "x_mm": x, "y_mm": y,
        })
    return lines


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


def _build_label(number: int, plate: dict, lines: list[dict]) -> dict:
    out_lines = []
    for ln in lines:
        measured = [f for f in ("x_mm", "y_mm", "size_mm")
                    if isinstance(ln.get(f), (int, float))]
        out_lines.append({
            "text": ln.get("text", ""),
            "x_mm": ln.get("x_mm"), "y_mm": ln.get("y_mm"),
            "size_mm": ln.get("size_mm"),
            "alignment": None, "bold": None,
            "measured_fields": measured,
        })
    return {
        "label_number": number,
        "width_mm": plate.get("width_mm"), "height_mm": plate.get("height_mm"),
        "quantity": plate.get("quantity"),
        "material": plate.get("material"),
        "background_color": plate.get("background_color"),
        "text_color": plate.get("text_color"),
        "fixing": plate.get("fixing"), "notes": plate.get("notes"),
        "bbox_px": plate.get("bbox_px"),
        "lines": out_lines, "holes": [],
    }


def _process_plate(
    image: Image.Image, detect: dict, plate: dict, warnings: list[str],
    allow_redetect: bool = True,
) -> list[dict]:
    """Measure one plate's geometry from its crop, then review-and-revise
    (capped). ``allow_redetect=False`` (hybrid mode) means the text and
    decomposition are authoritative from extract.py — the review may only
    re-measure size/position, never re-read the text."""
    pid = plate.get("plate_id", "?")
    crop = crop_plate(image, plate.get("bbox_px") or [0, 0, image.width, image.height],
                      detect.get("image_px"))
    crop_b64 = pil_to_base64(crop)
    texts = plate.get("texts") or []

    lines = _assemble_lines(texts, {"lines": []}, {"lines": []})
    _measure_size(lines, crop_b64, plate, warnings)
    _measure_position(lines, crop_b64, plate, warnings)

    for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
        render = render_plate({**plate, "lines": lines}, crop.width, crop.height)
        review = stage_review(crop_b64, pil_to_base64(render), plate, warnings)
        if review.get("verdict") == "pass":
            break
        fix = review.get("fix")
        note = review.get("notes") or "geometry off"

        if fix == "detect" and not allow_redetect:
            # trusted text from extract — the review misread the crop, or the
            # issue is really placement; nudge position once, never re-read text
            warnings.append(f"plate {pid}: review flagged 'detect' but text is "
                            f"authoritative — re-measuring position instead [attempt {attempt}]")
            _measure_position(lines, crop_b64, plate, warnings)
            continue

        warnings.append(f"plate {pid}: review revise ({fix}) — {note} [attempt {attempt}]")
        if fix == "detect":
            rd = stage_redetect(crop_b64, warnings, pid)
            if rd.get("texts"):
                plate["texts"] = rd["texts"]
                texts = rd["texts"]
            for dim in ("width_mm", "height_mm"):
                if isinstance(rd.get(dim), (int, float)):
                    plate[dim] = rd[dim]
            lines = _assemble_lines(texts, {"lines": []}, {"lines": []})
            _measure_size(lines, crop_b64, plate, warnings)
            _measure_position(lines, crop_b64, plate, warnings)
        elif fix == "size":
            _measure_size(lines, crop_b64, plate, warnings)
        elif fix == "position":
            _measure_position(lines, crop_b64, plate, warnings)
        else:
            break
    else:
        warnings.append(f"plate {pid}: still not clean after {MAX_REVIEW_ATTEMPTS} "
                        "review attempts — accepted as-is")
    return lines


def _label_to_plate(label: dict) -> dict:
    """Adapt an extract.py label (decomposition + text) into the plate shape
    the per-crop stages consume. Only text content is carried over — sizes and
    positions are re-measured from the crop (the layered geometry win)."""
    lines = label.get("lines") or []
    return {
        "plate_id": label.get("label_number"),
        "bbox_px": label.get("bbox_px"),
        "width_mm": label.get("width_mm"),
        "height_mm": label.get("height_mm"),
        "quantity": label.get("quantity"),
        "material": label.get("material"),
        "background_color": label.get("background_color"),
        "text_color": label.get("text_color"),
        "fixing": label.get("fixing"),
        "notes": label.get("notes"),
        "texts": [
            {"line": i + 1, "text": ln.get("text", "")}
            for i, ln in enumerate(lines)
        ],
    }


def _apply_measured(label: dict, refined: list[dict]) -> None:
    """Overwrite a label's line geometry with per-crop measured values
    (matched positionally — refined lines came from this label's texts)."""
    for orig, new in zip(label.get("lines") or [], refined):
        for field in ("x_mm", "y_mm", "size_mm"):
            orig[field] = new.get(field)
        orig["measured_fields"] = list(new.get("measured_fields") or [])
        orig.pop("computed_fields", None)


def refine_with_layers(spec: dict, image_path: str, warnings: list[str]) -> int:
    """Direction B (triggered): extract.py + calibrate is the geometry
    baseline; only plates flag_suspect_plates marked ``needs_refinement`` get
    re-measured by the per-crop layered stages (size/position/review). Text and
    decomposition stay authoritative from extract — never re-read here.
    Returns the number of plates refined."""
    flagged = [lab for lab in spec.get("labels") or []
               if lab.get("needs_refinement")]
    if not flagged:
        return 0

    detect = {"image_px": spec.get("image_px")}
    image = Image.open(image_path)
    count = 0
    for label in flagged[:MAX_PLATES]:
        plate = _label_to_plate(label)
        if not plate.get("texts") or not plate.get("bbox_px"):
            continue
        refined = _process_plate(image, detect, plate, warnings, allow_redetect=False)
        _apply_measured(label, refined)
        count += 1
    return count


def run_layered(image_path: str, warnings: list[str]) -> dict:
    detect = stage_detect(image_path, warnings)
    plates = detect.get("plates") or []
    if len(plates) > MAX_PLATES:
        warnings.append(f"detected {len(plates)} plates — capped at {MAX_PLATES}")
        plates = plates[:MAX_PLATES]

    image = Image.open(image_path)
    labels = []
    for i, plate in enumerate(plates, start=1):
        lines = _process_plate(image, detect, plate, warnings)
        labels.append(_build_label(i, plate, lines))

    return {
        "unit": detect.get("unit", "mm"),
        "image_px": detect.get("image_px"),
        "dimension_annotations": [],
        "labels": labels,
    }
