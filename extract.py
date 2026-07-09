"""Single-stage vision extraction: draft image -> raw label spec JSON.

The prompt is deliberately layout-agnostic: it describes HOW to read any
label draft (transcribe text, copy annotated mm values, never guess) and
contains no texts, positions, or offsets from any specific drawing.
Layout rules and arithmetic live in postprocess.py, not here.
"""

import base64
from pathlib import Path

from api_client import API_EXTRACT_MAX_TOKENS, call_chat, parse_json_response
from schema import LABEL_SPEC_SCHEMA, validate_against_schema

EXTRACTION_PROMPT = """\
You are reading a client's draft for engraved/printed label plates. The draft
may be a hand-drawn sketch, a photo of paper, or a CAD-style drawing.

Extract the label manufacturing specification as JSON (schema is enforced).

Definitions:
- A "label" is one physical plate to manufacture: a complete rectangular
  outline in the drawing. One drawn plate = one entry in labels[].
- Plate identity: the plate is the FULL outlined rectangle that gets cut as
  one piece. Thin vertical/horizontal guide lines INSIDE a plate that only
  align text do NOT split it into multiple plates — all text inside one
  continuous outline belongs to one labels[] entry. When one continuous
  dimension chain along an edge (e.g. 25, 100, 100, 25) spans several
  adjacent cells, those cells are segments of ONE plate, not separate
  plates. Only rectangles with their own complete independent border are
  separate plates.
- "lines" are the text strings that will appear ON the plate, in top-to-bottom
  order. Texts separated by a clear horizontal gap or a guide line are
  SEPARATE lines — never join them into one string (e.g. "DISABLE" and
  "ENABLE" side by side are two lines, not "DISABLE ENABLE").

Rules:
1. Transcribe label text exactly as written, including codes and punctuation.
2. Dimension annotations (numbers with mm, arrows, dimension lines, often in a
   different pen color than the label text) are NOT label text. Use them only
   to fill numeric fields.
3. Copy millimeter values ONLY from written annotations. Never calculate,
   estimate, average, or divide dimensions yourself. If a value is not
   written in the draft, use null — positions are computed later from your
   bounding boxes, so null is the correct answer, not a guess.
4. width_mm / height_mm: the plate's annotated outer dimensions.
5. Per line: y_mm = annotated distance from the TOP edge of its plate;
   x_mm = annotated distance from the LEFT edge; size_mm = annotated letter
   height (e.g. "Letter Size: 5mm"). null when not annotated.
   If a letter size is given per line number (e.g. "20mm 1st line, 10mm 2nd
   line"), assign each size to the matching line.
6. alignment: "center" only if the text is clearly centered on the plate or a
   marked guide line, "left"/"right" if clearly anchored; otherwise null.
7. bbox_px (REQUIRED whenever visible): pixel bounding box [x1, y1, x2, y2]
   in the image, x1<x2 top-left origin.
   - For each label: the box of the DRAWN PLATE OUTLINE (the full rectangle),
     excluding dimension arrows and annotations around it.
   - For each line: the TIGHT box hugging that text's ink only — not the row,
     not the cell, no surrounding empty space, no neighbouring annotations.
   Be precise — these boxes are measured to compute real positions.
8. bold: true only if the draft says BOLD for that line; otherwise null.
9. quantity: from markings like "3x", "x3", "QTY 2". null if not stated
   (do not assume 1).
10. material / background_color / text_color / fixing: fill from written notes
    (e.g. "Traffolyte", "Background: Yellow", "Writing: Black",
    "Self-adhesive"). A note next to the whole drawing applies to all labels
    it points to. null when absent.
11. notes: any remaining free-form remarks near that label, verbatim.
12. holes: only if the draft shows mounting holes with annotations.
13. Ignore title-block fields of the form sheet itself (Date, Client, Project,
    Quote/Job No, Drawn by, Checked) — they are not label content.
14. unit: "mm" unless the draft clearly uses another unit — then state it and
    still copy numbers as written.
15. image_px: the pixel width and height of the image as you see it — this
    anchors your bbox_px coordinate space.
16. dimension_annotations (IMPORTANT): list EVERY dimension line/arrow in the
    drawing — the small numbers with arrows or extension lines that state a
    measurement (e.g. "65 mm" above a cell, "20 mm" beside a strip, "216 mm"
    down the side, the "25 100 100 25" chain along an edge). For each:
    - value_mm: the written number.
    - axis: "horizontal" if it measures a width (left-right), "vertical" if a
      height (up-down).
    - span_px: [start, end] pixel coordinates of the distance it measures
      (between its two extension lines / arrowheads), along its axis —
      x-coordinates for horizontal, y-coordinates for vertical. This is the
      SPAN being measured, not the position of the number text.
    These anchor the real millimetre scale, so be accurate. Empty list only
    if the drawing truly has no dimension lines.
"""


def image_to_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def guess_mime(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def extract_specs(image_path: str | Path) -> tuple[dict, list[str]]:
    """Run the vision call. Returns (raw_spec, schema_errors)."""
    image_b64 = image_to_base64(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{guess_mime(image_path)};base64,{image_b64}"
                    },
                },
            ],
        }
    ]

    content = call_chat(
        messages,
        label="extract",
        max_tokens=API_EXTRACT_MAX_TOKENS,
        json_schema=LABEL_SPEC_SCHEMA,
    )

    parsed = parse_json_response(content)
    if parsed.get("error") == "parse_failed":
        return parsed, ["response is not valid JSON"]

    return parsed, validate_against_schema(parsed)
