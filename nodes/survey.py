"""Node 1 — survey the draft sheet."""

from pathlib import Path

from context import SheetContext
from plate_detect import file_image_px
from schema import SURVEY_SCHEMA
from vision import call_vision, guess_mime, image_to_base64

SURVEY_PROMPT = """\
You are surveying a client's label-plate draft (CAD drawing, sketch, or photo).
Read any GLOBAL specification notes written on the sheet — often a list such as
"Material: ...", "Background Colour: ...", "Text Colour: ...", "Fixing: ...",
"Letter Size: 20mm BOLD (1st Line)".

Return JSON only:
- draft_type: "cad", "sketch", "photo", or "other"
- material_notes: the spec-note block verbatim. null if none.
- material: the material ONLY, e.g. "Traffolyte" or "SS316". null if no material
  is stated (do NOT put letter-size / colour text here).
- background_color: e.g. "Yellow" or "White". null if not stated.
- text_color: e.g. "Black". null if not stated.
- fixing: e.g. "Self-adhesive". null if not stated.
- default_size_mm: a SINGLE letter height that applies to all text
  (e.g. "Letter Size: 5mm"). null if no single global size is stated.
- line_sizes: only when DIFFERENT heights are given per line
  (e.g. "20mm 1st Line", "10mm 2nd Line"): list {line: 1-based index, size_mm}.
  Empty list otherwise.

Do NOT list plates, text, dimensions, or image size here.
"""


def run_survey(image_path: str | Path, ctx: SheetContext) -> SheetContext:
    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    result = call_vision(
        SURVEY_PROMPT,
        [(image_b64, mime)],
        schema=SURVEY_SCHEMA,
        label="survey",
        max_tokens=2000,
    )
    if result.get("error") == "parse_failed":
        ctx.warnings.append("survey: invalid JSON response")
    else:
        ctx.draft_type = result.get("draft_type")
        ctx.material_notes = result.get("material_notes")
        ctx.material = result.get("material")
        ctx.background_color = result.get("background_color")
        ctx.text_color = result.get("text_color")
        ctx.fixing = result.get("fixing")
        ctx.default_size_mm = result.get("default_size_mm")
        ctx.line_sizes = result.get("line_sizes") or []

    ctx.image_px = file_image_px(image_path)
    return ctx
