"""Node 1 — survey the draft sheet."""

from pathlib import Path

from context import SheetContext
from plate_detect import file_image_px
from schema import SURVEY_SCHEMA
from vision import call_vision, guess_mime, image_to_base64

SURVEY_PROMPT = """\
You are surveying a client's label-plate draft (CAD drawing, sketch, or photo).

Return JSON only:
- draft_type: "cad", "sketch", "photo", or "other"
- material_notes: any global notes about material, colour, fixing, or Traffolyte
  that apply to multiple plates (verbatim). null if none.

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

    ctx.image_px = file_image_px(image_path)
    return ctx
