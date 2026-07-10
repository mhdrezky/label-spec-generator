"""Node 2 — map all dimension lines on the sheet."""

from pathlib import Path

from context import SheetContext
from plate_detect import file_image_px
from schema import DIMENSIONS_SCHEMA
from vision import call_vision, guess_mime, image_to_base64

DIMENSIONS_PROMPT = """\
Map EVERY dimension line / arrow in this label draft.

The image file is exactly {width}×{height} pixels. All span_px values MUST use
this coordinate system (origin top-left, x right, y down).

For each measurement number with extension lines or arrows:
- value_mm: the written number
- axis: "horizontal" for widths, "vertical" for heights
- span_px: [start, end] pixel coordinates of the distance measured
  (x-coordinates for horizontal, y-coordinates for vertical).
  This is the SPAN between arrowheads / extension lines, not the text position.

Ignore label text. Empty list only if there are truly no dimension lines.
"""


def run_dimensions(image_path: str | Path, ctx: SheetContext) -> SheetContext:
    px = file_image_px(image_path)
    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    prompt = DIMENSIONS_PROMPT.format(width=px["width"], height=px["height"])
    result = call_vision(
        prompt,
        [(image_b64, mime)],
        schema=DIMENSIONS_SCHEMA,
        label="dimensions",
        max_tokens=8000,
    )
    if result.get("error") == "parse_failed":
        ctx.warnings.append("dimensions: invalid JSON response")
        return ctx

    ctx.dimension_annotations = result.get("dimension_annotations") or []
    return ctx
