"""Layout-agnostic prompts for the 2-call extract path."""

STRUCTURE_PROMPT = """\
This label draft image is exactly {width}x{height} pixels.

Find every physical label PLATE to manufacture - one entry per separate
rectangular piece that would be cut/produced independently.

Computer vision pre-scan (hints only — confirm or correct from the image):
{cv_hint}

Use the VISIBLE OUTLINE or BORDER of each piece as the boundary:
- A bordered box/slot = one plate
- Multiple side-by-side bordered cells in one row = one plate PER CELL (not one full-width strip)
- A single horizontal row outline with column guides but NO per-cell borders = one plate for the whole row
- Do NOT split one bordered box into multiple plates
- Do NOT merge multiple bordered boxes into one plate
- Dimension lines / arrows are NOT plate borders — never use dimension span positions as plate bbox

For each plate return:
- id: 1, 2, 3... top-to-bottom then left-to-right
- bbox_frac: [x1, y1, x2, y2] as FRACTIONS of the full image in [0, 1]
  Tightly enclose the whole plate outline including every text line inside it.
- width_mm / height_mm: copy ONLY written mm numbers for that plate; null if not stated

Map EVERY dimension line / arrow on the sheet:
- value_mm: the written measurement number
- axis: "horizontal" for widths, "vertical" for heights
- span_frac: [start, end] as FRACTIONS along that axis (between arrowheads /
  extension lines), not the text position

Read global spec notes when visible: material, colors, fixing, draft_type.
If the sheet states a single letter size for all text (e.g. "Letter Size: 5mm"),
set default_size_mm. If individual lines have stated sizes (e.g. "20mm 1st Line"),
list them in line_sizes.

Do NOT transcribe label text or dimension annotation numbers as plate content.
"""

CONTENT_PROMPT = """\
This label draft image is exactly {width}x{height} pixels.

Structure pass detected these plates (use these ids):
{plate_list}

For each plate id:
1. Count how many separate text lines are visible on that plate → set line_count.
2. Return exactly line_count entries in lines (same count).

Line counting rules:
- One printed title block = one line (e.g. "ROOM 31" → line_count 1, text="ROOM 31").
- Side-by-side words in different columns = separate lines (e.g. CALLING | FAULT | DEFROST → 3).
- Stacked rows in a column = separate lines (e.g. ROOM, DISABLE, ENABLE → 3).
- Never merge words from different columns into one line text field.

If written mm dimensions for that plate are visible, set width_mm / height_mm:
- width_mm = TOTAL width of the full plate outline (left to right edge),
  NOT the width of one column cell inside a multi-column plate
- height_mm = total plate height; otherwise null

POSITION (relative to THAT plate's top-left corner):
- x_mm = horizontal center of text from the plate's LEFT edge
- y_mm = vertical center of text from the plate's TOP edge
- Each text string is its own line entry (e.g. ROOM, DISABLE, ENABLE are three separate lines)
- Never merge multiple words into one line text field
- Texts on the SAME horizontal row share the same y_mm (e.g. DISABLE and ENABLE side-by-side)
- Each genuinely stacked row gets its own y_mm — do NOT stack side-by-side pairs vertically
- A plate with only one row of text should have y_mm near the vertical center of the plate

SIZE:
- size_mm = capital letter height in millimetres
- If the drawing states a size (e.g. "20mm", "Letter Size: 5mm"), use that exact number
- Otherwise judge visually: compare letter height to the plate height in mm
- A title line can be much larger than fine print - do NOT flatten all lines to one size
- If you truly cannot estimate, return null (do NOT guess a default)

Also return bbox_frac [x1,y1,x2,y2] tight around each text line (fractions of full image).

Do NOT transcribe dimension-line annotation numbers as label text.
If text belongs to a different plate than assigned, put it under the correct plate id.
If a plate from the structure pass has no matching outline, still transcribe visible
text faithfully and note mismatches only by placing text under the correct plate id.
"""


def format_plate_list(plates: list[dict]) -> str:
    lines = []
    for plate in plates:
        frac = plate.get("bbox_frac") or []
        w = plate.get("width_mm")
        h = plate.get("height_mm")
        dims = f"{w}x{h}mm" if w is not None and h is not None else "dims unknown"
        lines.append(f"  #{plate.get('id')}: bbox_frac={frac} ({dims})")
    return "\n".join(lines) or "  (none)"
