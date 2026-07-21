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
- A tall bordered STRIP subdivided by internal horizontal lines where every
  cell holds ONLY a single sequential number (e.g. 73,74,75...84) is ONE
  plate — the whole strip; the numbers become its text lines, never one
  plate per number.
- This number-strip rule applies ONLY to pure number sequences. Stacked
  bordered cells that each contain their own distinct label WORDS
  (e.g. "GEOMETRY REJECTS" above "PINHOLE REJECTS") are SEPARATE plates —
  one plate per cell. When in doubt, keep bordered cells separate.
- Every plate you return MUST correspond to an outline you can actually see in the image —
  never invent an extra plate that has no visible border at all
- OVER-SPLIT CHECK: if your plate count is far larger than the visible boxes — or
  more than ~50% above the CV pre-scan hint above — you are splitting a repeated
  grid or a numbering strip into pieces. Stop and collapse it back to whole plates.
- Dimension lines / arrows are NOT plate borders — never use dimension span positions as plate bbox

Treat the image as a SPEC TABLE / SCHEDULE if it is a bordered grid whose rows
repeat one column layout — a description column plus attribute columns (LABEL
DESCRIPTION, FONT, SIZE, COLOUR, QTY). This holds EVEN WHEN the header row is not
visible (e.g. a continuation page): infer the columns from the repeating row
layout, where the SIZE column is the cell holding a value like "150 x 70". When
it is a spec table:
- Each DATA ROW below the header = one plate
- bbox_frac = label preview in the LABEL DESCRIPTION column only (not SIZE/COLOUR/QTY columns)
- width_mm / height_mm = copy the SIZE column (e.g. "150 x 70" → 150, 70)
- Skip page headers and column header rows
- FIRST count the total number of data rows N and report it, THEN return
  exactly N plates — one plate per DATA ROW. Work top-to-bottom without
  stopping; the final row MUST be included, and do NOT skip near-identical
  rows. But NEVER duplicate a row to match its QTY column value — a row
  with QTY 3 is still ONE plate (quantity is metadata, not extra plates)

If the image is a PHYSICAL PLATE DRAWING (bordered label pieces with text printed on them):
- bbox_frac = the full bordered plate outline including ALL columns of text on that piece
- One horizontal strip with ROOM | FANS | SUCTION columns = ONE plate (full strip width)

HAND-DRAWN SKETCH quantities & lists:
- GATE: apply this section ONLY to a rough hand-drawn/pencil sketch that has
  NO column headers anywhere. If the sheet shows ANY of FONT / SIZE / COLOUR /
  QTY column headers — OR is a printed/CAD grid with repeating attribute columns
  even without headers — it is a SPEC TABLE: SKIP this whole section, never
  expand or duplicate rows, and never turn a QTY value or a number range like
  "(12-20)" into extra plates.
- On a qualifying sketch, a quantity note written beside a plate ("x3", "3x")
  means make that many IDENTICAL plates — emit that many separate entries,
  each reusing that plate's bbox_frac and dims.
- On a qualifying sketch, a dashed / bulleted list of names beside or below one
  drawn example box (e.g. "- Room 28   - Room 29   - Room 30") means one plate
  PER list item — emit one plate per name, reusing the drawn example box's dims
  (width_mm / height_mm). The example box is not an extra plate beyond the list.

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

SHEET TYPE — apply the matching rules:

GLOBAL (all sheet types): never transcribe column-header words (LABEL
DESCRIPTION / FONT / SIZE / COLOUR / QTY) or bare attribute values (e.g.
"STD 4mm", "BOLD 3mm", "30 x 15", "W/B", "2") as label text — only the actual
finished label wording becomes lines.

SPEC TABLE (rows share a description column plus attribute columns FONT / SIZE /
COLOUR / QTY — even if the header row is not visible, e.g. a continuation page):
- Transcribe the label preview inside each row's LABEL DESCRIPTION cell only
- Copy width_mm / height_mm from the SIZE column on that row
- Do NOT transcribe the FONT / SIZE / COLOUR / QTY columns at all — neither the
  header cells NOR their data VALUES (e.g. "STD 4mm", "30 x 15", "W/B", "2").
  Only the LABEL DESCRIPTION text becomes lines.
- A row whose LABEL DESCRIPTION cell holds 2-3 stacked text lines is still ONE
  row; transcribe all its lines as SEPARATE line entries — never join them into
  one blob. Cover EVERY row from the structure plate list, including the
  multi-line rows at the top — never drop them.

PHYSICAL PLATE DRAWING (bordered pieces with text printed on the plate):
- Transcribe EVERY text string on that physical plate — all columns and rows
- A wide strip with ROOM | FANS | SUCTION and LOW/HIGH below = one plate, 9+ lines
- A status strip with CALLING | FAULT | DEFROST = one plate, 3 lines
- width_mm = FULL plate outline width (entire bordered strip), NOT one column width

For each plate id:
1. Count how many separate text lines are visible on that plate → set line_count.
2. Return exactly line_count entries in lines (same count).

Line counting rules:
- One printed title block = one line (e.g. "ROOM 31" → line_count 1, text="ROOM 31").
- A single printed phrase stays ONE line even if it contains spaces or numbers
  (e.g. "CIRCUITS 1 TO 72" = one line, not four) — but ONLY within a single
  physical line. This NEVER licenses merging text from different rows.
- Side-by-side words in different columns = separate lines (e.g. CALLING | FAULT | DEFROST → 3).
- Stacked rows in a column = separate lines (e.g. ROOM, DISABLE, ENABLE → 3).
- Every physically separate line is its OWN entry — including each line of a
  stacked multi-line block (an address, compliance panel, or notice). A block
  with N visible stacked lines = N line entries, NEVER one merged blob.
- Never merge words from different columns into one line text field, and never
  join stacked lines into a single entry.
- Transcribe EVERY visible line including faint, small, or handwritten secondary
  lines. A stacked warning plate (WARNING / TO BE ACCESSED BY / AUTHORISED
  PERSONNEL / ONLY) has 4 lines — never drop the hard-to-read ones.
- Keep punctuation exactly as printed; write slashed values with no added spaces
  (e.g. "MAX/250A", not "MAX / 250A").

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