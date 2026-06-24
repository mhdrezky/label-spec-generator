import json
import sys
from datetime import datetime
from pathlib import Path

from api_client import (
    API_READ_TIMEOUT,
    API_SPECS_MAX_TOKENS,
    API_URL,
    MODEL,
    API_ENABLE_THINKING,
    call_chat,
    check_api_health,
    parse_json_response,
    warmup_model,
)

RESULT_DIR = "results"
OUTPUT_MD = "output.md"
SPECS_FILENAME = "specs.json"

OBJECT_REQUIRED_FIELDS = ("text", "x_mm", "y_mm", "alignment", "text_size_mm")
VALID_ALIGNMENTS = frozenset({"left", "center", "right"})
SUBLABEL_TEXTS = frozenset({"DISABLE", "ENABLE", "LOW", "HIGH"})
PAIR_OFFSET_MM = 13


def zone_guide_line(zone: dict) -> float | None:
    if "guide_line_mm" in zone:
        return float(zone["guide_line_mm"])
    return None


def build_specs_prompt(markdown: str) -> str:
    return f"""
You are given a label layout analysis in Markdown format.
Convert it into manufacturing label specifications as JSON only. No markdown, no explanation.

## Input analysis

{markdown}

## Output rules

Read **Plate Inventory**, **Section Definitions**, **Grid Specifications**, and **Layout Preview** carefully.

### Plate counting logic
- `plate_divider` section: each cell in a divided row = one separate label plate
- `layout_guide` section: each horizontal row = one label plate; columns are layout guides, NOT separate plates
- NEVER merge plates from different sections into one label
- NEVER treat vertical columns as one tall plate spanning multiple sections
- Total labels in output MUST match **Total_Physical_Plates** from the analysis

### Per-label dimensions
- `label_length_mm` and `label_height_mm` = that plate's **Plate_Media_Width_mm** and **Plate_Media_Height_mm** from its section
- For layout_guide plates with internal segments (e.g. 25, 100, 100, 25): use the plate media width from the section, NOT the full canvas width unless the section explicitly states the plate is canvas-wide
- Do NOT use full canvas dimensions as label size when the section defines per-plate media size

### Schema: 3 layers per label

Each label uses `zones[]`, `rows[]`, and `objects[]`.

#### Layer 1: zones[] — engineering segments
- Build from **Horizontal_Segments** and **Layout_Guides** in the analysis for that plate's section
- For segment pattern 25, 100, 100, remainder (e.g. 25/100/100/35 on a 260mm plate):
  - `margin_left` (0–25): margin only — no column text
  - `zone_left` (25–125): left column — add `"guide_line_mm": 25`
  - `zone_center` (125–225): center column — add `"guide_line_mm": 125`
  - `zone_right` (225 to plate width): right column — add `"guide_line_mm": 225`
- `guide_line_mm` = vertical **layout guide / center line** position from the drawing — NOT the geometric midpoint `(start+end)/2`
- For `plate_divider` with a single centered text: `zones` may be `[]`

#### Layer 2: rows[] — semantic visual rows
- One entry per visual text row on the plate (count varies per plate — do NOT assume fixed row counts)
- Each row: `{{"row": 1, "y_mm": 2}}` where `y_mm` is distance from top edge of plate to that row's text baseline area
- For `plate_divider` with one text line: `rows` may be `[]`

#### Layer 3: objects[] — render-ready text elements
- ONE object per text string — never flatten multiple texts into a sequential numbered list
- Each object MUST have: `text`, `x_mm`, `y_mm`, `alignment`, `text_size_mm`
- Optional traceability: `row`, `zone` (reference to zones[].id and rows[].row)
- `x_mm`: horizontal anchor from left edge of THIS plate (mm)
- Column headers (ROOM, FANS, SUCTION, CALLING, FAULT, DEFROST): `alignment: "center"`, `x_mm` = that column's `guide_line_mm` (25, 125, or 225)
- Paired sub-labels (DISABLE/ENABLE, LOW/HIGH): offset ±13mm from the same column `guide_line_mm` — DISABLE/LOW use `alignment: "right"` at guide−13; ENABLE/HIGH use `alignment: "left"` at guide+13
- Do NOT use zone geometric center (75, 175, 242.5) — those are wrong for this layout
- `text_size_mm`: estimate from plate height — for 20mm strips with 2 text rows use ~6 for headers and ~4 for sub-labels; for 40mm single-line plates use ~12
- `y_mm`: vertical position from top edge of THIS plate (mm)
- `alignment`: `"left"` | `"center"` | `"right"` — renderer must not guess from x alone
- For paired sub-labels (DISABLE / ENABLE): create TWO separate objects with different alignment (e.g. right + left), NOT a texts array
- For single centered labels (ROOM 31): one object with alignment `"center"`

### grid_type rules
- `plate_divider` → required `objects[]`; `zones` and `rows` may be `[]`
- `layout_guide` → required `zones[]` (when segments exist in analysis), `rows[]` with at least 1 row, required `objects[]`

### Other fields
- Fields not visible in the analysis → null (not guesses)
- `no_of_holes`: 0 if no holes mentioned
- `label_quantity`: 1 per plate unless stated otherwise
- `label_number`: sequential plate # from Plate Inventory

Return this JSON schema exactly:

{{
  "unit": "mm",
  "total_labels": 0,
  "labels": [
    {{
      "label_number": 1,
      "panel_id": "",
      "section_id": "",
      "grid_type": "plate_divider",
      "label_length_mm": 0,
      "label_height_mm": 0,
      "plate_thickness_mm": null,
      "label_quantity": 1,
      "no_of_holes": 0,
      "hole_size_mm": null,
      "hole_distance_mm": null,
      "zones": [],
      "rows": [],
      "objects": [
        {{
          "text": "ROOM 31",
          "x_mm": 32.5,
          "y_mm": 10,
          "alignment": "center",
          "text_size_mm": 12
        }}
      ]
    }},
    {{
      "label_number": 5,
      "panel_id": "",
      "section_id": "",
      "grid_type": "layout_guide",
      "label_length_mm": 260,
      "label_height_mm": 20,
      "plate_thickness_mm": null,
      "label_quantity": 1,
      "no_of_holes": 0,
      "hole_size_mm": null,
      "hole_distance_mm": null,
      "zones": [
        {{"id": "margin_left", "start_mm": 0, "end_mm": 25}},
        {{"id": "zone_left", "start_mm": 25, "end_mm": 125, "guide_line_mm": 25}},
        {{"id": "zone_center", "start_mm": 125, "end_mm": 225, "guide_line_mm": 125}},
        {{"id": "zone_right", "start_mm": 225, "end_mm": 260, "guide_line_mm": 225}}
      ],
      "rows": [
        {{"row": 1, "y_mm": 2}},
        {{"row": 2, "y_mm": 12}}
      ],
      "objects": [
        {{"text": "ROOM", "row": 1, "zone": "zone_left", "x_mm": 25, "y_mm": 4, "alignment": "center", "text_size_mm": 6}},
        {{"text": "FANS", "row": 1, "zone": "zone_center", "x_mm": 125, "y_mm": 4, "alignment": "center", "text_size_mm": 6}},
        {{"text": "SUCTION", "row": 1, "zone": "zone_right", "x_mm": 225, "y_mm": 4, "alignment": "center", "text_size_mm": 6}},
        {{"text": "DISABLE", "row": 2, "zone": "zone_left", "x_mm": 12, "y_mm": 14, "alignment": "right", "text_size_mm": 4}},
        {{"text": "ENABLE", "row": 2, "zone": "zone_left", "x_mm": 38, "y_mm": 14, "alignment": "left", "text_size_mm": 4}},
        {{"text": "LOW", "row": 2, "zone": "zone_center", "x_mm": 112, "y_mm": 14, "alignment": "right", "text_size_mm": 4}},
        {{"text": "HIGH", "row": 2, "zone": "zone_center", "x_mm": 138, "y_mm": 14, "alignment": "left", "text_size_mm": 4}},
        {{"text": "LOW", "row": 2, "zone": "zone_right", "x_mm": 212, "y_mm": 14, "alignment": "right", "text_size_mm": 4}},
        {{"text": "HIGH", "row": 2, "zone": "zone_right", "x_mm": 238, "y_mm": 14, "alignment": "left", "text_size_mm": 4}}
      ]
    }}
  ]
}}

The example above is illustrative — output ALL plates from Plate Inventory with complete objects for every text element.

Return valid JSON only.
"""


def validate_specs(parsed: dict) -> list[str]:
    warnings: list[str] = []
    labels = parsed.get("labels", [])
    total_labels = parsed.get("total_labels")

    if total_labels is not None and isinstance(labels, list) and total_labels != len(labels):
        warnings.append(
            f"total_labels ({total_labels}) does not match len(labels) ({len(labels)})"
        )

    if not isinstance(labels, list):
        warnings.append("labels is not a list")
        return warnings

    for label in labels:
        if not isinstance(label, dict):
            warnings.append("label entry is not an object")
            continue

        num = label.get("label_number", "?")
        grid_type = label.get("grid_type")
        zones = label.get("zones", [])
        rows = label.get("rows", [])
        objects = label.get("objects", [])
        zone_by_id: dict[str, dict] = {}

        if grid_type == "plate_divider":
            if not objects:
                warnings.append(f"label #{num}: plate_divider has no objects")
            if zones:
                warnings.append(f"label #{num}: plate_divider has non-empty zones[]")
            if rows:
                warnings.append(f"label #{num}: plate_divider has non-empty rows[]")

        elif grid_type == "layout_guide":
            if not objects:
                warnings.append(f"label #{num}: layout_guide has no objects")
            if not isinstance(rows, list) or len(rows) < 1:
                warnings.append(f"label #{num}: layout_guide must have at least 1 row")
            if not zones:
                warnings.append(f"label #{num}: layout_guide has empty zones[]")

            zone_by_id = {
                z["id"]: z for z in zones if isinstance(z, dict) and "id" in z
            }

        elif grid_type:
            warnings.append(f"label #{num}: unknown grid_type '{grid_type}'")
        else:
            warnings.append(f"label #{num}: missing grid_type")

        if not isinstance(objects, list):
            warnings.append(f"label #{num}: objects is not a list")
            continue

        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                warnings.append(f"label #{num}: objects[{i}] is not an object")
                continue
            for field in OBJECT_REQUIRED_FIELDS:
                if field not in obj:
                    warnings.append(f"label #{num}: objects[{i}] missing '{field}'")
            alignment = obj.get("alignment")
            if alignment is not None and alignment not in VALID_ALIGNMENTS:
                warnings.append(
                    f"label #{num}: objects[{i}] invalid alignment '{alignment}'"
                )

            if grid_type == "layout_guide":
                text = obj.get("text", "")
                zone_id = obj.get("zone")
                guide = zone_guide_line(zone_by_id[zone_id]) if zone_id in zone_by_id else None
                if (
                    guide is not None
                    and isinstance(obj.get("x_mm"), (int, float))
                ):
                    x = float(obj["x_mm"])
                    if alignment == "center":
                        expected = guide
                    elif text in SUBLABEL_TEXTS and alignment == "right":
                        expected = guide - PAIR_OFFSET_MM
                    elif text in SUBLABEL_TEXTS and alignment == "left":
                        expected = guide + PAIR_OFFSET_MM
                    else:
                        expected = None
                    if expected is not None and abs(x - expected) > 0.1:
                        warnings.append(
                            f"label #{num}: objects[{i}] '{text}' x_mm={x} "
                            f"!= expected {expected} (guide_line {guide})"
                        )
                elif (
                    alignment == "center"
                    and zone_id
                    and zone_id in zone_by_id
                    and zone_guide_line(zone_by_id[zone_id]) is None
                ):
                    warnings.append(
                        f"label #{num}: zone '{zone_id}' missing guide_line_mm"
                    )

    return warnings


def resolve_input_path(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if path.is_dir():
            path = path / OUTPUT_MD
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path

    results = Path(RESULT_DIR)
    if not results.is_dir():
        print(f"Error: no {RESULT_DIR}/ directory found.", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(results.iterdir(), reverse=True)
    for folder in candidates:
        if folder.is_dir():
            md_path = folder / OUTPUT_MD
            if md_path.exists():
                return md_path

    print(f"Error: no {OUTPUT_MD} found in {RESULT_DIR}/", file=sys.stderr)
    sys.exit(1)


def read_markdown_body(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    if "---" in content:
        parts = content.split("---", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return content.strip()


def save_specs(output_path: Path, data: dict, source_md: str) -> None:
    output = {
        "source_md": source_md,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def format_label_summary(label: dict) -> str:
    num = label.get("label_number", "?")
    panel_id = label.get("panel_id", "")
    w = label.get("label_length_mm", "?")
    h = label.get("label_height_mm", "?")
    zone_count = len(label.get("zones", []))
    row_count = len(label.get("rows", []))
    object_count = len(label.get("objects", []))

    name = f" {panel_id}" if panel_id else ""
    parts = [f"#{num}{name} ({w}x{h}mm):"]

    if label.get("grid_type") == "layout_guide":
        parts.append(f"{zone_count} zones, {row_count} rows, {object_count} objects")
    else:
        parts.append(f"{object_count} object{'s' if object_count != 1 else ''}")

    return " ".join(parts)


def generate_specs(md_path: Path) -> Path:
    markdown = read_markdown_body(md_path)
    prompt = build_specs_prompt(markdown)

    content = call_chat(
        [{"role": "user", "content": prompt}],
        label="specs",
        max_tokens=API_SPECS_MAX_TOKENS,
    )

    parsed = parse_json_response(content)
    output_path = md_path.parent / SPECS_FILENAME

    if parsed.get("error") == "parse_failed":
        save_specs(
            output_path,
            {"error": "parse_failed", "raw_response": parsed.get("raw_response", content)},
            str(md_path),
        )
        print(f"Warning: could not parse JSON. Raw response saved to {output_path}", file=sys.stderr)
        sys.exit(1)

    labels = parsed.get("labels", [])
    if not isinstance(labels, list) or not labels:
        save_specs(
            output_path,
            {"error": "empty_labels", "raw_response": content},
            str(md_path),
        )
        print("Warning: no labels in response.", file=sys.stderr)
        sys.exit(1)

    for warning in validate_specs(parsed):
        print(f"Validation warning: {warning}", file=sys.stderr)

    save_specs(output_path, parsed, str(md_path))
    return output_path


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    md_path = resolve_input_path(arg)

    print(f"Generating specs from {md_path}...")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print(
        f"Timeout: read={API_READ_TIMEOUT}s, "
        f"specs_max_tokens={API_SPECS_MAX_TOKENS}, "
        f"thinking={'on' if API_ENABLE_THINKING else 'off'}"
    )

    if not check_api_health():
        sys.exit(1)

    if not warmup_model():
        sys.exit(1)

    try:
        output_path = generate_specs(md_path)
    except Exception as exc:
        print(f"Specs generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = data.get("labels", [])
    total = data.get("total_labels", len(labels))
    print(f"Specs saved to {output_path}")
    print(f"Total labels: {total}")
    for label in labels:
        print(f"  - {format_label_summary(label)}")


if __name__ == "__main__":
    main()
