"""Deterministic Markdown rendering of a processed spec (no LLM)."""

from datetime import datetime


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _line_row(line: dict) -> str:
    computed = set(line.get("computed_fields") or [])
    measured = set(line.get("measured_fields") or [])

    def cell(field: str) -> str:
        value = _fmt(line.get(field))
        if field in computed:
            return f"{value} *"
        if field in measured:
            return f"{value} ~"
        return value

    return (
        f"| {line.get('text', '')} | {cell('x_mm')} | {cell('y_mm')} "
        f"| {cell('size_mm')} | {cell('alignment')} | {_fmt(line.get('bold'))} |"
    )


def render_markdown(spec: dict, source_image: str) -> str:
    parts: list[str] = [
        "# Label Spec",
        "",
        f"- **Source image:** {source_image}",
        f"- **Generated at:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Unit:** {spec.get('unit', 'mm')}",
        f"- **Total labels:** {spec.get('total_labels', 0)}",
        "",
        "Value provenance: plain = annotated in the draft, `~` = measured "
        "from the drawing via pixel calibration, `*` = computed by the "
        "layout resolver — verify `*` values before manufacturing.",
        "",
    ]

    for label in spec.get("labels") or []:
        num = label.get("label_number", "?")
        w = _fmt(label.get("width_mm"))
        h = _fmt(label.get("height_mm"))
        parts.append(f"## Label {num} — {w} x {h} mm")
        parts.append("")

        meta = [
            ("Quantity", label.get("quantity")),
            ("Material", label.get("material")),
            ("Background", label.get("background_color")),
            ("Text colour", label.get("text_color")),
            ("Fixing", label.get("fixing")),
            ("Notes", label.get("notes")),
        ]
        for key, value in meta:
            if value is not None:
                parts.append(f"- **{key}:** {_fmt(value)}")
        parts.append("")

        lines = label.get("lines") or []
        if lines:
            parts.append(
                "| Text | Position from left (mm) | Position from top (mm) "
                "| Text size (mm) | Alignment | Bold |"
            )
            parts.append("|---|---|---|---|---|---|")
            parts.extend(_line_row(line) for line in lines)
            parts.append("")

        holes = label.get("holes") or []
        if holes:
            parts.append("| Hole diameter (mm) | x (mm) | y (mm) |")
            parts.append("|---|---|---|")
            for hole in holes:
                parts.append(
                    f"| {_fmt(hole.get('diameter_mm'))} | {_fmt(hole.get('x_mm'))} "
                    f"| {_fmt(hole.get('y_mm'))} |"
                )
            parts.append("")

    warnings = spec.get("warnings") or []
    if warnings:
        parts.append("## Warnings")
        parts.append("")
        parts.extend(f"- {w}" for w in warnings)
        parts.append("")

    return "\n".join(parts)
