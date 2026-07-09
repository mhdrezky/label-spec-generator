"""JSON Schema (draft-07) for label specs.

Used twice:
1. Sent to vLLM as ``response_format: json_schema`` (constrained decoding).
2. Local structural validation of the parsed response.

Principle: every numeric mm field is nullable at extraction time. ``null``
means "not annotated in the drawing" — never a guess. The model additionally
returns pixel bounding boxes (``bbox_px``, [x1, y1, x2, y2] in image pixels)
for every plate and text line; calibrate.py converts those to mm using
annotated dimensions as the scale reference, and postprocess.py fills any
remaining gaps deterministically.
"""

BBOX_SCHEMA = {
    "type": ["array", "null"],
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
}

LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "x_mm": {"type": ["number", "null"]},
        "y_mm": {"type": ["number", "null"]},
        "size_mm": {"type": ["number", "null"]},
        "alignment": {"enum": ["left", "center", "right", None]},
        "bold": {"type": ["boolean", "null"]},
        "bbox_px": BBOX_SCHEMA,
    },
    "required": ["text", "x_mm", "y_mm", "size_mm", "alignment", "bold", "bbox_px"],
    "additionalProperties": False,
}

HOLE_SCHEMA = {
    "type": "object",
    "properties": {
        "diameter_mm": {"type": ["number", "null"]},
        "x_mm": {"type": ["number", "null"]},
        "y_mm": {"type": ["number", "null"]},
    },
    "required": ["diameter_mm", "x_mm", "y_mm"],
    "additionalProperties": False,
}

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label_number": {"type": "integer"},
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
        "quantity": {"type": ["integer", "null"]},
        "material": {"type": ["string", "null"]},
        "background_color": {"type": ["string", "null"]},
        "text_color": {"type": ["string", "null"]},
        "fixing": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
        "bbox_px": BBOX_SCHEMA,
        "lines": {"type": "array", "items": LINE_SCHEMA},
        "holes": {"type": "array", "items": HOLE_SCHEMA},
    },
    "required": [
        "label_number",
        "width_mm",
        "height_mm",
        "quantity",
        "material",
        "background_color",
        "text_color",
        "fixing",
        "notes",
        "bbox_px",
        "lines",
        "holes",
    ],
    "additionalProperties": False,
}

IMAGE_PX_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "width": {"type": "number"},
        "height": {"type": "number"},
    },
    "required": ["width", "height"],
    "additionalProperties": False,
}

# A dimension line / arrow in the drawing: its written value and the pixel
# range it MEASURES (between its two extension lines), along its axis.
# This is the ground-truth scale anchor — far more reliable than the model's
# own guess at a plate's width_mm.
DIMENSION_SCHEMA = {
    "type": "object",
    "properties": {
        "value_mm": {"type": "number"},
        "axis": {"enum": ["horizontal", "vertical"]},
        "span_px": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "required": ["value_mm", "axis", "span_px"],
    "additionalProperties": False,
}

LABEL_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string"},
        "image_px": IMAGE_PX_SCHEMA,
        "dimension_annotations": {"type": "array", "items": DIMENSION_SCHEMA},
        "labels": {"type": "array", "items": LABEL_SCHEMA},
    },
    "required": ["unit", "image_px", "dimension_annotations", "labels"],
    "additionalProperties": False,
}

# ─────────────────────────────────────────────────────────────────────────
# Layered pipeline (layered.py) — each stage is its own vision call.
# Designers work in passes: decide the plates, then sizes, then spacing, then
# review. Each stage has a tiny schema so the model answers one question.
# ─────────────────────────────────────────────────────────────────────────

# Stage 1 — detect: plates + their text content. NO layout math here.
DETECT_TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "line": {"type": "integer"},          # stable id, top-to-bottom
        "text": {"type": "string"},
    },
    "required": ["line", "text"],
    "additionalProperties": False,
}

DETECT_PLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "plate_id": {"type": "integer"},
        "bbox_px": BBOX_SCHEMA,
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
        "quantity": {"type": ["integer", "null"]},
        "material": {"type": ["string", "null"]},
        "background_color": {"type": ["string", "null"]},
        "text_color": {"type": ["string", "null"]},
        "fixing": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
        "texts": {"type": "array", "items": DETECT_TEXT_SCHEMA},
    },
    "required": [
        "plate_id", "bbox_px", "width_mm", "height_mm", "quantity",
        "material", "background_color", "text_color", "fixing", "notes", "texts",
    ],
    "additionalProperties": False,
}

DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string"},
        "image_px": IMAGE_PX_SCHEMA,
        "plates": {"type": "array", "items": DETECT_PLATE_SCHEMA},
    },
    "required": ["unit", "image_px", "plates"],
    "additionalProperties": False,
}

# Stage 1b — re-detect ONE plate from its crop (dims + texts only).
REDETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
        "texts": {"type": "array", "items": DETECT_TEXT_SCHEMA},
    },
    "required": ["width_mm", "height_mm", "texts"],
    "additionalProperties": False,
}

# Stage 2 — size: capital-letter height per text, relative to the plate.
SIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "text": {"type": "string"},
                    "size_mm": {"type": "number"},
                },
                "required": ["line", "text", "size_mm"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}

# Stage 3 — position: center of each text as distance from left / top edge.
POSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "text": {"type": "string"},
                    "x_mm": {"type": "number"},
                    "y_mm": {"type": "number"},
                },
                "required": ["line", "text", "x_mm", "y_mm"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}

# Stage 4 — review: checklist comparing a render of the spec against the crop.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "plate_size_ok": {"type": "boolean"},
        "text_count_ok": {"type": "boolean"},
        "text_size_ok": {"type": "boolean"},
        "spacing_ok": {"type": "boolean"},
        "verdict": {"enum": ["pass", "revise"]},
        "fix": {"enum": ["detect", "size", "position", None]},
        "notes": {"type": ["string", "null"]},
    },
    "required": [
        "plate_size_ok", "text_count_ok", "text_size_ok", "spacing_ok",
        "verdict", "fix", "notes",
    ],
    "additionalProperties": False,
}

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _matches_type(value, type_name: str) -> bool:
    if not isinstance(value, _TYPE_MAP[type_name]):
        return False
    if type_name in ("integer", "number") and isinstance(value, bool):
        return False
    return True


def _check(value, schema: dict, path: str, errors: list[str]) -> None:
    if "enum" in schema:
        if value not in schema["enum"]:
            errors.append(f"{path}: {value!r} not in {schema['enum']}")
        return

    allowed = schema.get("type")
    if allowed is not None:
        types = allowed if isinstance(allowed, list) else [allowed]
        if not any(_matches_type(value, t) for t in types):
            errors.append(f"{path}: expected {allowed}, got {type(value).__name__}")
            return

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field '{key}'")
        props = schema.get("properties", {})
        for key, sub in value.items():
            if key in props:
                _check(sub, props[key], f"{path}.{key}", errors)
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{i}]", errors)


def validate_against_schema(data: dict) -> list[str]:
    """Return a list of structural errors (empty = valid)."""
    errors: list[str] = []
    _check(data, LABEL_SPEC_SCHEMA, "$", errors)
    return errors
