"""JSON Schema (draft-07) for label specs.

Used for vLLM ``response_format: json_schema`` (constrained decoding) and
optional local structural validation.
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
# Designer pipeline — one small schema per node
# ─────────────────────────────────────────────────────────────────────────

SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_type": {"enum": ["cad", "sketch", "photo", "other", None]},
        "material_notes": {"type": ["string", "null"]},
    },
    "required": ["draft_type", "material_notes"],
    "additionalProperties": False,
}

DIMENSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "dimension_annotations": {"type": "array", "items": DIMENSION_SCHEMA},
    },
    "required": ["dimension_annotations"],
    "additionalProperties": False,
}

PLATE_REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "bbox_px": BBOX_SCHEMA,
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
    },
    "required": ["id", "bbox_px", "width_mm", "height_mm"],
    "additionalProperties": False,
}

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "plates": {"type": "array", "items": PLATE_REGION_SCHEMA},
    },
    "required": ["plates"],
    "additionalProperties": False,
}

# Cheap coarse plate-count yardstick (integer only, no geometry). Reused by the
# Phase-2 gate and the Phase-3 QC arbiter — computed once per run.
PLATE_COUNT_SCHEMA = {
    "type": "object",
    "properties": {"plate_count": {"type": "integer"}},
    "required": ["plate_count"],
    "additionalProperties": False,
}

# LLM decompose fallback. Coordinates are FRACTIONS of the image in [0, 1] — the
# model never emits absolute pixels (that path hallucinated coords past the
# canvas). Code converts frac→px against the known image size and rejects any
# out-of-range box (see nodes/decompose._llm_decompose).
PLATE_REGION_FRAC_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "bbox_frac": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 4,
            "maxItems": 4,
        },
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
    },
    "required": ["id", "bbox_frac", "width_mm", "height_mm"],
    "additionalProperties": False,
}

DECOMPOSE_FRAC_SCHEMA = {
    "type": "object",
    "properties": {
        "plates": {"type": "array", "items": PLATE_REGION_FRAC_SCHEMA},
    },
    "required": ["plates"],
    "additionalProperties": False,
}

PLATE_DIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "plates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "width_mm": {"type": ["number", "null"]},
                    "height_mm": {"type": ["number", "null"]},
                },
                "required": ["id", "width_mm", "height_mm"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["plates"],
    "additionalProperties": False,
}

TRANSCRIBE_LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "bbox_px": BBOX_SCHEMA,
    },
    "required": ["text", "bbox_px"],
    "additionalProperties": False,
}

TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": TRANSCRIBE_LINE_SCHEMA},
    },
    "required": ["lines"],
    "additionalProperties": False,
}

SHEET_QC_SCHEMA = {
    "type": "object",
    "properties": {
        "plate_count_ok": {"type": "boolean"},
        "decompose_ok": {"type": "boolean"},
        "verdict": {"enum": ["pass", "revise"]},
        "fix": {"enum": ["decompose", "position", "size", None]},
        "plate_ids": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
        },
        "notes": {"type": ["string", "null"]},
    },
    "required": [
        "plate_count_ok", "decompose_ok", "verdict", "fix", "plate_ids", "notes",
    ],
    "additionalProperties": False,
}

# ─────────────────────────────────────────────────────────────────────────
# Per-plate geometry nodes
# ─────────────────────────────────────────────────────────────────────────

# size: capital-letter height per text, relative to the plate.
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


def validate_against_schema(data: dict, schema: dict = LABEL_SPEC_SCHEMA) -> list[str]:
    """Return a list of structural errors (empty = valid)."""
    errors: list[str] = []
    _check(data, schema, "$", errors)
    return errors
