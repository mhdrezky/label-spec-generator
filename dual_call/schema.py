"""JSON schemas for the 2-call extract path (fractional bboxes)."""

BBOX_FRAC_SCHEMA = {
    "type": "array",
    "items": {"type": "number", "minimum": 0, "maximum": 1},
    "minItems": 4,
    "maxItems": 4,
}

SPAN_FRAC_SCHEMA = {
    "type": "array",
    "items": {"type": "number", "minimum": 0, "maximum": 1},
    "minItems": 2,
    "maxItems": 2,
}

LINE_SIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "line": {"type": "integer"},
        "size_mm": {"type": "number"},
    },
    "required": ["line", "size_mm"],
    "additionalProperties": False,
}

STRUCTURE_PLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "bbox_frac": BBOX_FRAC_SCHEMA,
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
    },
    "required": ["id", "bbox_frac", "width_mm", "height_mm"],
    "additionalProperties": False,
}

DIMENSION_FRAC_SCHEMA = {
    "type": "object",
    "properties": {
        "value_mm": {"type": "number"},
        "axis": {"enum": ["horizontal", "vertical"]},
        "span_frac": SPAN_FRAC_SCHEMA,
    },
    "required": ["value_mm", "axis", "span_frac"],
    "additionalProperties": False,
}

STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_type": {"enum": ["cad", "sketch", "photo", "other", None]},
        "material": {"type": ["string", "null"]},
        "background_color": {"type": ["string", "null"]},
        "text_color": {"type": ["string", "null"]},
        "fixing": {"type": ["string", "null"]},
        "default_size_mm": {"type": ["number", "null"]},
        "line_sizes": {
            "type": "array",
            "items": LINE_SIZE_SCHEMA,
        },
        "dimension_annotations": {
            "type": "array",
            "items": DIMENSION_FRAC_SCHEMA,
        },
        "plates": {
            "type": "array",
            "items": STRUCTURE_PLATE_SCHEMA,
        },
    },
    "required": [
        "draft_type",
        "material",
        "background_color",
        "text_color",
        "fixing",
        "default_size_mm",
        "line_sizes",
        "dimension_annotations",
        "plates",
    ],
    "additionalProperties": False,
}

CONTENT_LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "x_mm": {"type": ["number", "null"]},
        "y_mm": {"type": ["number", "null"]},
        "size_mm": {"type": ["number", "null"]},
        "bbox_frac": BBOX_FRAC_SCHEMA,
    },
    "required": ["text", "x_mm", "y_mm", "size_mm", "bbox_frac"],
    "additionalProperties": False,
}

CONTENT_PLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "line_count": {"type": "integer"},
        "width_mm": {"type": ["number", "null"]},
        "height_mm": {"type": ["number", "null"]},
        "lines": {
            "type": "array",
            "items": CONTENT_LINE_SCHEMA,
        },
    },
    "required": ["id", "line_count", "width_mm", "height_mm", "lines"],
    "additionalProperties": False,
}

CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "plates": {
            "type": "array",
            "items": CONTENT_PLATE_SCHEMA,
        },
    },
    "required": ["plates"],
    "additionalProperties": False,
}
