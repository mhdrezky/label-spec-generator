"""Shared helpers for vision calls and image cropping."""

import base64
import io
from pathlib import Path

from PIL import Image

from api_client import call_chat, parse_json_response


def image_to_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def guess_mime(path: str | Path) -> str:
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(Path(path).suffix.lower(), "image/png")


def _image_block(b64: str, mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def call_vision(
    prompt: str,
    images: list[tuple[str, str]],
    *,
    schema: dict,
    label: str,
    max_tokens: int = 8000,
) -> dict:
    """One schema-constrained vision call.

    ``images`` is a list of (base64, mime) blocks, shown after the prompt in
    order. Returns the parsed JSON (or {"error": "parse_failed", ...})."""
    content = [{"type": "text", "text": prompt}]
    content += [_image_block(b64, mime) for b64, mime in images]
    raw = call_chat(
        [{"role": "user", "content": content}],
        label=label,
        max_tokens=max_tokens,
        json_schema=schema,
    )
    return parse_json_response(raw)


def rescale_bbox(bbox: list, image: Image.Image, stated_px: dict | None) -> list:
    """Map a model-space bbox (in the image_px the model reported) to the
    file's real pixels, so crops line up with the actual image."""
    fx = fy = 1.0
    if (
        isinstance(stated_px, dict)
        and isinstance(stated_px.get("width"), (int, float))
        and isinstance(stated_px.get("height"), (int, float))
        and stated_px["width"] > 0
        and stated_px["height"] > 0
    ):
        fx = image.width / stated_px["width"]
        fy = image.height / stated_px["height"]
    return [bbox[0] * fx, bbox[1] * fy, bbox[2] * fx, bbox[3] * fy]


def crop_plate(
    image: Image.Image, bbox: list, stated_px: dict | None, pad_fraction: float = 0.06
) -> Image.Image:
    x1, y1, x2, y2 = rescale_bbox(bbox, image, stated_px)
    pad_x = (x2 - x1) * pad_fraction
    pad_y = (y2 - y1) * pad_fraction
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(image.width, int(x2 + pad_x))
    bottom = min(image.height, int(y2 + pad_y))
    return image.crop((left, top, right, bottom))
