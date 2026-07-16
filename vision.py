"""Shared helpers for vision calls."""

import base64
from pathlib import Path

from api_client import call_chat, parse_json_response


def image_to_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def guess_mime(path: str | Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
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
    return_meta: bool = False,
) -> dict | tuple[dict, dict]:
    """One schema-constrained vision call.

    ``images`` is a list of (base64, mime) blocks, shown after the prompt in
    order. Returns the parsed JSON (or {"error": "parse_failed", ...})."""
    content = [{"type": "text", "text": prompt}]
    content += [_image_block(b64, mime) for b64, mime in images]
    raw, meta = call_chat(
        [{"role": "user", "content": content}],
        label=label,
        max_tokens=max_tokens,
        json_schema=schema,
    )
    result = parse_json_response(raw)

    if (
        result.get("error") == "parse_failed"
        and meta.get("finish_reason") == "length"
        and max_tokens < 16000
    ):
        print(f"{label}: retrying with max_tokens=16000 after truncation...")
        raw, meta = call_chat(
            [{"role": "user", "content": content}],
            label=label,
            max_tokens=16000,
            json_schema=schema,
        )
        result = parse_json_response(raw)

    if return_meta:
        return result, meta
    return result
