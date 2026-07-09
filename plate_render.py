"""Render one plate spec to a PNG (PIL) for the review stage.

The reviewer compares this render against the original plate crop — like a
designer holding their mock-up next to the brief. It is approximate by
design (layout, not exact typography): text is drawn at its center anchor
with a cap height matching size_mm, positions from x_mm / y_mm.
"""

from PIL import Image, ImageDraw, ImageFont

# cap height / font em-box, so requested size_mm maps to visible letter height
CAP_RATIO = 0.7
_FONT_CANDIDATES = [
    "arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
    "calibri.ttf", "segoeui.ttf",
]


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    px = max(6, int(px))
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


MARGIN_FRACTION = 0.18  # grey margin so text overflowing the plate stays visible


def render_plate(plate: dict, out_w: int, out_h: int) -> Image.Image:
    """Render at ~(out_w, out_h) px — pass the crop's pixel size so the review
    model sees two images of the same scale. The plate sits inside a grey
    margin so text that spills past the plate edge is shown (not clipped),
    letting the reviewer judge 'too big' instead of 'text missing'."""
    width_mm = plate.get("width_mm") or 100
    height_mm = plate.get("height_mm") or 40
    sx = out_w / width_mm
    sy = out_h / height_mm

    mx = int(out_w * MARGIN_FRACTION)
    my = int(out_h * MARGIN_FRACTION)
    canvas_w = out_w + 2 * mx
    canvas_h = out_h + 2 * my

    img = Image.new("RGB", (max(1, canvas_w), max(1, canvas_h)), "#8a8a8a")
    draw = ImageDraw.Draw(img)
    # plate itself, inset by the margin
    draw.rectangle([mx, my, mx + out_w - 1, my + out_h - 1], fill="white",
                   outline="black", width=2)

    for line in plate.get("lines") or plate.get("texts") or []:
        text = line.get("text", "")
        x = line.get("x_mm")
        y = line.get("y_mm")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        size_mm = line.get("size_mm") or (height_mm * 0.4)
        font_px = size_mm * sy / CAP_RATIO
        draw.text(
            (mx + x * sx, my + y * sy), text, fill="black",
            font=_load_font(font_px), anchor="mm",
        )
    return img
