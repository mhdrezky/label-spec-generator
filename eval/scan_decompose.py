"""Phase 4 — cross-image decompose proxy scan (breadth without full GT).

    python eval/scan_decompose.py [--no-llm]

For every image in eval/images/, run CV detection + the cheap llm_count yardstick
+ the gate, and print how decompose would route (CV vs LLM fallback). This is a
smoke test, not a score: it answers "is decompose alive or dead per image?"
without hand-labelled geometry. Rows are flagged when CV looks dead
(cv_count in {0, 1}) or CV and llm_count grossly disagree.

llm_count calls are cached under eval/predictions/llm/_proxy so reruns are cheap.
Pass --no-llm to skip the vision calls and show CV-only counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_cache import set_run_cache_dir  # noqa: E402
from nodes.decompose import PLATE_COUNT_PROMPT  # noqa: E402  (shared, no drift)
from plate_detect import detect_plates, file_image_px  # noqa: E402
from plate_gate import evaluate_gate  # noqa: E402
from schema import PLATE_COUNT_SCHEMA  # noqa: E402
from vision import call_vision, guess_mime, image_to_base64  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _llm_count(image_path: Path) -> int | None:
    result = call_vision(
        PLATE_COUNT_PROMPT,
        [(image_to_base64(image_path), guess_mime(image_path))],
        schema=PLATE_COUNT_SCHEMA,
        label=f"proxy-count-{image_path.stem}",
        max_tokens=200,
    )
    count = result.get("plate_count")
    return count if isinstance(count, int) else None


def main() -> None:
    use_llm = "--no-llm" not in sys.argv
    if use_llm:
        set_run_cache_dir(EVAL_DIR / "predictions" / "llm" / "_proxy")

    images = sorted(
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    header = (
        f"{'image':<44} {'WxH':<11} {'cv':>4} {'llm':>4} {'gap':>5} "
        f"{'cov':>5} {'route':<12} flag"
    )
    print(header)
    print("-" * len(header))

    dead = 0
    fallback = 0
    for image in images:
        px = file_image_px(image)
        plates, _ = detect_plates(image)
        cv = len(plates)
        llm = _llm_count(image) if use_llm else None
        gate = evaluate_gate(plates, px, llm)
        route = "CV" if (plates and gate["trust_cv"]) else "LLM-fallback"
        if route == "LLM-fallback":
            fallback += 1
        flag = ""
        if cv in (0, 1):
            flag = "CV-DEAD"
            dead += 1
        elif gate["count_gap"] is not None and gate["count_gap"] > 0.6:
            flag = "GROSS-GAP"
        gap = "" if gate["count_gap"] is None else f"{gate['count_gap']:.0%}"
        wh = f"{px['width']}x{px['height']}"
        llm_str = str(llm) if llm is not None else "-"
        print(
            f"{image.name:<44} {wh:<11} {cv:>4} {llm_str:>4} {gap:>5} "
            f"{gate['coverage']:>5} {route:<12} {flag}"
        )

    print(
        f"\n{len(images)} images: {fallback} route to LLM fallback, "
        f"{dead} CV-dead (cv<=1)."
    )


if __name__ == "__main__":
    main()
