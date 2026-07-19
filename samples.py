"""Sample images under sample-data/ for batch runs."""

from pathlib import Path

SAMPLE_DIR = Path("sample-data")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def sample_paths(root: Path | None = None) -> list[Path]:
    """All image files in sample-data/, sorted by name."""
    base = (root or Path.cwd()) / SAMPLE_DIR
    if not base.is_dir():
        return []
    return sorted(
        p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
