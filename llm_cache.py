"""Per-run LLM response cache under results/<timestamp>/llm/."""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

_run_cache_dir: Path | None = None
_label_seq: dict[str, int] = {}
_label_seq_lock = threading.Lock()
CACHE_ENABLED = os.environ.get("LLM_CACHE", "1").lower() not in ("0", "false", "no")
SKIP_LABELS = {"warmup"}


def set_run_cache_dir(path: str | Path | None) -> None:
    """Point cache I/O at ``results/<ts>/llm/`` for the current run."""
    global _run_cache_dir, _label_seq
    with _label_seq_lock:
        _run_cache_dir = Path(path) if path else None
        _label_seq = {}


def _safe_stage_name(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower() or "chat"


def stage_cache_path(label: str) -> tuple[Path | None, bool]:
    """Return ``(path, may_load)`` for this call.

    First call for a label uses ``<stage>.json`` and may load a prior save.
    Retries (e.g. review re-running size) use ``<stage>-2.json``, … and always
    hit the API."""
    if not CACHE_ENABLED or _run_cache_dir is None or label in SKIP_LABELS:
        return None, False
    base = _safe_stage_name(label)
    with _label_seq_lock:
        seq = _label_seq.get(label, 0) + 1
        _label_seq[label] = seq
        cache_dir = _run_cache_dir
    name = f"{base}.json" if seq == 1 else f"{base}-{seq}.json"
    return cache_dir / name, seq == 1


def _content_to_str(content) -> str | None:
    """Normalize cached content back to the raw string ``call_chat`` expects."""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _content_from_str(raw: str):
    """Store LLM output as parsed JSON when possible, else plain text."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def load(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return _content_to_str(data.get("content"))


def save(path: Path, label: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": label,
        "content": _content_from_str(content),
    }
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")


def repair_cache_file(path: Path) -> bool:
    """Re-write a cache file whose ``content`` is a JSON string instead of an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    content = data.get("content")
    if not isinstance(content, str):
        return False
    parsed = _content_from_str(content)
    if isinstance(parsed, str):
        return False
    data["content"] = parsed
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True
