from __future__ import annotations

from datetime import datetime


def progress_line(stage: str, current: int, total: int, *, detail: str = "") -> str:
    total = max(int(total), 0)
    current = max(int(current), 0)
    if total:
        percent = min(current / total * 100, 100.0)
        prefix = f"{current}/{total} ({percent:.1f}%)"
    else:
        prefix = f"{current}/0"
    timestamp = datetime.now().isoformat(timespec="seconds")
    suffix = f" {detail}" if detail else ""
    return f"[{timestamp}] {stage} {prefix}{suffix}"


def print_progress(stage: str, current: int, total: int, *, detail: str = "") -> None:
    print(progress_line(stage, current, total, detail=detail), flush=True)
