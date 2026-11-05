"""Pure parsing helpers for the F1 live-timing feed."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

DRS_OPEN_CODES = (10, 12, 14)
_num_re = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def to_float(v: Any) -> Optional[float]:
    """Lenient float conversion; None for junk, NaN, inf."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def to_int(v: Any) -> Optional[int]:
    f = to_float(v)
    return None if f is None else int(f)


def to_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def parse_time(s: Any) -> Optional[float]:
    """'1:32.123' -> 92.123 ; '28.1' -> 28.1 ; '1:02:03.5' -> 3723.5 ; junk/'' -> None."""
    if s is None or isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return to_float(s)
    s = str(s).strip()
    if not s:
        return None
    try:
        total = 0.0
        for part in s.split(":"):
            total = total * 60.0 + float(part)
    except ValueError:
        return None
    if math.isnan(total) or math.isinf(total):
        return None
    return total


def parse_gap(s: Any) -> Tuple[str, Optional[float]]:
    """Return (display text, seconds|None).

    '+1.234' -> ('+1.234', 1.234); '1L'/'LAP 2'/'2 LAPS' -> (text, None); ''/None -> ('', None).
    """
    if s is None:
        return "", None
    if isinstance(s, dict):
        s = s.get("Value")
        if s is None:
            return "", None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        f = to_float(s)
        return ("" if f is None else "+%.3f" % f), f
    txt = str(s).strip()
    if not txt:
        return "", None
    t = txt.lstrip("+")
    if _num_re.match(t):
        return txt, float(t)
    if ":" in t:  # '1:02.345' style gaps
        sec = parse_time(t)
        if sec is not None:
            return txt, sec
    return txt, None


def parse_iso(s: Any) -> Optional[float]:
    """ISO8601 ('2024-03-02T15:04:05.123Z') -> unix seconds."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return to_float(s)
    try:
        t = str(s).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def parse_clock(s: Any) -> Optional[float]:
    """'01:23:45' / '23:45' / '01:23:45.5' -> seconds."""
    return parse_time(s)


def is_drs_open(code: Any) -> bool:
    i = to_int(code)
    return i is not None and i in DRS_OPEN_CODES


def norm_brake(v: Any) -> int:
    """Brake comes as 0/100 or 0/1 (or bool): normalise to 0 or 100."""
    if isinstance(v, bool):
        return 100 if v else 0
    f = to_float(v)
    return 100 if (f is not None and f > 0) else 0


def as_list(v: Any) -> list:
    """Normalise a list-or-dict-keyed-by-index container to a list (ordered by numeric key)."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        def k(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, 0)
        return [val for _, val in sorted(v.items(), key=k)]
    return []


def indexed_items(v: Any):
    """Yield (int index, item) for list or dict-keyed-by-index containers."""
    if isinstance(v, list):
        for i, it in enumerate(v):
            yield i, it
    elif isinstance(v, dict):
        for k, it in v.items():
            try:
                yield int(k), it
            except (TypeError, ValueError):
                continue


def deep_merge(dst: dict, src: dict) -> dict:
    """In-place recursive dict merge (private copy; list values replace)."""
    for k, v in src.items():
        if isinstance(v, dict):
            cur = dst.get(k)
            if isinstance(cur, dict):
                deep_merge(cur, v)
            else:
                dst[k] = _copy(v)
        else:
            dst[k] = v
    return dst


def _copy(v):
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    return v
