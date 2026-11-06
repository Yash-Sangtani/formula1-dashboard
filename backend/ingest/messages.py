"""Raw F1 live-timing message model and line parsing.

A *recording* (what ``fastf1.livetiming.client.SignalRClient`` writes) is a text file with one
python-repr list per line::

    ['TimingData', {...}, '2024-03-02T15:04:05.123Z']
    ['CarData.z', '<base64(raw-deflate(json))>', '2024-03-02T15:04:05.123Z']

The first lines of a recording are usually a ``CompletionMessage`` dump of the form
``['Topic', '<json string>', '']`` (full snapshots with no timestamp).

``parse_line`` turns any of those (and JSON-lines ``{"topic":..,"data":..,"utc":..}``) into a
:class:`Msg`.  ``.z`` topics are decoded and the suffix is stripped, so the engine only ever sees
``CarData`` / ``Position`` with plain dict payloads.
"""
from __future__ import annotations

import ast
import base64
import calendar
import json
import re
import time
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional

__all__ = [
    "Msg",
    "parse_line",
    "parse_ts",
    "format_ts",
    "decode_z",
    "encode_z",
    "deep_merge",
    "as_list",
    "to_raw_line",
    "to_raw_record",
    "Z_TOPICS",
]

#: Topics whose payload is base64(raw-deflate(json)) on the wire (name WITHOUT the ``.z``).
Z_TOPICS = frozenset({"CarData", "Position"})


@dataclass
class Msg:
    topic: str  # F1 SignalR topic, ".z" suffix already stripped & payload decoded
    data: Any  # decoded JSON (dict/list). Patches, NOT snapshots: engine must deep-merge.
    utc: float  # unix seconds (float) of the message (from the feed timestamp)


# --------------------------------------------------------------------------- timestamps
_TS_RE = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?\s*(Z|[+-]\d{2}(?::?\d{2})?)?\s*$"
)


def parse_ts(value: Any) -> Optional[float]:
    """ISO-8601 string (``Z``/offset/naive-as-UTC, any fractional digits) or number -> unix seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v:  # NaN
            return None
        return v / 1000.0 if v > 1e11 else v  # tolerate epoch milliseconds
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    m = _TS_RE.match(value)
    if not m:
        try:
            return parse_ts(float(value))
        except ValueError:
            return None
    y, mo, d, h, mi, s, frac, tz = m.groups()
    try:
        t = float(calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s), 0, 0, 0)))
    except (ValueError, OverflowError):
        return None
    if frac:
        t += float("0." + frac[:9])
    if tz and tz != "Z":
        sign = -1 if tz[0] == "-" else 1
        digits = tz[1:].replace(":", "")
        off = int(digits[:2]) * 3600 + (int(digits[2:4]) * 60 if len(digits) >= 4 else 0)
        t -= sign * off
    return t


def format_ts(utc: float) -> str:
    """unix seconds -> ``2024-03-02T15:04:05.123Z`` (millisecond precision, like the F1 feed)."""
    ms_total = int(round(utc * 1000))
    secs, ms = divmod(ms_total, 1000)
    tm = time.gmtime(secs)
    return "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ" % (tm[0], tm[1], tm[2], tm[3], tm[4], tm[5], ms)


# --------------------------------------------------------------------------- .z payloads
def decode_z(payload: str | bytes) -> Any:
    """``base64 -> raw deflate -> utf-8 -> json`` (the F1 ``.z`` encoding)."""
    raw = base64.b64decode(payload, validate=False)
    try:
        text = zlib.decompress(raw, -zlib.MAX_WBITS)
    except zlib.error:
        text = zlib.decompress(raw, zlib.MAX_WBITS | 32)  # zlib/gzip framed (tolerated)
    return json.loads(text.decode("utf-8-sig"))


def encode_z(obj: Any) -> str:
    """Inverse of :func:`decode_z`."""
    comp = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    raw = comp.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8")) + comp.flush()
    return base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------- merge helpers
def _clone(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clone(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clone(x) for x in v]
    return v


def _is_index_dict(d: dict) -> bool:
    return bool(d) and all(isinstance(k, (str, int)) and str(k).isdigit() for k in d)


def deep_merge(dst: Any, patch: Any) -> Any:
    """Merge ``patch`` into ``dst`` in place and return the result.

    * dict + dict  -> recursive key-wise merge (patch values are copied, never aliased)
    * list + dict keyed by index (``{"2": {...}}``, the way F1 patches arrays) -> element-wise merge,
      growing the list with ``None`` holes if needed
    * anything else -> ``patch`` (a copy) replaces ``dst``

    If ``dst`` is not a dict the (copied) ``patch`` is returned, so always use the return value.
    """
    if isinstance(dst, dict) and isinstance(patch, dict):
        for k, v in patch.items():
            cur = dst.get(k)
            if isinstance(v, dict) and isinstance(cur, dict):
                deep_merge(cur, v)
            elif isinstance(v, dict) and isinstance(cur, list) and _is_index_dict(v):
                deep_merge(cur, v)
            else:
                dst[k] = _clone(v)
        return dst
    if isinstance(dst, list) and isinstance(patch, dict) and _is_index_dict(patch):
        for k, v in patch.items():
            i = int(k)
            while len(dst) <= i:
                dst.append(None)
            if isinstance(v, dict) and isinstance(dst[i], dict):
                deep_merge(dst[i], v)
            else:
                dst[i] = _clone(v)
        return dst
    return _clone(patch)


def as_list(value: Any) -> list:
    """Normalise F1's "list OR dict keyed by index" arrays (``Messages``, ``Captures``, ``Stints``)
    to a list ordered by index. ``None`` -> ``[]``."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if _is_index_dict(value):
            return [value[k] for k in sorted(value, key=lambda x: int(x))]
        return list(value.values())
    return []


# --------------------------------------------------------------------------- line parsing
def _loads_lenient(text: str) -> Any:
    """python-repr list/dict (fastf1 recording) first, JSON as fallback."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        pass
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError("unparseable line")


def _decode_payload(topic: str, data: Any, is_z: bool) -> Any:
    """Resolve string payloads (CompletionMessage dumps are JSON strings; .z are base64)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    if not isinstance(data, str):
        return data
    if is_z:
        try:
            return decode_z(data)
        except Exception:  # noqa: BLE001 - maybe a JSON string wrapping the base64 string
            pass
        inner = json.loads(data)  # raises ValueError for garbage
        if isinstance(inner, str):
            return decode_z(inner)
        return inner
    return json.loads(data)


def _guess_utc(data: Any) -> Optional[float]:
    """Fall back to a timestamp inside the payload (Heartbeat/ExtrapolatedClock/.z entries)."""
    if isinstance(data, dict):
        for key in ("Utc", "utc", "Timestamp"):
            if key in data:
                t = parse_ts(data[key])
                if t is not None:
                    return t
        for key in ("Entries", "Position"):
            v = data.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                t = _guess_utc(v[0])
                if t is not None:
                    return t
    return None


def parse_line(line: str | bytes, default_utc: float = 0.0) -> Optional[Msg]:
    """Parse one recording line. Returns ``None`` for blanks/garbage (never raises).

    ``default_utc`` is used when the line carries no timestamp (e.g. the initial snapshot dump).
    """
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        s = line.strip()
        if not s or s[0] not in "[{":
            return None
        ts_raw: Any = None
        if s[0] == "{":
            obj = json.loads(s)
            if not isinstance(obj, dict):
                return None
            topic = obj.get("topic", obj.get("Topic"))
            data = obj.get("data", obj.get("Data"))
            ts_raw = obj.get("utc", obj.get("ts", obj.get("timestamp", obj.get("Utc"))))
        else:
            obj = _loads_lenient(s)
            if not isinstance(obj, (list, tuple)) or len(obj) < 2:
                return None
            topic, data = obj[0], obj[1]
            if len(obj) > 2:
                ts_raw = obj[2]
        if not isinstance(topic, str) or not topic:
            return None
        is_z = topic.endswith(".z")
        if is_z:
            topic = topic[:-2]
        data = _decode_payload(topic, data, is_z)
        utc = parse_ts(ts_raw)
        if utc is None:
            utc = _guess_utc(data)
        if utc is None:
            utc = default_utc
        return Msg(topic, data, utc)
    except Exception:  # noqa: BLE001 - robustness over strictness: garbage in, None out
        return None


# --------------------------------------------------------------------------- serialisation
def to_raw_record(msg: Msg, encode: bool = True) -> list:
    """``Msg`` -> the ``[topic, payload, timestamp]`` list fastf1 would have recorded.

    With ``encode=True`` CarData/Position become ``'CarData.z'`` + base64 string; otherwise the
    decoded dict is kept (and the topic keeps no ``.z``).
    """
    ts = format_ts(msg.utc)
    if encode and msg.topic in Z_TOPICS:
        return [msg.topic + ".z", encode_z(msg.data), ts]
    return [msg.topic, msg.data, ts]


def to_raw_line(msg: Msg, encode: bool = True) -> str:
    """Exact fastf1 recording line (``str(list)``), no trailing newline."""
    return str(to_raw_record(msg, encode))


def iter_decoded(lines: Iterable[str]) -> Iterable[Msg]:
    """Convenience: parse many lines, inheriting the previous timestamp for untimed ones."""
    last = 0.0
    for ln in lines:
        m = parse_line(ln, default_utc=last)
        if m is not None:
            last = m.utc or last
            yield m
