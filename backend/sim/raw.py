"""Writing simulated (or any) messages in the exact fastf1 recording format."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Iterable, Iterator, Optional, Union

from ..ingest.messages import Msg, to_raw_line, to_raw_record

__all__ = ["to_raw_record", "to_raw_line", "raw_lines", "record_to_file", "arecord_to_file"]


def raw_lines(msgs: Iterable[Msg], encode: bool = True) -> Iterator[str]:
    """Yield recording lines (no newline). ``encode=True`` emits CarData/Position as ``.z`` base64 blobs
    (exercising the decoder path); ``encode=False`` keeps them as decoded dicts under the bare topic."""
    for m in msgs:
        yield to_raw_line(m, encode)


def _iter_source(src: Any) -> Optional[Iterable[Msg]]:
    """Synchronously iterable view of ``src`` (iterable of Msg, ``SimRace``, ``SimSource``), else ``None``."""
    if hasattr(src, "all_messages"):
        return src.all_messages()
    if hasattr(src, "control"):  # a real-time async source
        return None
    if hasattr(src, "__iter__"):
        return src
    return None


def _prep_path(path: Union[str, "os.PathLike[str]"]) -> str:
    path = os.fspath(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def record_to_file(source_or_iterable: Any, path: Union[str, "os.PathLike[str]"], *, encode: bool = True,
                   limit: Optional[int] = None) -> int:
    """Write messages to ``path`` in fastf1 recording format. Returns the number of lines written.

    ``source_or_iterable`` may be an iterable of :class:`Msg` (list, ``SimRace``), a ``SimSource``
    (written without real-time pacing), or any async source (consumed with ``asyncio.run`` -- paced by that
    source's own control, so only use that for sources that are not rate-limited).
    """
    it = _iter_source(source_or_iterable)
    if it is None:
        return asyncio.run(arecord_to_file(source_or_iterable, path, encode=encode, limit=limit))
    n = 0
    with open(_prep_path(path), "w", encoding="utf-8") as f:
        for ln in raw_lines(it, encode):
            f.write(ln + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    return n


async def arecord_to_file(source: Any, path: Union[str, "os.PathLike[str]"], *, encode: bool = True,
                          limit: Optional[int] = None) -> int:
    it = _iter_source(source)
    n = 0
    with open(_prep_path(path), "w", encoding="utf-8") as f:
        if it is not None:
            for ln in raw_lines(it, encode):
                f.write(ln + "\n")
                n += 1
                if limit is not None and n >= limit:
                    break
        else:
            async for m in source.messages():
                f.write(to_raw_line(m, encode) + "\n")
                n += 1
                if limit is not None and n >= limit:
                    break
    return n
