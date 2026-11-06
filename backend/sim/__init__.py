"""Deterministic synthetic F1 race generator (see ``race.py``)."""
from ..ingest.messages import Msg
from .raw import arecord_to_file, raw_lines, record_to_file
from .race import SimConfig, SimRace, generate
from .tracks import Track, get_track, list_tracks

__all__ = [
    "Msg", "SimConfig", "SimRace", "generate", "generate_messages", "Track", "get_track", "list_tracks",
    "record_to_file", "arecord_to_file", "raw_lines",
]


def generate_messages(seed: int = 1, laps: int = 20, **kw):
    """Convenience: ``list[Msg]`` for a full simulated race."""
    return generate(SimConfig(seed=seed, laps=laps, **kw)).messages()
