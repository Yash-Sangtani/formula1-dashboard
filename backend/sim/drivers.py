"""Static grid data for the simulator: 20 drivers, 10 teams (2025-style colours) and pace offsets."""
from __future__ import annotations

from typing import Dict, List, NamedTuple


class Driver(NamedTuple):
    num: str
    tla: str
    first: str
    last: str
    team: str


#: team -> (TeamColour hex without '#', car pace delta in s/lap vs the best car)
TEAMS: Dict[str, tuple] = {
    "McLaren": ("FF8000", 0.00),
    "Red Bull Racing": ("3671C6", 0.30),
    "Ferrari": ("E80020", 0.25),
    "Mercedes": ("27F4D2", 0.45),
    "Williams": ("64C4FF", 1.25),
    "Aston Martin": ("229971", 1.50),
    "Racing Bulls": ("6692FF", 1.65),
    "Alpine": ("FF87BC", 2.00),
    "Haas F1 Team": ("B6BABD", 2.15),
    "Kick Sauber": ("52E252", 2.50),
}

DRIVERS: List[Driver] = [
    Driver("1", "VER", "Max", "Verstappen", "Red Bull Racing"),
    Driver("22", "TSU", "Yuki", "Tsunoda", "Red Bull Racing"),
    Driver("4", "NOR", "Lando", "Norris", "McLaren"),
    Driver("81", "PIA", "Oscar", "Piastri", "McLaren"),
    Driver("16", "LEC", "Charles", "Leclerc", "Ferrari"),
    Driver("44", "HAM", "Lewis", "Hamilton", "Ferrari"),
    Driver("63", "RUS", "George", "Russell", "Mercedes"),
    Driver("12", "ANT", "Kimi", "Antonelli", "Mercedes"),
    Driver("14", "ALO", "Fernando", "Alonso", "Aston Martin"),
    Driver("18", "STR", "Lance", "Stroll", "Aston Martin"),
    Driver("10", "GAS", "Pierre", "Gasly", "Alpine"),
    Driver("43", "COL", "Franco", "Colapinto", "Alpine"),
    Driver("23", "ALB", "Alexander", "Albon", "Williams"),
    Driver("55", "SAI", "Carlos", "Sainz", "Williams"),
    Driver("30", "LAW", "Liam", "Lawson", "Racing Bulls"),
    Driver("6", "HAD", "Isack", "Hadjar", "Racing Bulls"),
    Driver("27", "HUL", "Nico", "Hulkenberg", "Kick Sauber"),
    Driver("5", "BOR", "Gabriel", "Bortoleto", "Kick Sauber"),
    Driver("31", "OCO", "Esteban", "Ocon", "Haas F1 Team"),
    Driver("87", "BEA", "Oliver", "Bearman", "Haas F1 Team"),
]

#: compound -> (pace delta s/lap, degradation s per lap of age, life in laps before the cliff)
COMPOUNDS: Dict[str, tuple] = {
    "SOFT": (-0.45, 0.075, 16),
    "MEDIUM": (0.0, 0.045, 26),
    "HARD": (0.35, 0.028, 38),
}
