"""Curriculum level definitions and the advancement gate.

Five levels of increasing difficulty. Levels 1-3 are a "receive"
task (a launcher serves balls at the agent, no opponent); levels
4-5 add a scripted returner so opponent_y is real and rallies form.
Level 5 also enables domain randomization of the sim physics.
Advance when the pass rate clears the gate overall AND on each
side independently (the anti-asymmetry floor).
"""

from dataclasses import dataclass
from typing import Tuple

from agent import config


@dataclass(frozen=True)
class Level:
    """One curriculum stage.

    speed_range/angle_range parametrize served-ball velocity;
    has_returner toggles the scripted opponent; randomize_physics
    enables per-episode domain randomization (level 5 only).
    """

    index: int
    speed_range: Tuple[float, float]
    angle_range: Tuple[float, float]
    has_returner: bool
    randomize_physics: bool


# angle is radians off the horizontal serve direction
LEVELS: Tuple[Level, ...] = (
    Level(1, (0.9, 1.4), (0.0, 0.30), False, False),
    Level(2, (1.4, 2.2), (0.0, 0.55), False, False),
    Level(3, (2.2, 3.2), (0.0, 0.70), False, False),
    Level(4, (0.9, 3.2), (0.0, 0.70), True, False),
    Level(5, (0.9, 3.2), (0.0, 0.70), True, True),
)

assert len(LEVELS) == config.MAX_LEVEL


def get_level(index: int) -> Level:
    """Return the Level for a 1-based index (clamped to MAX_LEVEL)."""
    if index < 1:
        raise ValueError(f"level index must be >= 1, got {index}")
    return LEVELS[min(index, config.MAX_LEVEL) - 1]


def passes_gate(
    overall_rate: float,
    left_rate: float,
    right_rate: float,
) -> bool:
    """True if eval clears the overall and per-side thresholds."""
    return (
        overall_rate >= config.PASS_SUCCESS_RATE
        and left_rate >= config.PER_SIDE_PASS_RATE
        and right_rate >= config.PER_SIDE_PASS_RATE
    )
