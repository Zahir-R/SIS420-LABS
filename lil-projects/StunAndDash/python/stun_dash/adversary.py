from __future__ import annotations

import random

THREAT_NONE = 0
TRAP_LEFT = 1
TRAP_RIGHT = 2
TRAP_SAME = 3
TRAP_ABOVE = 4
TRAP_BELOW = 5
FIRE_LEFT = 6
FIRE_RIGHT = 7
FIRE_SAME = 8
FIRE_ABOVE = 9
FIRE_BELOW = 10

TRAP_CODES = (TRAP_LEFT, TRAP_RIGHT, TRAP_SAME, TRAP_ABOVE, TRAP_BELOW)
FIRE_CODES = (FIRE_LEFT, FIRE_RIGHT, FIRE_SAME, FIRE_ABOVE, FIRE_BELOW)

DISTRIBUTION = (
    (THREAT_NONE, 0.50),
    (TRAP_LEFT, 0.06),
    (TRAP_RIGHT, 0.06),
    (TRAP_SAME, 0.06),
    (TRAP_ABOVE, 0.06),
    (TRAP_BELOW, 0.06),
    (FIRE_LEFT, 0.05),
    (FIRE_RIGHT, 0.05),
    (FIRE_SAME, 0.05),
    (FIRE_ABOVE, 0.025),
    (FIRE_BELOW, 0.025),
)


def sample_threat(rng: random.Random) -> int:
    roll = rng.random()
    cumulative = 0.0
    for code, probability in DISTRIBUTION:
        cumulative += probability
        if roll <= cumulative:
            return code
    return THREAT_NONE


def successors() -> tuple[tuple[int, float], ...]:
    return DISTRIBUTION


def hazard_hit(threat: int, action: int) -> bool:
    if threat == THREAT_NONE:
        return False
    if threat == TRAP_LEFT:
        return action in (0, 4)
    if threat == TRAP_RIGHT:
        return action in (1, 5)
    if threat == TRAP_SAME:
        return action == 3
    if threat == TRAP_ABOVE:
        return action in (2, 4, 5)
    if threat == TRAP_BELOW:
        return action == 3
    if threat == FIRE_LEFT:
        return action == 0
    if threat == FIRE_RIGHT:
        return action == 1
    if threat == FIRE_SAME:
        return action == 3
    if threat == FIRE_ABOVE:
        return action in (2, 4, 5)
    if threat == FIRE_BELOW:
        return action == 3
    return False
