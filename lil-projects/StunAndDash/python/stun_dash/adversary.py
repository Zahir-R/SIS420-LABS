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

THREAT_DISTANCES = (0, 1, 2)
DISTANCE_PROBABILITIES = (0.25, 0.5, 0.25)

THREAT_DISTRIBUTION = (
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


def sample_threat(rng: random.Random) -> tuple[int, int]:
    roll = rng.random()
    cumulative = 0.0
    code = THREAT_NONE
    for candidate, probability in THREAT_DISTRIBUTION:
        cumulative += probability
        if roll <= cumulative:
            code = candidate
            break
    if code == THREAT_NONE:
        return code, 0
    distance = rng.choices(THREAT_DISTANCES, weights=DISTANCE_PROBABILITIES, k=1)[0]
    return code, distance


def successors() -> tuple[tuple[int, int, float], ...]:
    result: list[tuple[int, int, float]] = []
    for code, code_probability in THREAT_DISTRIBUTION:
        if code == THREAT_NONE:
            result.append((code, 0, code_probability))
            continue
        for distance, distance_probability in zip(
            THREAT_DISTANCES, DISTANCE_PROBABILITIES
        ):
            result.append((code, distance, code_probability * distance_probability))
    return tuple(result)


def hazard_hit(threat: int, distance: int, action: int) -> bool:
    if threat == THREAT_NONE or distance > 1:
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
