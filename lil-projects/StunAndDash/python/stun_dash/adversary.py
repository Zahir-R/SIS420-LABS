from __future__ import annotations

import random

DANGER_NONE = 0
DANGER_TRAP = 1
DANGER_FIRE = 2

OUTCOMES = (
    (DANGER_NONE, False, 0.84),
    (DANGER_TRAP, True, 0.16 * 0.5 * 0.45),
    (DANGER_TRAP, False, 0.16 * 0.5 * 0.55),
    (DANGER_FIRE, True, 0.16 * 0.5 * 0.45),
    (DANGER_FIRE, False, 0.16 * 0.5 * 0.55),
)


def sample(rng: random.Random) -> tuple[int, bool]:
    roll = rng.random()
    cumulative = 0.0
    for danger, hit, probability in OUTCOMES:
        cumulative += probability
        if roll <= cumulative:
            return danger, hit
    return DANGER_NONE, False


def outcomes() -> tuple[tuple[int, bool, float], ...]:
    return OUTCOMES
