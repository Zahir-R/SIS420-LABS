from __future__ import annotations

import random
from dataclasses import dataclass

from .env import ACTIONS, StunDashEnv
from .level import Level


@dataclass
class QLearningResult:
    q_table: dict[str, list[float]]
    policy: dict[str, int]
    metrics: dict
    learning_curve: list[float]


def _ensure(q_table: dict[str, list[float]], key: str) -> list[float]:
    if key not in q_table:
        q_table[key] = [0.0] * len(ACTIONS)
    return q_table[key]


def _choose(row: list[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    return row.index(max(row))


def train(
    level: Level,
    episodes: int = 4000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int | None = None,
) -> QLearningResult:
    env = StunDashEnv(level, seed=seed)
    rng = random.Random(seed)
    q_table: dict[str, list[float]] = {}
    returns: list[float] = []
    wins = 0
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset(seed=None if seed is None else seed + episode)
        total = 0.0
        done = False
        for _ in range(160):
            row = _ensure(q_table, state.key())
            action = _choose(row, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            next_row = _ensure(q_table, next_state.key())
            target = reward + gamma * max(next_row)
            row[action] += alpha * (target - row[action])
            total += reward
            state = next_state
            if done:
                if reward > 0:
                    wins += 1
                break
        returns.append(round(total, 6))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    policy: dict[str, int] = {}
    for key, row in q_table.items():
        policy[key] = row.index(max(row))
    metrics = {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_end": round(epsilon, 6),
        "mean_return": round(sum(returns) / len(returns), 6),
        "best_return": round(max(returns), 6),
        "last_return": round(returns[-1], 6),
        "win_rate": round(wins / episodes, 4),
        "states_seen": len(q_table),
    }
    return QLearningResult(
        q_table={
            key: [round(value, 6) for value in row] for key, row in q_table.items()
        },
        policy=policy,
        metrics=metrics,
        learning_curve=returns,
    )
