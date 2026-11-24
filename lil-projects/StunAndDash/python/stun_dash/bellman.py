from __future__ import annotations

from dataclasses import dataclass

from .env import ACTIONS, StunDashEnv, State
from .level import Level


@dataclass
class BellmanResult:
    values: dict[str, float]
    policy: dict[str, int]
    iterations: int
    convergence: list[dict]


def _terminal(level: Level, state: State) -> bool:
    return state.time_bucket <= 0 or level.is_goal(state.x, state.y)


def value_iteration(
    level: Level, gamma: float = 0.95, theta: float = 1e-4, max_iterations: int = 300
) -> BellmanResult:
    env = StunDashEnv(level)
    states = env.all_states()
    values = {state.key(): 0.0 for state in states}
    convergence: list[dict] = []
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        new_values = dict(values)
        for state in states:
            if _terminal(level, state):
                continue
            best = float("-inf")
            for action in range(len(ACTIONS)):
                value = 0.0
                for (
                    next_state,
                    reward,
                    probability,
                    terminal,
                ) in env.transition_outcomes(state, action):
                    future = 0.0 if terminal else values[next_state.key()]
                    value += probability * (reward + gamma * future)
                best = max(best, value)
            new_values[state.key()] = best
            delta = max(delta, abs(best - values[state.key()]))
        values = new_values
        iterations = iteration
        convergence.append({"iteration": iteration, "delta": round(delta, 10)})
        if delta < theta:
            break

    policy: dict[str, int] = {}
    for state in states:
        if _terminal(level, state):
            continue
        best_action = 0
        best_value = float("-inf")
        for action in range(len(ACTIONS)):
            value = 0.0
            for next_state, reward, probability, terminal in env.transition_outcomes(
                state, action
            ):
                future = 0.0 if terminal else values[next_state.key()]
                value += probability * (reward + gamma * future)
            if value > best_value:
                best_value = value
                best_action = action
        policy[state.key()] = best_action
    return BellmanResult(values, policy, iterations, convergence)
