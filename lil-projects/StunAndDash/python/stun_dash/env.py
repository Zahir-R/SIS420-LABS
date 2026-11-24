from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace

from . import adversary
from .level import Level

ACTIONS = ("left", "right", "jump", "wait", "jump_left", "jump_right")
TIME_BUCKETS = 8
TIME_DECAY_PROBABILITY = 0.08
STUN_TICKS = 2


@dataclass(frozen=True)
class State:
    x: int
    y: int
    stun: int
    time_bucket: int
    threat: int

    def key(self) -> str:
        return f"{self.x},{self.y},{self.stun},{self.time_bucket},{self.threat}"


class StunDashEnv:
    def __init__(self, level: Level, seed: int | None = None) -> None:
        self.level = level
        self.rng = random.Random(seed)
        self.state = self._start_state()
        self.done = False
        self.steps = 0

    def _start_state(self) -> State:
        x, y = self.level.agent_start
        return State(x, y, 0, TIME_BUCKETS - 1, adversary.THREAT_NONE)

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = self._start_state()
        self.done = False
        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        if self.done:
            return self.state, 0.0, True, {"events": []}
        previous = self.state
        events: list[str] = []
        if previous.stun > 0:
            moved = replace(previous, stun=previous.stun - 1)
            reward = -1.0
            events.append("stunned")
        else:
            x, y = self._move(previous.x, previous.y, action)
            if self.level.is_goal(x, y):
                self.state = replace(previous, x=x, y=y)
                self.done = True
                return self.state, 100.0, True, {"events": ["goal"]}
            hit = adversary.hazard_hit(previous.threat, action)
            new_threat = adversary.sample_threat(self.rng)
            stun = STUN_TICKS if hit else 0
            reward = -1.0
            if x > previous.x:
                reward += 2.0
            elif x < previous.x:
                reward -= 1.0
            if hit:
                reward -= 8.0
                events.append("threat_hit")
            moved = State(x, y, stun, previous.time_bucket, new_threat)

        time_bucket = moved.time_bucket
        if time_bucket > 0 and self.rng.random() < TIME_DECAY_PROBABILITY:
            time_bucket -= 1
        moved = replace(moved, time_bucket=time_bucket)
        if time_bucket == 0:
            self.state = moved
            self.done = True
            return self.state, -50.0, True, {"events": events + ["timeout"]}
        self.state = moved
        self.steps += 1
        return self.state, reward, False, {"events": events}

    def transition_outcomes(
        self, state: State, action: int
    ) -> list[tuple[State, float, float, bool]]:
        """Returns (next_state, reward, probability, terminal) for Bellman."""
        if state.stun > 0:
            base = replace(state, stun=state.stun - 1)
            reward = -1.0
            return self._time_outcomes(base, reward)
        x, y = self._move(state.x, state.y, action)
        if self.level.is_goal(x, y):
            return [(replace(state, x=x, y=y), 100.0, 1.0, True)]
        result: list[tuple[State, float, float, bool]] = []
        base_reward = -1.0
        if x > state.x:
            base_reward += 2.0
        elif x < state.x:
            base_reward -= 1.0
        hit = adversary.hazard_hit(state.threat, action)
        for new_threat, probability in adversary.successors():
            stun = STUN_TICKS if hit else 0
            reward = base_reward - (8.0 if hit else 0.0)
            next_state = State(x, y, stun, state.time_bucket, new_threat)
            for (
                candidate,
                candidate_reward,
                time_probability,
                terminal,
            ) in self._time_outcomes(next_state, reward):
                result.append(
                    (
                        candidate,
                        candidate_reward,
                        probability * time_probability,
                        terminal,
                    )
                )
        return result

    def _time_outcomes(
        self, state: State, reward: float
    ) -> list[tuple[State, float, float, bool]]:
        if state.time_bucket <= 0:
            return [(state, -50.0, 1.0, True)]
        decayed = replace(state, time_bucket=state.time_bucket - 1)
        if decayed.time_bucket == 0:
            return [
                (decayed, -50.0, TIME_DECAY_PROBABILITY, True),
                (state, reward, 1.0 - TIME_DECAY_PROBABILITY, False),
            ]
        return [
            (decayed, reward, TIME_DECAY_PROBABILITY, False),
            (state, reward, 1.0 - TIME_DECAY_PROBABILITY, False),
        ]

    def move_path(
        self, x: int, y: int, action: int
    ) -> tuple[list[tuple[int, int]], tuple[int, int], bool]:
        jump = max(1, self.level.jump_height)
        cells: list[tuple[int, int]] = []
        blocked = False
        destination = (x, y)

        if action in (0, 1):
            nx = x - 1 if action == 0 else x + 1
            if not self.level.walkable(nx, y):
                blocked = True
            else:
                destination = (nx, y)
                cells = [destination]
        elif action == 2:
            cells = [(x, y - step) for step in range(1, jump + 1)]
            if any(not self.level.walkable(cx, cy) for cx, cy in cells):
                blocked = True
                cells = []
            else:
                destination = (x, y - jump)
        elif action in (4, 5):
            dx = -1 if action == 4 else 1
            cells = [(x, y - 1)]
            cells.extend((x, y - step) for step in range(2, jump))
            cells.extend((x + dx, y - step) for step in range(1, jump + 1))
            if any(not self.level.walkable(cx, cy) for cx, cy in cells):
                blocked = True
                cells = []
            else:
                destination = (x + dx, y - jump)

        if blocked:
            fall = self._gravity_path(x, y)
            final = fall[-1] if fall else (x, y)
            return fall, final, True

        fall = self._gravity_path(*destination)
        full = cells + fall
        final = full[-1] if full else destination
        return full, final, False

    def _move(self, x: int, y: int, action: int) -> tuple[int, int]:
        return self.move_path(x, y, action)[1]

    def _gravity_path(self, x: int, y: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        while y + 1 < self.level.height and not self.level.has_floor(x, y):
            y += 1
            cells.append((x, y))
        return cells

    def is_goal_reachable(self) -> bool:
        queue = deque([self.level.agent_start])
        seen = {self.level.agent_start}
        while queue:
            cell = queue.popleft()
            if cell == self.level.goal:
                return True
            for action in range(len(ACTIONS)):
                _, final, _ = self.move_path(cell[0], cell[1], action)
                if final not in seen:
                    seen.add(final)
                    queue.append(final)
        return False

    def all_states(self) -> list[State]:
        states: list[State] = []
        for y in range(self.level.height):
            for x in range(self.level.width):
                if not self.level.walkable(x, y):
                    continue
                for stun in range(STUN_TICKS + 1):
                    for time_bucket in range(TIME_BUCKETS):
                        for threat in range(11):
                            states.append(State(x, y, stun, time_bucket, threat))
        return states
