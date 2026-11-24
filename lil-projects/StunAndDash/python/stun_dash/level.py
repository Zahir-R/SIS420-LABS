from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Level:
    id: str
    name: str
    width: int
    height: int
    rows: tuple[str, ...]
    agent_start: tuple[int, int]
    goal: tuple[int, int]
    time_limit_s: float
    tick_ms: int
    jump_height: int
    trap_slots: tuple[tuple[int, int], ...]
    fire_lanes: tuple[int, ...]
    human_start: tuple[int, int] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "Level":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = tuple(str(row) for row in data["rows"])
        width = int(data["width"])
        height = int(data["height"])
        if len(rows) != height or any(len(row) != width for row in rows):
            raise ValueError("Las filas no coinciden con width/height")
        starts = [
            (x, y)
            for y, row in enumerate(rows)
            for x, tile in enumerate(row)
            if tile == "S"
        ]
        goals = [
            (x, y)
            for y, row in enumerate(rows)
            for x, tile in enumerate(row)
            if tile == "G"
        ]
        if len(starts) != 1:
            raise ValueError("El nivel debe tener exactamente una S")
        if len(goals) != 1:
            raise ValueError("El nivel debe tener exactamente una G")
        start = starts[0]
        goal = goals[0]
        human_raw = data.get("human_start")
        human_start = (
            (int(human_raw[0]), int(human_raw[1])) if human_raw is not None else None
        )
        if human_start is not None:
            hx, hy = human_start
            if hx < 0 or hy < 0 or hx >= width or hy >= height or rows[hy][hx] == "#":
                raise ValueError("human_start debe estar dentro y fuera de paredes")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            width=width,
            height=height,
            rows=rows,
            agent_start=start,
            goal=goal,
            time_limit_s=float(data["time_limit_s"]),
            tick_ms=int(data["tick_ms"]),
            jump_height=int(data.get("jump_height", 2)),
            trap_slots=tuple(
                tuple(int(v) for v in slot) for slot in data["trap_slots"]
            ),
            fire_lanes=tuple(int(v) for v in data["fire_lanes"]),
            human_start=human_start,
        )

    def tile(self, x: int, y: int) -> str:
        if x < 0 or y < 0 or y >= self.height or x >= self.width:
            return "#"
        return self.rows[y][x]

    def walkable(self, x: int, y: int) -> bool:
        return self.tile(x, y) != "#"

    def has_floor(self, x: int, y: int) -> bool:
        return self.tile(x, y + 1) == "#"

    def is_goal(self, x: int, y: int) -> bool:
        return (x, y) == self.goal

    def time_bucket_count(self) -> int:
        return 8
