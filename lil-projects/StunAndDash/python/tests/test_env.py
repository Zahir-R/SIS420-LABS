from pathlib import Path

from stun_dash.env import ACTIONS, StunDashEnv
from stun_dash.level import Level

ROOT = Path(__file__).resolve().parents[2]


def test_reset_and_step() -> None:
    level = Level.from_json(ROOT / "shared/levels/level01.json")
    env = StunDashEnv(level, seed=7)
    state = env.reset(seed=7)
    assert state.key() == "7,1,0,7,0"
    next_state, _, _, _ = env.step(1)
    assert next_state.x >= state.x
    assert 0 <= next_state.time_bucket <= 7
    assert 0 <= next_state.danger <= 2


def test_actions_exist() -> None:
    assert len(ACTIONS) == 6


def _path_level() -> Level:
    return Level(
        id="path-test",
        name="Path",
        width=5,
        height=5,
        rows=(
            "#####",
            "#...#",
            "##..#",
            "#...#",
            "#####",
        ),
        agent_start=(1, 3),
        goal=(3, 3),
        time_limit_s=15,
        tick_ms=250,
        jump_height=2,
        trap_slots=(),
        fire_lanes=(3,),
    )


def test_jump_checks_intermediate_cell() -> None:
    env = StunDashEnv(_path_level(), seed=1)
    cells, final, blocked = env.move_path(1, 3, 2)
    assert blocked is True
    assert final == (1, 3)
    assert cells == []


def test_diagonal_jump_exposes_intermediate_cells() -> None:
    level = Level(
        id="diag-test",
        name="Diag",
        width=5,
        height=5,
        rows=("#####", "#...#", "#...#", "#...#", "#####"),
        agent_start=(1, 3),
        goal=(3, 3),
        time_limit_s=15,
        tick_ms=250,
        jump_height=2,
        trap_slots=(),
        fire_lanes=(3,),
    )
    env = StunDashEnv(level, seed=1)
    cells, _, blocked = env.move_path(1, 3, 5)
    assert blocked is False
    assert cells[:3] == [(1, 2), (2, 2), (2, 1)]


def test_small_diagonal_jump_includes_vertical_cell() -> None:
    level = Level(
        id="diag-one",
        name="Diag one",
        width=5,
        height=5,
        rows=("#####", "#...#", "#...#", "#...#", "#####"),
        agent_start=(1, 3),
        goal=(3, 3),
        time_limit_s=15,
        tick_ms=250,
        jump_height=1,
        trap_slots=(),
        fire_lanes=(3,),
    )
    env = StunDashEnv(level, seed=1)
    cells, _, blocked = env.move_path(1, 3, 5)
    assert blocked is False
    assert cells[:2] == [(1, 2), (2, 2)]
