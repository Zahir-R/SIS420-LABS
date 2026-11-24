from pathlib import Path

from stun_dash.level import Level

ROOT = Path(__file__).resolve().parents[2]


def test_level_contract() -> None:
    level = Level.from_json(ROOT / "shared/levels/level01.json")
    assert level.width == 20
    assert level.height == 9
    assert level.tile(*level.agent_start) == "S"
    assert level.tile(*level.goal) == "G"
    assert level.trap_slots
    assert level.fire_lanes == (7,)


def test_goal_is_reachable() -> None:
    from stun_dash.env import StunDashEnv

    level = Level.from_json(ROOT / "shared/levels/level01.json")
    assert StunDashEnv(level).is_goal_reachable()
