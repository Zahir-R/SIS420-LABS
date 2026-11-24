import json
from pathlib import Path

from stun_dash.bellman import value_iteration
from stun_dash.export import export_qlearning
from stun_dash.level import Level
from stun_dash.qlearning import train

ROOT = Path(__file__).resolve().parents[2]


def test_bellman_policy_and_qlearning_export(tmp_path: Path) -> None:
    level = Level.from_json(ROOT / "shared/levels/level01.json")
    bellman = value_iteration(level, max_iterations=2)
    assert bellman.policy
    q = train(level, episodes=40, seed=2)
    assert q.policy
    assert q.metrics["episodes"] == 40
    out = tmp_path / "q.json"
    metrics = tmp_path / "metrics.json"
    export_qlearning(level, q, out, metrics)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "qlearning"
    assert payload["state_schema"] == "x,y,stun,time_bucket,danger"
