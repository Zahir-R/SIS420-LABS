from __future__ import annotations

import json
from pathlib import Path

from .bellman import BellmanResult
from .env import ACTIONS
from .level import Level
from .qlearning import QLearningResult

ACTION_NAMES = list(ACTIONS)
STATE_SCHEMA = "x,y,stun,time_bucket,threat,threat_distance"


def export_bellman(
    level: Level,
    result: BellmanResult,
    iterations: int,
    out_path: Path,
    metrics_path: Path,
) -> None:
    payload = {
        "algorithm": "bellman",
        "level_id": level.id,
        "state_schema": STATE_SCHEMA,
        "actions": ACTION_NAMES,
        "policy": result.policy,
        "values": result.values,
        "metrics": {
            "iterations": iterations,
            "states": len(result.policy),
            "convergence": result.convergence,
        },
    }
    _write_json(out_path, payload)
    _write_json(
        metrics_path,
        {
            "algorithm": "bellman",
            "iterations": iterations,
            "convergence": result.convergence,
            "states": len(result.policy),
        },
    )


def export_qlearning(
    level: Level, result: QLearningResult, out_path: Path, metrics_path: Path
) -> None:
    payload = {
        "algorithm": "qlearning",
        "level_id": level.id,
        "state_schema": STATE_SCHEMA,
        "actions": ACTION_NAMES,
        "policy": result.policy,
        "q": result.q_table,
        "metrics": result.metrics,
        "learning_curve": result.learning_curve,
    }
    _write_json(out_path, payload)
    _write_json(metrics_path, {"algorithm": "qlearning", **result.metrics})


def export_manifest(level: Level, policies: dict[str, str], out_path: Path) -> None:
    _write_json(
        out_path,
        {
            "level": level.id,
            "level_name": level.name,
            "state_schema": STATE_SCHEMA,
            "actions": ACTION_NAMES,
            "policies": policies,
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
