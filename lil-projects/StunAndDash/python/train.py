from __future__ import annotations

import argparse
from pathlib import Path

from stun_dash.bellman import value_iteration
from stun_dash.env import StunDashEnv
from stun_dash.export import export_bellman, export_manifest, export_qlearning
from stun_dash.level import Level
from stun_dash.qlearning import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el agente de Stun & Dash")
    parser.add_argument("--mode", choices=["bellman", "qlearning"], required=True)
    parser.add_argument("--level", default=str(ROOT / "shared/levels/level01.json"))
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    level = Level.from_json(args.level)
    env = StunDashEnv(level)
    if not env.is_goal_reachable():
        raise SystemExit("ERROR: la meta no es alcanzable desde S")
    policies = ROOT / "shared/policies" / level.id
    out = Path(args.out) if args.out else policies / f"{args.mode}.json"
    metrics_path = out.parent / f"metrics-{args.mode}.json"
    if args.mode == "bellman":
        result = value_iteration(level)
        export_bellman(level, result, result.iterations, out, metrics_path)
        print(
            f"Bellman {level.id}: {result.iterations} iteraciones, "
            f"{len(result.policy)} estados"
        )
    else:
        result = train(level, episodes=args.episodes, seed=args.seed)
        export_qlearning(level, result, out, metrics_path)
        print(f"Q-Learning {level.id}: {result.metrics}")

    export_manifest(
        level,
        {"bellman": "bellman.json", "qlearning": "qlearning.json"},
        policies / "manifest.json",
    )


if __name__ == "__main__":
    main()
