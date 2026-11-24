from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from stun_dash.env import StunDashEnv
from stun_dash.level import Level

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa una política exportada")
    parser.add_argument(
        "--policy", default=str(ROOT / "shared/policies/level01/qlearning.json")
    )
    parser.add_argument("--level", default=str(ROOT / "shared/levels/level01.json"))
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    level = Level.from_json(args.level)
    data = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    policy = {key: int(value) for key, value in data["policy"].items()}
    rng = random.Random(7)
    wins = 0
    total = 0.0
    for episode in range(args.episodes):
        env = StunDashEnv(level, seed=episode)
        state = env.reset(seed=episode)
        episode_return = 0.0
        done = False
        for _ in range(160):
            action = policy.get(state.key(), rng.randrange(6))
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                wins += int(reward > 0)
                break
        total += episode_return
    print(
        f"policy={Path(args.policy).name} episodes={args.episodes} "
        f"win_rate={wins / args.episodes:.3f} mean_return={total / args.episodes:.2f}"
    )


if __name__ == "__main__":
    main()
