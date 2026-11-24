from __future__ import annotations

import argparse
import sys

from stun_dash.env import StunDashEnv
from stun_dash.level import Level


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida un nivel de Stun & Dash")
    parser.add_argument("level", help="Ruta al JSON del nivel")
    args = parser.parse_args()
    level = Level.from_json(args.level)
    env = StunDashEnv(level)
    if not env.is_goal_reachable():
        print(f"ERROR {level.id}: la meta no es alcanzable desde S", file=sys.stderr)
        sys.exit(1)
    print(f"OK {level.id}: start={level.agent_start} goal={level.goal}")
    if level.human_start is not None:
        print(f"human_start={level.human_start}")


if __name__ == "__main__":
    main()
