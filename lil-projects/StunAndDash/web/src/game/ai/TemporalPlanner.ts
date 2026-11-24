import { movePath, type LevelData } from "../level/LevelLoader";

export interface PlanResult {
  action: number;
  actions: number[];
  cost: number;
  safe: boolean;
}

interface QueueNode {
  x: number;
  y: number;
  tick: number;
  cost: number;
  actions: number[];
  stallRun: number;
}

type HazardMap = Set<string>;

const ACTIONS = [0, 1, 2, 3, 4, 5];
const MAX_STALLS = 4;

export class TemporalPlanner {
  plan(
    level: LevelData,
    startX: number,
    startY: number,
    traps: Array<[number, number]>,
    projectileCells: Array<[number, number, number]>,
    horizon = 24,
  ): PlanResult | null {
    const safeTraps = this.toSet(traps);
    const safeProjectiles = this.toTimeSet(projectileCells);
    const safe = this.search(
      level, startX, startY, safeTraps, safeProjectiles, horizon, true, 0, true,
    );
    const risk = this.search(
      level, startX, startY, safeTraps, safeProjectiles, horizon, false, 2, true,
    );
    if (safe && (!risk || safe.cost <= risk.cost)) {
      return { ...safe, safe: true };
    }
    if (risk) {
      return { ...risk, safe: false };
    }
    return null;
  }

  findSafe(
    level: LevelData,
    startX: number,
    startY: number,
    traps: Array<[number, number]>,
    projectileCells: Array<[number, number, number]>,
    horizon = 24,
  ): PlanResult | null {
    return this.search(
      level,
      startX,
      startY,
      this.toSet(traps),
      this.toTimeSet(projectileCells),
      horizon,
      true,
      0,
      true,
    );
  }

  findSafeNoStall(
    level: LevelData,
    startX: number,
    startY: number,
    traps: Array<[number, number]>,
    projectileCells: Array<[number, number, number]>,
    horizon = 24,
  ): PlanResult | null {
    return this.search(
      level,
      startX,
      startY,
      this.toSet(traps),
      this.toTimeSet(projectileCells),
      horizon,
      true,
      0,
      false,
    );
  }

  private search(
    level: LevelData,
    startX: number,
    startY: number,
    traps: HazardMap,
    projectileCells: HazardMap,
    horizon: number,
    blockHazards: boolean,
    hazardPenalty: number,
    allowStall: boolean,
  ): PlanResult | null {
    const queue: QueueNode[] = [
      { x: startX, y: startY, tick: 0, cost: 0, actions: [], stallRun: 0 },
    ];
    const best = new Map<string, number>();
    while (queue.length > 0) {
      queue.sort((a, b) => a.cost - b.cost);
      const node = queue.shift()!;
      const stateKey = `${node.x},${node.y},${node.tick},${node.stallRun}`;
      if ((best.get(stateKey) ?? Number.POSITIVE_INFINITY) <= node.cost) continue;
      best.set(stateKey, node.cost);

      if (node.x === level.goal[0] && node.y === level.goal[1]) {
        return {
          action: node.actions[0] ?? -1,
          actions: node.actions,
          cost: node.cost,
          safe: blockHazards,
        };
      }
      if (node.tick >= horizon) continue;

      for (const action of ACTIONS) {
        const path = movePath(level, node.x, node.y, action);
        const [nx, ny] = path.final;
        if (level.rows[ny][nx] === "#") continue;
        const stalled = nx === node.x && ny === node.y;
        if (stalled && (!allowStall || node.stallRun >= MAX_STALLS)) continue;
        const nextTick = node.tick + 1;
        const isHazard = path.cells.some(([cx, cy]) => {
          const cellKey = `${cx},${cy}`;
          return traps.has(cellKey) || projectileCells.has(`${nextTick}:${cellKey}`);
        });
        if (isHazard && blockHazards) continue;
        const nextCost = node.cost + 1 + (isHazard ? hazardPenalty : 0);
        const nextStall = stalled ? node.stallRun + 1 : 0;
        const nextKey = `${nx},${ny},${nextTick},${nextStall}`;
        if ((best.get(nextKey) ?? Number.POSITIVE_INFINITY) <= nextCost) continue;
        queue.push({
          x: nx,
          y: ny,
          tick: nextTick,
          cost: nextCost,
          actions: [...node.actions, action],
          stallRun: nextStall,
        });
      }
    }
    return null;
  }

  private toSet(cells: Array<[number, number]>): HazardMap {
    return new Set(cells.map(([x, y]) => `${x},${y}`));
  }

  private toTimeSet(cells: Array<[number, number, number]>): HazardMap {
    const result = new Set<string>();
    cells.forEach(([x, y, tick]) => result.add(`${tick}:${x},${y}`));
    return result;
  }
}
