import { Projectile } from "../entities/Projectile";
import { movePath, type LevelData } from "../level/LevelLoader";
import { TemporalPlanner } from "./TemporalPlanner";

export type PolicyChooser = (stateKey: string) => number;

export interface RouteDecision {
  action: number;
  mode: "policy" | "detour";
  policyReturn: number;
  detourReturn: number;
  policyActions: number[];
  detourActions: number[];
}

interface TimedProjectile {
  cells: Map<number, Set<string>>;
  active: boolean;
}

interface SimulationResult {
  reward: number;
  actions: number[];
  goal: boolean;
  timeout: boolean;
}

const GAMMA = 0.95;
const MAX_HORIZON = 36;
const MAX_STALLS = 4;

export class RouteEvaluator {
  constructor(private readonly planner: TemporalPlanner) {}

  evaluate(
    level: LevelData,
    startX: number,
    startY: number,
    remainingSeconds: number,
    traps: Array<[number, number]>,
    projectiles: Projectile[],
    choose: PolicyChooser,
  ): RouteDecision | null {
    const timedProjectiles = this.snapshotProjectiles(level, projectiles);
    const policy = this.simulatePolicy(
      level,
      startX,
      startY,
      remainingSeconds,
      traps,
      timedProjectiles,
      choose,
    );
    if (policy.actions.length === 0) return null;

    const safe = this.planner.findSafe(
      level,
      startX,
      startY,
      traps,
      this.flattenProjectiles(timedProjectiles),
      MAX_HORIZON,
    );
    const detour = safe ? this.simulateSequence(level, startX, startY, remainingSeconds, safe.actions) : null;
    const policyReward = policy.goal ? policy.reward : Number.NEGATIVE_INFINITY;
    const detourReward =
      detour && detour.goal ? detour.reward : Number.NEGATIVE_INFINITY;

    if (!policy.goal && !(detour && detour.goal)) {
      return null;
    }

    if (detour && detourReward > policyReward) {
      return {
        action: detour.actions[0],
        mode: "detour",
        policyReturn: policyReward,
        detourReturn: detourReward,
        policyActions: policy.actions,
        detourActions: detour.actions,
      };
    }
    return {
      action: policy.actions[0],
      mode: "policy",
      policyReturn: policyReward,
      detourReturn: detourReward,
      policyActions: policy.actions,
      detourActions: detour?.actions ?? [],
    };
  }

  private simulatePolicy(
    level: LevelData,
    startX: number,
    startY: number,
    remainingSeconds: number,
    trapCells: Array<[number, number]>,
    projectiles: TimedProjectile[],
    choose: PolicyChooser,
  ): SimulationResult {
    let x = startX;
    let y = startY;
    let stun = 0;
    let timeLeft = remainingSeconds;
    let reward = 0;
    let discount = 1;
    let goal = false;
    let timeout = false;
    let stallRun = 0;
    const actions: number[] = [];
    const traps = new Set(trapCells.map(([tx, ty]) => `${tx},${ty}`));
    const tickSeconds = level.tick_ms / 1000;

    for (let tick = 1; tick <= MAX_HORIZON; tick += 1) {
      if (timeLeft <= 0) {
        timeout = true;
        reward -= 50 * discount;
        break;
      }
      if (stun > 0) {
        stun -= 1;
        timeLeft -= tickSeconds;
        reward -= discount;
        discount *= GAMMA;
        continue;
      }

      const danger = this.dangerAt(x, y, traps, projectiles);
      const timeBucket = Math.min(
        7,
        Math.max(0, Math.floor((timeLeft / level.time_limit_s) * 7)),
      );
      let action = choose(`${x},${y},${stun},${timeBucket},${danger}`);
      let path = movePath(level, x, y, action);
      let stalled = path.final[0] === x && path.final[1] === y;
      if (stalled && stallRun >= MAX_STALLS) {
        action = this.escapeAction(level, x, y);
        path = movePath(level, x, y, action);
        stalled = path.final[0] === x && path.final[1] === y;
      }
      stallRun = stalled ? stallRun + 1 : 0;
      actions.push(action);
      const checkCells: Array<[number, number]> = [[x, y], ...path.cells];
      const trapCell = checkCells.find(([cx, cy]) => traps.has(`${cx},${cy}`));
      let hit = false;
      if (trapCell) {
        traps.delete(`${trapCell[0]},${trapCell[1]}`);
        hit = true;
      }
      for (const projectile of projectiles) {
        if (!projectile.active) continue;
        const cells = projectile.cells.get(tick);
        if (!cells) continue;
        if (checkCells.some(([cx, cy]) => cells.has(`${cx},${cy}`))) {
          projectile.active = false;
          hit = true;
        }
      }

      const destination = trapCell ?? path.final;
      reward -= discount;
      if (destination[0] > x) reward += 2 * discount;
      if (destination[0] < x) reward -= discount;
      if (hit) reward -= 8 * discount;
      x = destination[0];
      y = destination[1];
      if (hit) stun = 2;
      if (x === level.goal[0] && y === level.goal[1]) {
        reward += 100 * discount;
        goal = true;
        break;
      }
      timeLeft -= tickSeconds;
      discount *= GAMMA;
    }
    return { reward, actions, goal, timeout };
  }

  private simulateSequence(
    level: LevelData,
    startX: number,
    startY: number,
    remainingSeconds: number,
    actions: number[],
  ): SimulationResult {
    let x = startX;
    let y = startY;
    let timeLeft = remainingSeconds;
    let reward = 0;
    let discount = 1;
    let goal = false;
    let timeout = false;
    const tickSeconds = level.tick_ms / 1000;

    for (const action of actions) {
      if (timeLeft <= 0) {
        timeout = true;
        reward -= 50 * discount;
        break;
      }
      const path = movePath(level, x, y, action);
      const [nx, ny] = path.final;
      reward -= discount;
      if (nx > x) reward += 2 * discount;
      if (nx < x) reward -= discount;
      x = nx;
      y = ny;
      if (x === level.goal[0] && y === level.goal[1]) {
        reward += 100 * discount;
        goal = true;
        break;
      }
      timeLeft -= tickSeconds;
      discount *= GAMMA;
    }
    return { reward, actions, goal, timeout };
  }

  private snapshotProjectiles(level: LevelData, projectiles: Projectile[]): TimedProjectile[] {
    return projectiles
      .filter((projectile) => projectile.active)
      .map((projectile) => {
        const cells = new Map<number, Set<string>>();
        for (const [x, y, tick] of projectile.predictCells(
          level,
          level.tick_ms,
          MAX_HORIZON,
        )) {
          if (!cells.has(tick)) cells.set(tick, new Set());
          cells.get(tick)!.add(`${x},${y}`);
        }
        return { cells, active: true };
      });
  }

  private flattenProjectiles(projectiles: TimedProjectile[]): Array<[number, number, number]> {
    const result: Array<[number, number, number]> = [];
    for (const projectile of projectiles) {
      for (const [tick, cells] of projectile.cells) {
        for (const cell of cells) {
          const [x, y] = cell.split(",").map(Number);
          result.push([x, y, tick]);
        }
      }
    }
    return result;
  }

  private dangerAt(
    x: number,
    y: number,
    traps: Set<string>,
    projectiles: TimedProjectile[],
  ): number {
    for (const projectile of projectiles) {
      if (!projectile.active) continue;
      const nearby = [projectile.cells.get(1), projectile.cells.get(2)];
      if (nearby.some((cells) => cells && [...cells].some((cell) => {
        const [px, py] = cell.split(",").map(Number);
        return py === y && Math.abs(px - x) <= 2;
      }))) {
        return 2;
      }
    }
    for (const cell of traps) {
      const [tx, ty] = cell.split(",").map(Number);
      if (ty === y && Math.abs(tx - x) <= 1) return 1;
    }
    return 0;
  }

  private escapeAction(level: LevelData, x: number, y: number): number {
    const candidates = level.goal[0] >= x ? [1, 5, 2, 0, 4] : [0, 4, 2, 1, 5];
    for (const action of candidates) {
      const path = movePath(level, x, y, action);
      if (path.final[0] !== x || path.final[1] !== y) return action;
    }
    return 1;
  }
}
