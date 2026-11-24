export interface LevelData {
  id: string;
  name: string;
  width: number;
  height: number;
  rows: string[];
  agent_start: number[];
  human_start?: number[];
  goal: number[];
  time_limit_s: number;
  tick_ms: number;
  jump_height: number;
  trap_slots: number[][];
  fire_lanes: number[];
}

export function isWall(level: LevelData, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= level.width || y >= level.height) return true;
  return level.rows[y][x] === "#";
}

export function hasFloor(level: LevelData, x: number, y: number): boolean {
  return isWall(level, x, y + 1);
}

export function gravity(level: LevelData, x: number, y: number): [number, number] {
  while (y + 1 < level.height && !hasFloor(level, x, y)) y += 1;
  return [x, y];
}

export interface MovePath {
  cells: Array<[number, number]>;
  final: [number, number];
  blocked: boolean;
}

export function gravityPath(level: LevelData, x: number, y: number): Array<[number, number]> {
  const cells: Array<[number, number]> = [];
  let cy = y;
  while (cy + 1 < level.height && !hasFloor(level, x, cy)) {
    cy += 1;
    cells.push([x, cy]);
  }
  return cells;
}

export function movePath(
  level: LevelData,
  x: number,
  y: number,
  action: number,
): MovePath {
  const jump = Math.max(1, level.jump_height);
  let destination: [number, number] = [x, y];
  let cells: Array<[number, number]> = [];
  let blocked = false;

  if (action === 0 || action === 1) {
    const dx = action === 0 ? -1 : 1;
    if (isWall(level, x + dx, y)) {
      blocked = true;
    } else {
      destination = [x + dx, y];
      cells = [destination];
    }
  } else if (action === 2) {
    for (let step = 1; step <= jump; step += 1) {
      cells.push([x, y - step]);
    }
    if (cells.some(([cx, cy]) => isWall(level, cx, cy))) {
      blocked = true;
      cells = [];
    } else {
      destination = [x, y - jump];
    }
  } else if (action === 4 || action === 5) {
    const dx = action === 4 ? -1 : 1;
    // Diagonal: siempre vertical sobre el origen + diagonal + destino.
    cells.push([x, y - 1]);
    for (let step = 2; step < jump; step += 1) {
      cells.push([x, y - step]);
    }
    for (let step = 1; step <= jump; step += 1) {
      cells.push([x + dx, y - step]);
    }
    if (cells.some(([cx, cy]) => isWall(level, cx, cy))) {
      blocked = true;
      cells = [];
    } else {
      destination = [x + dx, y - jump];
    }
  }

  if (blocked) {
    const fall = gravityPath(level, x, y);
    return {
      cells: fall,
      final: fall.length > 0 ? fall[fall.length - 1] : [x, y],
      blocked: true,
    };
  }

  const fall = gravityPath(level, destination[0], destination[1]);
  const full = [...cells, ...fall];
  return {
    cells: full,
    final: full.length > 0 ? full[full.length - 1] : destination,
    blocked: false,
  };
}

export function moveAgent(
  level: LevelData,
  x: number,
  y: number,
  action: number,
): [number, number] {
  return movePath(level, x, y, action).final;
}

export function cellCenter(level: LevelData, x: number, y: number): [number, number] {
  return [x * 64 + 32, y * 64 + 32];
}


export function normalizeLevel(raw: LevelData): LevelData {
  if (raw.rows.length !== raw.height) {
    throw new Error("El nivel no coincide con height");
  }
  if (raw.rows.some((row) => row.length !== raw.width)) {
    throw new Error("El nivel no coincide con width");
  }
  const starts: Array<[number, number]> = [];
  const goals: Array<[number, number]> = [];
  for (let y = 0; y < raw.height; y += 1) {
    for (let x = 0; x < raw.width; x += 1) {
      if (raw.rows[y][x] === "S") starts.push([x, y]);
      if (raw.rows[y][x] === "G") goals.push([x, y]);
    }
  }
  if (starts.length !== 1) throw new Error("El nivel debe tener exactamente una S");
  if (goals.length !== 1) throw new Error("El nivel debe tener exactamente una G");
  const human = raw.human_start ?? starts[0];
  if (
    human[0] < 0 ||
    human[1] < 0 ||
    human[0] >= raw.width ||
    human[1] >= raw.height ||
    raw.rows[human[1]][human[0]] === "#"
  ) {
    throw new Error("human_start debe estar dentro del nivel y fuera de una pared");
  }
  if (raw.time_limit_s <= 0 || raw.tick_ms <= 0) {
    throw new Error("time_limit_s y tick_ms deben ser positivos");
  }
  const normalized: LevelData = {
    ...raw,
    agent_start: starts[0],
    goal: goals[0],
    human_start: human,
  };
  if (!isGoalReachable(normalized)) {
    throw new Error("La meta no es alcanzable desde S con las reglas actuales");
  }
  return normalized;
}

export function isGoalReachable(level: LevelData): boolean {
  const start: [number, number] = [level.agent_start[0], level.agent_start[1]];
  const queue: Array<[number, number]> = [start];
  const seen = new Set<string>([start.join(",")]);
  while (queue.length > 0) {
    const [x, y] = queue.shift()!;
    if (x === level.goal[0] && y === level.goal[1]) return true;
    for (let action = 0; action < 6; action += 1) {
      const final = movePath(level, x, y, action).final;
      const key = final.join(",");
      if (!seen.has(key)) {
        seen.add(key);
        queue.push(final);
      }
    }
  }
  return false;
}
