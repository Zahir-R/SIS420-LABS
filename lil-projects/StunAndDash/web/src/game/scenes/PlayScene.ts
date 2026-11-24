import Phaser from "phaser";
import { PolicyRunner, type PolicyData, type PolicyMode } from "../ai/PolicyRunner";
import { RouteEvaluator } from "../ai/RouteEvaluator";
import { TemporalPlanner } from "../ai/TemporalPlanner";
import { Projectile } from "../entities/Projectile";
import { SaboteurController, type SaboteurKeys } from "../entities/SaboteurController";
import {
  cellCenter,
  isWall,
  movePath,
  normalizeLevel,
  type LevelData,
} from "../level/LevelLoader";

interface Trap {
  x: number;
  y: number;
  sprite: Phaser.GameObjects.Sprite;
  active: boolean;
}

export class PlayScene extends Phaser.Scene {
  private mode: PolicyMode = "bellman";
  private level!: LevelData;
  private policy!: PolicyRunner;
  private planner = new TemporalPlanner();
  private routeEvaluator = new RouteEvaluator(this.planner);
  private agent!: Phaser.GameObjects.Sprite;
  private saboteur!: SaboteurController;
  private keys!: SaboteurKeys;
  private agentX = 0;
  private agentY = 0;
  private remaining = 0;
  private stunTicks = 0;
  private danger = 0;
  private traps: Trap[] = [];
  private projectiles: Projectile[] = [];
  private trapsLeft = 2;
  private fireballsLeft = 2;
  private trapCooldown = 0;
  private fireCooldown = 0;
  private stallRun = 0;
  private forcedActions: number[] = [];
  private gameOver = false;
  private ticker!: Phaser.Time.TimerEvent;
  private hud!: Phaser.GameObjects.Text;
  private message!: Phaser.GameObjects.Text;
  private trapButton!: Phaser.GameObjects.Text;
  private fireButton!: Phaser.GameObjects.Text;
  private sessionTimer = 0;

  constructor() {
    super("Play");
  }

  init(data: { mode?: PolicyMode }): void {
    this.mode = data.mode ?? "bellman";
  }

  create(): void {
    const rawLevel = this.cache.json.get("level") as LevelData;
    try {
      this.level = normalizeLevel(rawLevel);
    } catch (error) {
      this.add
        .text(480, 260, `Nivel inválido: ${(error as Error).message}`, {
          fontFamily: "monospace",
          fontSize: "20px",
          color: "#ff8a86",
          align: "center",
          wordWrap: { width: 760 },
        })
        .setOrigin(0.5)
        .setScrollFactor(0)
        .setDepth(100);
      return;
    }
    const bellman = this.cache.json.get("bellman") as PolicyData;
    const qlearning = this.cache.json.get("qlearning") as PolicyData;
    this.policy = new PolicyRunner(this.mode, bellman, qlearning);
    this.agentX = this.level.agent_start[0];
    this.agentY = this.level.agent_start[1];
    this.remaining = this.level.time_limit_s;
    this.stunTicks = 0;
    this.danger = 0;
    this.traps = [];
    this.projectiles = [];
    this.trapsLeft = 10;
    this.fireballsLeft = 10;
    this.trapCooldown = 0;
    this.fireCooldown = 0;
    this.stallRun = 0;
    this.forcedActions = [];
    this.gameOver = false;
    this.sessionTimer = 0;

    this.add
      .tileSprite(0, 0, 960, 540, "background")
      .setOrigin(0)
      .setScrollFactor(0)
      .setDepth(-100)
      .setAlpha(0.9);
    this.drawLevel();
    this.createSaboteur();
    this.createHud();
    this.createInput();
    this.updateCamera();
    this.ticker = this.time.addEvent({
      delay: this.level.tick_ms,
      loop: true,
      callback: this.tick,
      callbackScope: this,
    });
    this.updateHud();
  }

  update(_time: number, deltaMs: number): void {
    if (this.gameOver) return;
    this.sessionTimer += deltaMs;
    this.saboteur.update(deltaMs, this.keys, this.level);
    this.updateCamera();

    for (const projectile of this.projectiles) {
      if (!projectile.active) continue;
      if (projectile.update(deltaMs, this.level, this.agent)) {
        this.triggerStun(2);
      }
    }
    this.projectiles = this.projectiles.filter((projectile) => projectile.active);
  }

  private drawLevel(): void {
    const worldWidth = this.level.width * 64;
    const worldHeight = this.level.height * 64;
    this.cameras.main.setBounds(0, 0, worldWidth, worldHeight);

    for (let y = 0; y < this.level.height; y += 1) {
      for (let x = 0; x < this.level.width; x += 1) {
        if (this.level.rows[y][x] !== "#") continue;
        const above = y > 0 ? this.level.rows[y - 1][x] : "#";
        for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const frame = dy === 0 && above !== "#" ? 7 : 29;
            this.add
              .image(x * 64 + dx * 32, y * 64 + dy * 32, "terrain", frame)
              .setScale(2)
              .setOrigin(0)
              .setDepth(0);
          }
        }
      }
    }

    const start = cellCenter(this.level, this.level.agent_start[0], this.level.agent_start[1]);
    this.add.image(start[0], start[1] + 32, "start").setOrigin(0.5, 1).setDepth(2);
    const goal = cellCenter(this.level, this.level.goal[0], this.level.goal[1]);
    this.add.image(goal[0], goal[1] + 32, "goal").setOrigin(0.5, 1).setDepth(2);

    const [ax, ay] = cellCenter(this.level, this.agentX, this.agentY);
    this.agent = this.add.sprite(ax, ay, "vg-idle").setScale(2).setDepth(10);
    this.agent.play("vg-idle");
  }

  private createSaboteur(): void {
    const start = this.level.human_start ?? [2, 5];
    const [hx, hy] = cellCenter(this.level, start[0], start[1]);
    this.saboteur = new SaboteurController(this, hx, hy);
  }

  private createHud(): void {
    const style = {
      fontFamily: "monospace",
      fontSize: "15px",
      color: "#f2e9c8",
      backgroundColor: "#09130fdd",
      padding: { left: 8, right: 8, top: 5, bottom: 5 },
    };
    this.hud = this.add.text(10, 10, "", style).setScrollFactor(0).setDepth(100);
    this.trapButton = this.add
      .text(690, 10, "TRAMPA 2", style)
      .setScrollFactor(0)
      .setDepth(100)
      .setInteractive({ useHandCursor: true });
    this.fireButton = this.add
      .text(810, 10, "FUEGO 2", style)
      .setScrollFactor(0)
      .setDepth(100)
      .setInteractive({ useHandCursor: true });
    this.trapButton.on("pointerdown", () => this.placeTrapAtSaboteur());
    this.fireButton.on("pointerdown", () => this.fire());
    this.message = this.add
      .text(480, 470, "", {
        fontFamily: "monospace",
        fontSize: "20px",
        color: "#ffe16a",
        backgroundColor: "#09130fdd",
        padding: { left: 12, right: 12, top: 7, bottom: 7 },
      })
      .setOrigin(0.5)
      .setScrollFactor(0)
      .setDepth(100);
    this.message.setVisible(false);
    this.add
      .text(10, 510, "SABOTEADOR: WASD mover · E trampa · Q disparo", {
        fontFamily: "monospace",
        fontSize: "13px",
        color: "#9fb18d",
        backgroundColor: "#09130faa",
        padding: { left: 8, right: 8, top: 4, bottom: 4 },
      })
      .setScrollFactor(0)
      .setDepth(100);
  }

  private createInput(): void {
    this.keys = this.input.keyboard!.addKeys("W,A,S,D,E,Q") as SaboteurKeys;
    this.input.keyboard?.on("keydown-E", () => this.placeTrapAtSaboteur());
    this.input.keyboard?.on("keydown-Q", () => this.fire());
    this.input.keyboard?.on("keydown-F", () => this.fire());
    this.input.keyboard?.on("keydown-R", () => this.scene.restart({ mode: this.mode }));
    this.input.keyboard?.on("keydown-M", () => this.scene.start("Menu"));
  }

  private placeTrapAtSaboteur(): void {
    if (this.trapsLeft <= 0 || this.trapCooldown > 0 || this.gameOver) return;
    const x = this.saboteur.cellX;
    const y = this.saboteur.cellY;
    if (isWall(this.level, x, y) || this.level.rows[y][x] === "G") {
      this.flash("No puedes colocar una trampa dentro de una pared o en la meta.");
      return;
    }
    if (this.traps.some((trap) => trap.active && trap.x === x && trap.y === y)) return;
    const [cx, cy] = cellCenter(this.level, x, y);
    const sprite = this.add.sprite(cx, cy, "saw").setScale(1.4).setDepth(7);
    sprite.play("saw-spin");
    const trap: Trap = { x, y, sprite, active: true };
    this.traps.push(trap);
    this.trapsLeft -= 1;
    this.trapCooldown = 0.4;
    this.time.delayedCall(7000, () => {
      trap.active = false;
      sprite.destroy();
    });
    this.flash("Trampa colocada.");
    this.updateHud();
  }

  private fire(): void {
    if (this.fireballsLeft <= 0 || this.fireCooldown > 0 || this.gameOver) return;
    this.fireballsLeft -= 0;
    this.fireCooldown = 0.01;
    const pointer = this.input.activePointer;
    const direction = pointer.worldX >= this.saboteur.x ? 1 : -1;
    this.projectiles.push(
      new Projectile(this, this.saboteur.x, this.saboteur.y, direction),
    );
    this.updateHud();
  }

  private tick(): void {
    if (this.gameOver) return;
    this.remaining = Math.max(0, this.remaining - this.level.tick_ms / 1000);
    this.trapCooldown = Math.max(0, this.trapCooldown - this.level.tick_ms / 1000);
    this.fireCooldown = Math.max(0, this.fireCooldown - this.level.tick_ms / 1000);
    if (this.remaining <= 0) {
      this.finish(false);
      return;
    }

    this.danger = this.computeDanger();
    if (this.stunTicks > 0) {
      this.stunTicks -= 1;
      this.updateHud();
      return;
    }

    const [chosenAction, planLabel] = this.chooseAction();
    let action = chosenAction;
    let path = movePath(this.level, this.agentX, this.agentY, action);
    let stalled = path.final[0] === this.agentX && path.final[1] === this.agentY;
    if (stalled && this.stallRun >= 4) {
      action = this.escapeAction();
      path = movePath(this.level, this.agentX, this.agentY, action);
      stalled = path.final[0] === this.agentX && path.final[1] === this.agentY;
    }
    this.stallRun = stalled ? this.stallRun + 1 : 0;
    const changed = path.final[0] !== this.agentX || path.final[1] !== this.agentY;
    const hitIndex = path.cells.findIndex(([x, y]) =>
      this.traps.some((trap) => trap.active && trap.x === x && trap.y === y),
    );
    const visiblePath = hitIndex >= 0 ? path.cells.slice(0, hitIndex + 1) : path.cells;
    const destination = hitIndex >= 0 ? path.cells[hitIndex] : path.final;
    this.agentX = destination[0];
    this.agentY = destination[1];
    this.agent.play(action === 2 || action === 4 || action === 5 ? "vg-jump" : changed ? "vg-run" : "vg-idle");
    this.animatePath(visiblePath);
    if (hitIndex >= 0) {
      const trap = this.traps.find(
        (candidate) => candidate.active && candidate.x === destination[0] && candidate.y === destination[1],
      );
      if (trap) {
        trap.active = false;
        trap.sprite.destroy();
      }
      this.triggerStun(1);
    }
    if (this.agentX === this.level.goal[0] && this.agentY === this.level.goal[1]) {
      this.finish(true);
      return;
    }
    this.message.setText(planLabel);
    this.message.setVisible(true);
    this.time.delayedCall(700, () => {
      if (!this.gameOver) {
        this.message.setText("");
        this.message.setVisible(false);
      }
    });
    this.updateHud();
  }


  private animatePath(cells: Array<[number, number]>): void {
    if (cells.length === 0) return;
    this.tweens.killTweensOf(this.agent);
    const duration = Math.max(24, (this.level.tick_ms * 0.72) / cells.length);
    let index = 0;
    const next = () => {
      if (index >= cells.length) return;
      const [x, y] = cells[index];
      index += 1;
      const [px, py] = cellCenter(this.level, x, y);
      this.tweens.add({
        targets: this.agent,
        x: px,
        y: py,
        duration,
        ease: "Linear",
        onComplete: next,
      });
    };
    next();
  }

  private chooseAction(): [number, string] {
    const activeTraps = this.traps.filter((trap) => trap.active);
    const activeProjectiles = this.projectiles.filter((projectile) => projectile.active);

    if (this.forcedActions.length > 0) {
      const next = this.forcedActions[0];
      if (this.isSafeAction(next, activeTraps, activeProjectiles)) {
        this.forcedActions.shift();
        return [next, "FORCED DETOUR"];
      }
      this.forcedActions = [];
      const rebuilt = this.findForcedDetour(activeTraps, activeProjectiles);
      if (rebuilt && rebuilt.length > 0) {
        this.forcedActions = rebuilt.slice(1);
        return [rebuilt[0], "FORCED DETOUR"];
      }
      return [this.escapeAction(), "FORCED HIT"];
    }

    if (this.stallRun >= 4) {
      const forced = this.findForcedDetour(activeTraps, activeProjectiles);
      if (forced && forced.length > 0) {
        this.forcedActions = forced.slice(1);
        return [forced[0], "FORCED DETOUR"];
      }
      return [this.escapeAction(), "FORCED HIT"];
    }

    if (activeTraps.length > 0 || activeProjectiles.length > 0) {
      const trapCells: Array<[number, number]> = activeTraps.map((trap) => [trap.x, trap.y]);
      const decision = this.routeEvaluator.evaluate(
        this.level,
        this.agentX,
        this.agentY,
        this.remaining,
        trapCells,
        activeProjectiles,
        (key) => this.policy.choose(key),
      );
      if (decision) {
        if (decision.mode === "detour" && decision.detourActions.length > 1) {
          this.forcedActions = decision.detourActions.slice(1);
        }
        const returnLabel =
          decision.mode === "detour"
            ? `DETOUR  Q=${this.formatReturn(decision.detourReturn)} vs ${this.formatReturn(decision.policyReturn)}`
            : `POLICY  Q=${this.formatReturn(decision.policyReturn)} vs ${this.formatReturn(decision.detourReturn)}`;
        return [decision.action, returnLabel];
      }
      return [this.escapeAction(), "NO ROUTE"];
    }

    const timeBucket = Math.min(
      7,
      Math.max(0, Math.floor((this.remaining / this.level.time_limit_s) * 7)),
    );
    const key = `${this.agentX},${this.agentY},${this.stunTicks},${timeBucket},${this.danger}`;
    return [this.policy.choose(key), "POLICY"];
  }

  private findForcedDetour(
    activeTraps: Trap[],
    activeProjectiles: Projectile[],
  ): number[] | null {
    const trapCells: Array<[number, number]> = activeTraps.map((trap) => [trap.x, trap.y]);
    const projectileCells = activeProjectiles.flatMap((projectile) =>
      projectile.predictCells(this.level, this.level.tick_ms, 36),
    );
    const safe = this.planner.findSafeNoStall(
      this.level,
      this.agentX,
      this.agentY,
      trapCells,
      projectileCells,
      36,
    );
    return safe && safe.actions.length > 0 ? safe.actions : null;
  }

  private isSafeAction(
    action: number,
    activeTraps: Trap[],
    activeProjectiles: Projectile[],
  ): boolean {
    const path = movePath(this.level, this.agentX, this.agentY, action);
    if (path.final[0] === this.agentX && path.final[1] === this.agentY) {
      return false;
    }
    const cells: Array<[number, number]> = [[this.agentX, this.agentY], ...path.cells];
    if (
      activeTraps.some((trap) =>
        cells.some(([x, y]) => trap.x === x && trap.y === y),
      )
    ) {
      return false;
    }
    const nextProjectileCells = activeProjectiles.flatMap((projectile) =>
      projectile.predictCells(this.level, this.level.tick_ms, 1),
    );
    return !nextProjectileCells.some(([x, y]) =>
      cells.some(([cx, cy]) => cx === x && cy === y),
    );
  }

  private formatReturn(value: number): string {
    return Number.isFinite(value) ? value.toFixed(1) : "—";
  }

  private computeDanger(): number {
    if (
      this.projectiles.some(
        (projectile) =>
          projectile.active &&
          projectile.cellY === this.agentY &&
          Math.abs(projectile.cellX - this.agentX) <= 2,
      )
    ) {
      return 2;
    }
    const trapNear = this.traps.some(
      (trap) => trap.active && trap.y === this.agentY && Math.abs(trap.x - this.agentX) <= 1,
    );
    return trapNear ? 1 : 0;
  }

  private escapeAction(): number {
    const candidates = this.level.goal[0] >= this.agentX ? [1, 5, 2, 0, 4] : [0, 4, 2, 1, 5];
    for (const action of candidates) {
      const path = movePath(this.level, this.agentX, this.agentY, action);
      if (path.final[0] !== this.agentX || path.final[1] !== this.agentY) {
        return action;
      }
    }
    return 1;
  }

  private updateCamera(): void {
    const worldWidth = this.level.width * 64;
    const worldHeight = this.level.height * 64;
    const centerX = Phaser.Math.Clamp(
      (this.agent.x + this.saboteur.x) / 2,
      480,
      worldWidth - 480,
    );
    const centerY = Phaser.Math.Clamp(
      (this.agent.y + this.saboteur.y) / 2,
      270,
      worldHeight - 270,
    );
    this.cameras.main.centerOn(centerX, centerY);
  }

  private triggerStun(kind: number): void {
    if (this.gameOver) return;
    this.stunTicks = 2;
    this.danger = kind;
    this.agent.play("vg-hit");
    this.cameras.main.shake(120, 0.008);
    this.flash(kind === 1 ? "¡TRAMPA! STUN 0.5s" : "¡FUEGO! STUN 0.5s");
    this.updateHud();
  }

  private flash(text: string): void {
    this.message.setText(text);
    this.message.setVisible(true);
    this.time.delayedCall(1000, () => {
      if (!this.gameOver) {
        this.message.setText("");
        this.message.setVisible(false);
      }
    });
  }

  private finish(agentWon: boolean): void {
    if (this.gameOver) return;
    this.gameOver = true;
    this.ticker.paused = true;
    this.add.rectangle(0, 0, 960, 540, 0x06110d, 0.72).setOrigin(0).setScrollFactor(0).setDepth(200);
    this.add
      .text(480, 220, agentWon ? "AGENTE GANA" : "JUGADOR GANA", {
        fontFamily: "monospace",
        fontSize: "42px",
        color: agentWon ? "#8fffb0" : "#ff8a86",
      })
      .setOrigin(0.5)
      .setScrollFactor(0)
      .setDepth(201);
    this.add
      .text(480, 290, agentWon ? "Llegó a la meta antes del tiempo límite." : "El agente no llegó a tiempo.", {
        fontFamily: "monospace",
        fontSize: "18px",
        color: "#e8e2c8",
      })
      .setOrigin(0.5)
      .setScrollFactor(0)
      .setDepth(201);
    this.add
      .text(480, 350, "R: reiniciar | M: menú", {
        fontFamily: "monospace",
        fontSize: "16px",
        color: "#9fb18d",
      })
      .setOrigin(0.5)
      .setScrollFactor(0)
      .setDepth(201);
  }

  private updateHud(): void {
    this.hud.setText(
      `AGENTE ${this.mode.toUpperCase()}  T ${this.remaining.toFixed(1)}s  STUN ${this.stunTicks}  PLAN ${this.danger > 0 ? "ALERTA" : "AUTO"}`,
    );
    this.trapButton.setText(`TRAMPA ${this.trapsLeft}`);
    this.fireButton.setText(`FUEGO ${this.fireballsLeft}`);
  }
}
