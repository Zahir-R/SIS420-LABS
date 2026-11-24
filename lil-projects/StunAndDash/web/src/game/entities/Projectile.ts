import Phaser from "phaser";
import type { LevelData } from "../level/LevelLoader";

export class Projectile {
  private readonly sprite: Phaser.GameObjects.Sprite;
  private x: number;
  private y: number;
  private readonly direction: number;
  private readonly speed = 620;
  private dead = false;

  constructor(scene: Phaser.Scene, x: number, y: number, direction: number) {
    this.x = x;
    this.y = y;
    this.direction = direction >= 0 ? 1 : -1;
    this.sprite = scene.add.sprite(x, y, "fire").setScale(1.5).setDepth(9);
    this.sprite.play("fire-burn");
  }

  update(deltaMs: number, level: LevelData, agent: Phaser.GameObjects.Sprite): boolean {
    if (this.dead) return false;
    this.x += this.direction * (this.speed * deltaMs) / 1000;
    this.sprite.x = this.x;
    this.sprite.y = this.y;

    if (this.x < -40 || this.x > level.width * 64 + 40) {
      this.destroy();
      return false;
    }
    const dx = Math.abs(this.x - agent.x);
    const dy = Math.abs(this.y - agent.y);
    if (dx <= 42 && dy <= 42) {
      this.destroy();
      return true;
    }
    return false;
  }

  predictCells(
    level: LevelData,
    tickMs: number,
    horizon: number,
  ): Array<[number, number, number]> {
    const cells: Array<[number, number, number]> = [];
    const cellY = Math.floor(this.y / 64);
    for (let tick = 1; tick <= horizon; tick += 1) {
      const previousX =
        this.x + this.direction * (this.speed * tickMs * (tick - 1)) / 1000;
      const currentX =
        this.x + this.direction * (this.speed * tickMs * tick) / 1000;
      const fromCell = Math.floor(Math.min(previousX, currentX) / 64);
      const toCell = Math.floor(Math.max(previousX, currentX) / 64);
      for (let cellX = fromCell; cellX <= toCell; cellX += 1) {
        if (cellX >= 0 && cellX < level.width && cellY >= 0 && cellY < level.height) {
          cells.push([cellX, cellY, tick]);
        }
      }
    }
    return cells;
  }

  get cellX(): number {
    return Math.floor(this.x / 64);
  }

  get cellY(): number {
    return Math.floor(this.y / 64);
  }

  destroy(): void {
    if (this.dead) return;
    this.dead = true;
    this.sprite.destroy();
  }

  get active(): boolean {
    return !this.dead;
  }
}
