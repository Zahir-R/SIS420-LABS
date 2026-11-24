import Phaser from "phaser";
import type { LevelData } from "../level/LevelLoader";

export type SaboteurKeys = {
  W: Phaser.Input.Keyboard.Key;
  A: Phaser.Input.Keyboard.Key;
  S: Phaser.Input.Keyboard.Key;
  D: Phaser.Input.Keyboard.Key;
  E: Phaser.Input.Keyboard.Key;
  Q: Phaser.Input.Keyboard.Key;
};

export class SaboteurController extends Phaser.GameObjects.Sprite {
  private readonly speed = 520;

  constructor(scene: Phaser.Scene, x: number, y: number) {
    super(scene, x, y, "pm-idle");
    this.setScale(2).setDepth(12).setAlpha(0.88);
    scene.add.existing(this);
    this.play("pm-idle");
  }

  update(deltaMs: number, keys: SaboteurKeys, level: LevelData): void {
    let vx = 0;
    let vy = 0;
    if (keys.A.isDown) vx -= 1;
    if (keys.D.isDown) vx += 1;
    if (keys.W.isDown) vy -= 1;
    if (keys.S.isDown) vy += 1;
    const length = Math.hypot(vx, vy) || 1;
    this.x = Phaser.Math.Clamp(
      this.x + (vx / length) * this.speed * (deltaMs / 1000),
      24,
      level.width * 64 - 24,
    );
    this.y = Phaser.Math.Clamp(
      this.y + (vy / length) * this.speed * (deltaMs / 1000),
      24,
      level.height * 64 - 24,
    );
    if (vx !== 0) {
      this.setFlipX(vx < 0);
      this.play("pm-run", true);
    } else {
      this.play("pm-idle", true);
    }
  }

  get cellX(): number {
    return Math.floor(this.x / 64);
  }

  get cellY(): number {
    return Math.floor(this.y / 64);
  }
}
