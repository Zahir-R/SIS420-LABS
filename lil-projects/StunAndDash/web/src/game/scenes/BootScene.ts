import Phaser from "phaser";

export class BootScene extends Phaser.Scene {
  constructor() {
    super("Boot");
  }

  preload(): void {
    this.load.spritesheet("vg-idle", "/assets/virtual-guy/Idle (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("vg-run", "/assets/virtual-guy/Run (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("vg-jump", "/assets/virtual-guy/Jump (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("vg-fall", "/assets/virtual-guy/Fall (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("vg-hit", "/assets/virtual-guy/Hit (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("pm-idle", "/assets/pink-man/Idle (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("pm-run", "/assets/pink-man/Run (32x32).png", {
      frameWidth: 32,
      frameHeight: 32,
    });
    this.load.spritesheet("terrain", "/assets/terrain.png", {
      frameWidth: 16,
      frameHeight: 16,
    });
    this.load.spritesheet("saw", "/assets/saw.png", {
      frameWidth: 38,
      frameHeight: 38,
    });
    this.load.spritesheet("fire", "/assets/fire.png", {
      frameWidth: 16,
      frameHeight: 32,
    });
    this.load.image("cherries", "/assets/cherries.png");
    this.load.image("start", "/assets/start.png");
    this.load.image("goal", "/assets/goal.png");
    this.load.image("background", "/assets/background-green.png");
    const levelIds = ["level01", "level02", "level03", "level04", "level05"];
    for (const id of levelIds) {
      this.load.json(id, `/levels/${id}.json`);
      this.load.json(`bellman-${id}`, `/policies/${id}/bellman.json`);
      this.load.json(`qlearning-${id}`, `/policies/${id}/qlearning.json`);
    }
  }

  create(): void {
    this.anims.create({
      key: "vg-idle",
      frames: this.anims.generateFrameNumbers("vg-idle", { start: 0, end: 10 }),
      frameRate: 8,
      repeat: -1,
    });
    this.anims.create({
      key: "vg-run",
      frames: this.anims.generateFrameNumbers("vg-run", { start: 0, end: 11 }),
      frameRate: 12,
      repeat: -1,
    });
    this.anims.create({
      key: "vg-jump",
      frames: this.anims.generateFrameNumbers("vg-jump", { start: 0, end: 0 }),
      frameRate: 1,
      repeat: 0,
    });
    this.anims.create({
      key: "vg-fall",
      frames: this.anims.generateFrameNumbers("vg-fall", { start: 0, end: 0 }),
      frameRate: 1,
      repeat: 0,
    });
    this.anims.create({
      key: "vg-hit",
      frames: this.anims.generateFrameNumbers("vg-hit", { start: 0, end: 6 }),
      frameRate: 10,
      repeat: 0,
    });
    this.anims.create({
      key: "pm-idle",
      frames: this.anims.generateFrameNumbers("pm-idle", { start: 0, end: 10 }),
      frameRate: 8,
      repeat: -1,
    });
    this.anims.create({
      key: "pm-run",
      frames: this.anims.generateFrameNumbers("pm-run", { start: 0, end: 11 }),
      frameRate: 12,
      repeat: -1,
    });
    this.anims.create({
      key: "saw-spin",
      frames: this.anims.generateFrameNumbers("saw", { start: 0, end: 7 }),
      frameRate: 16,
      repeat: -1,
    });
    this.anims.create({
      key: "fire-burn",
      frames: this.anims.generateFrameNumbers("fire", { start: 0, end: 2 }),
      frameRate: 10,
      repeat: -1,
    });
    const autoPlay = new URLSearchParams(window.location.search).get("play");
    if (autoPlay === "bellman" || autoPlay === "qlearning" || autoPlay === "random") {
      this.scene.start("Play", { mode: autoPlay });
    } else {
      this.scene.start("Menu");
    }
  }
}
