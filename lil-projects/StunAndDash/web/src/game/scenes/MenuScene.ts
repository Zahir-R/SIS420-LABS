import Phaser from "phaser";
import type { PolicyMode } from "../ai/PolicyRunner";

const MODES: { mode: PolicyMode; label: string; color: number }[] = [
  { mode: "bellman", label: "BELLMAN", color: 0x8fd8ff },
  { mode: "qlearning", label: "Q-LEARNING", color: 0xff7ac8 },
  { mode: "random", label: "ALEATORIO", color: 0xffd76a },
];

export class MenuScene extends Phaser.Scene {
  constructor() {
    super("Menu");
  }

  create(): void {
    this.add
      .tileSprite(0, 0, 960, 540, "background")
      .setOrigin(0)
      .setAlpha(0.35)
      .setScrollFactor(0);
    this.add.rectangle(0, 0, 960, 540, 0x06110d, 0.72).setOrigin(0);

    this.add
      .text(480, 82, "STUN & DASH", {
        fontFamily: "monospace",
        fontSize: "46px",
        color: "#ffe16a",
      })
      .setOrigin(0.5);
    this.add
      .text(
        480,
        138,
        "El agente debe llegar a la meta. Tú colocas trampas y disparas fuego.",
        {
          fontFamily: "monospace",
          fontSize: "16px",
          color: "#d8e4cf",
          align: "center",
          wordWrap: { width: 780 },
        },
      )
      .setOrigin(0.5);

    this.add
      .text(480, 202, "ELIGE LA POLÍTICA DEL AGENTE", {
        fontFamily: "monospace",
        fontSize: "15px",
        color: "#9fb18d",
      })
      .setOrigin(0.5);

    MODES.forEach((entry, index) => {
      const x = 480 + (index - 1) * 240;
      const y = 280;
      const button = this.add
        .rectangle(x, y, 210, 74, 0x101c17, 0.95)
        .setStrokeStyle(2, entry.color)
        .setInteractive({ useHandCursor: true });
      this.add
        .text(x, y, entry.label, {
          fontFamily: "monospace",
          fontSize: "18px",
          color: Phaser.Display.Color.IntegerToColor(entry.color).rgba,
        })
        .setOrigin(0.5);
      button.on("pointerdown", () => this.scene.start("Play", { mode: entry.mode }));
    });

    this.add
      .text(
        480,
        390,
        "SABOTEADOR: WASD mover | E: trampa | Q: disparo | R: reiniciar | M: menú",
        {
          fontFamily: "monospace",
          fontSize: "14px",
          color: "#c9d5bf",
          align: "center",
          wordWrap: { width: 840 },
        },
      )
      .setOrigin(0.5);
    this.add
      .text(
        480,
        466,
        "Bellman y Q-Learning fueron entrenados en Python. La web solo ejecuta sus políticas.",
        {
          fontFamily: "monospace",
          fontSize: "13px",
          color: "#7f9382",
          align: "center",
          wordWrap: { width: 800 },
        },
      )
      .setOrigin(0.5);
  }
}
