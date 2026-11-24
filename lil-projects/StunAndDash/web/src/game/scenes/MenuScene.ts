import Phaser from "phaser";
import type { PolicyMode } from "../ai/PolicyRunner";

const LEVELS = ["level01", "level02", "level03", "level04", "level05"];
const MODES: { mode: PolicyMode; label: string; color: number }[] = [
  { mode: "bellman", label: "BELLMAN", color: 0x8fd8ff },
  { mode: "qlearning", label: "Q-LEARNING", color: 0xff7ac8 },
  { mode: "random", label: "ALEATORIO", color: 0xffd76a },
];

export class MenuScene extends Phaser.Scene {
  private selectedLevel = 0;
  private levelButtons: Phaser.GameObjects.Rectangle[] = [];

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
      .text(480, 48, "STUN & DASH", {
        fontFamily: "monospace",
        fontSize: "42px",
        color: "#ffe16a",
      })
      .setOrigin(0.5);
    this.add
      .text(480, 94, "El agente debe llegar a la meta. Tú colocas trampas y disparas fuego.", {
        fontFamily: "monospace",
        fontSize: "15px",
        color: "#d8e4cf",
        align: "center",
        wordWrap: { width: 820 },
      })
      .setOrigin(0.5);

    this.add
      .text(480, 142, "NIVEL", {
        fontFamily: "monospace",
        fontSize: "14px",
        color: "#9fb18d",
      })
      .setOrigin(0.5);
    LEVELS.forEach((id, index) => {
      const x = 480 + (index - 2) * 130;
      const button = this.add
        .rectangle(x, 182, 112, 44, 0x101c17, 0.95)
        .setStrokeStyle(2, 0x4b5b48)
        .setInteractive({ useHandCursor: true });
      this.add
        .text(x, 182, `N${index + 1}`, {
          fontFamily: "monospace",
          fontSize: "18px",
          color: "#d8e4cf",
        })
        .setOrigin(0.5);
      button.on("pointerdown", () => this.selectLevel(index));
      this.levelButtons.push(button);
      void id;
    });

    this.add
      .text(480, 236, "ELIGE LA POLÍTICA DEL AGENTE", {
        fontFamily: "monospace",
        fontSize: "14px",
        color: "#9fb18d",
      })
      .setOrigin(0.5);
    MODES.forEach((entry, index) => {
      const x = 480 + (index - 1) * 260;
      const button = this.add
        .rectangle(x, 292, 230, 64, 0x101c17, 0.95)
        .setStrokeStyle(2, entry.color)
        .setInteractive({ useHandCursor: true });
      this.add
        .text(x, 292, entry.label, {
          fontFamily: "monospace",
          fontSize: "18px",
          color: Phaser.Display.Color.IntegerToColor(entry.color).rgba,
        })
        .setOrigin(0.5);
      button.on("pointerdown", () =>
        this.scene.start("Play", {
          mode: entry.mode,
          level: LEVELS[this.selectedLevel],
        }),
      );
    });

    this.add
      .text(
        480,
        382,
        "SABOTEADOR: WASD mover | E: trampa | Q: disparo | R: reiniciar | M: menú",
        {
          fontFamily: "monospace",
          fontSize: "14px",
          color: "#c9d5bf",
          align: "center",
          wordWrap: { width: 860 },
        },
      )
      .setOrigin(0.5);
    this.add
      .text(
        480,
        452,
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

    this.selectLevel(0);
  }

  private selectLevel(index: number): void {
    this.selectedLevel = index;
    this.levelButtons.forEach((button, current) => {
      button.setStrokeStyle(2, current === index ? 0xffe16a : 0x4b5b48);
    });
  }
}
