export interface PolicyData {
  policy: Record<string, number>;
  algorithm: string;
}

export type PolicyMode = "bellman" | "qlearning" | "random";

export class PolicyRunner {
  constructor(
    private readonly mode: PolicyMode,
    private readonly bellman: PolicyData,
    private readonly qlearning: PolicyData,
  ) {}

  choose(stateKey: string): number {
    if (this.mode === "random") return Math.floor(Math.random() * 6);
    const data = this.mode === "bellman" ? this.bellman : this.qlearning;
    return data.policy[stateKey] ?? 1;
  }
}
