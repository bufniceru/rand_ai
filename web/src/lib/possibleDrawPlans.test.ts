import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  activePossibleDrawState,
  configurePossibleDrawContext,
  cyclePossibleDrawNumberState,
  getDrawPortfolioMode,
  getPossibleDrawNumberState,
  resetPossibleDrawStoreForTests,
  setDrawPortfolioMode,
  setPossibleDrawNumberState,
  togglePossibleDrawExcluded,
} from "./possibleDrawPlans";

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();
  get length(): number { return this.values.size; }
  clear(): void { this.values.clear(); }
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  key(index: number): string | null { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string): void { this.values.delete(key); }
  setItem(key: string, value: string): void { this.values.set(key, value); }
}

let storage: MemoryStorage;

beforeEach(() => {
  storage = new MemoryStorage();
  vi.stubGlobal("window", { localStorage: storage });
  resetPossibleDrawStoreForTests();
});

describe("Possible Draw plan store", () => {
  it("cycles neutral, Candidate, and Fixed while Excluded uses its explicit toggle", () => {
    configurePossibleDrawContext({ datasetId: "one", targetDrawId: "101" });
    expect(getPossibleDrawNumberState(7)).toBe("neutral");
    cyclePossibleDrawNumberState(7);
    expect(getPossibleDrawNumberState(7)).toBe("candidate");
    cyclePossibleDrawNumberState(7);
    expect(getPossibleDrawNumberState(7)).toBe("fixed");
    cyclePossibleDrawNumberState(7);
    expect(getPossibleDrawNumberState(7)).toBe("neutral");
    togglePossibleDrawExcluded(7);
    expect(getPossibleDrawNumberState(7)).toBe("excluded");
    cyclePossibleDrawNumberState(7);
    expect(getPossibleDrawNumberState(7)).toBe("excluded");
    togglePossibleDrawExcluded(7);
    expect(getPossibleDrawNumberState(7)).toBe("neutral");
  });

  it("enforces six Fixed numbers without removing the seventh Candidate", () => {
    configurePossibleDrawContext({ datasetId: "one", targetDrawId: "101" });
    for (let number = 1; number <= 6; number += 1) {
      expect(setPossibleDrawNumberState(number, "fixed").ok).toBe(true);
    }
    setPossibleDrawNumberState(7, "candidate");
    const result = setPossibleDrawNumberState(7, "fixed");
    expect(result.ok).toBe(false);
    expect(getPossibleDrawNumberState(7)).toBe("candidate");
  });

  it("isolates plans by dataset and target and modes by dataset", () => {
    configurePossibleDrawContext({ datasetId: "one", targetDrawId: "101" });
    setPossibleDrawNumberState(3, "fixed");
    setDrawPortfolioMode("one", "guided");
    configurePossibleDrawContext({ datasetId: "one", targetDrawId: "102" });
    expect(activePossibleDrawState.value.fixedNumbers).toEqual([]);
    expect(getDrawPortfolioMode("one")).toBe("guided");
    configurePossibleDrawContext({ datasetId: "two", targetDrawId: "101" });
    expect(getDrawPortfolioMode("two")).toBe("classic");
    configurePossibleDrawContext({ datasetId: "one", targetDrawId: "101" });
    expect(activePossibleDrawState.value.fixedNumbers).toEqual([3]);
  });

  it("migrates the v2 selected, uncertain, and dropped fields", () => {
    storage.setItem("rand-ai.possible-draw.plans.v2", JSON.stringify({
      activePlanId: "legacy",
      plans: [{
        id: "legacy",
        name: "Legacy draw",
        selected: [4, 5],
        uncertain: [6, 4],
        dropped: [7, 6],
      }],
    }));
    configurePossibleDrawContext({ datasetId: "one", targetDrawId: "101" });
    expect(activePossibleDrawState.value).toEqual({
      fixedNumbers: [4, 5],
      candidateNumbers: [6],
      excludedNumbers: [7],
    });
  });
});
