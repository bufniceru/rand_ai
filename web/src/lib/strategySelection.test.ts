import { describe, expect, it } from "vitest";
import type { StrategyPlugin } from "../types";
import {
  enabledStrategyPlugins,
  orderedStrategySelection,
  strategySelectionsEqual,
} from "./strategySelection";

const plugins: StrategyPlugin[] = [
  { id: "freshness", label: "Freshness", enabled: true },
  { id: "proximity", label: "Proximity", enabled: true },
  { id: "chained", label: "Chained Strategy", enabled: false },
];

describe("strategy selection", () => {
  it("shows only strategies enabled in Settings", () => {
    expect(
      enabledStrategyPlugins(plugins, ["proximity", "freshness"]),
    ).toEqual(plugins.slice(0, 2));
  });

  it("returns selected strategies in canonical plugin order", () => {
    expect(
      orderedStrategySelection(plugins, ["chained", "freshness"]),
    ).toEqual(["freshness", "chained"]);
  });

  it("ignores unknown or duplicate input IDs", () => {
    expect(
      orderedStrategySelection(plugins, [
        "freshness",
        "freshness",
        "entropy",
      ]),
    ).toEqual(["freshness"]);
  });

  it("compares selections independently of input order", () => {
    expect(
      strategySelectionsEqual(
        plugins,
        ["proximity", "freshness"],
        new Set(["freshness", "proximity"]),
      ),
    ).toBe(true);
    expect(
      strategySelectionsEqual(plugins, ["freshness"], ["proximity"]),
    ).toBe(false);
  });
});
