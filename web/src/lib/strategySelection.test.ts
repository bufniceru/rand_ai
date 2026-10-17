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
  it("restores V2 as an opt-in strategy alongside V1", () => {
    const v1: StrategyPlugin = { id: "xgboost_classifier", label: "XGBoost Classifier", enabled: false };
    const v2: StrategyPlugin = { id: "xgboost_classifier_v2", label: "XGBoost V2 (Experimental)", enabled: false };
    const registry = [...plugins, v1, v2];
    const restored = JSON.parse(JSON.stringify(["xgboost_classifier_v2", "xgboost_classifier"])) as StrategyPlugin["id"][];
    expect(orderedStrategySelection(registry, restored)).toEqual([v1.id, v2.id]);
    expect(enabledStrategyPlugins(registry, restored)).toEqual([v1, v2]);
    expect(enabledStrategyPlugins(registry, ["freshness"])).not.toContainEqual(v2);
  });
  it("round-trips an opt-in XGBoost saved selection", () => {
    const xgboost: StrategyPlugin = { id: "xgboost_classifier", label: "XGBoost Classifier", enabled: false };
    const registry = [...plugins, xgboost];
    const restored = JSON.parse(JSON.stringify(["xgboost_classifier", "freshness"])) as StrategyPlugin["id"][];
    expect(orderedStrategySelection(registry, restored)).toEqual(["freshness", "xgboost_classifier"]);
    expect(enabledStrategyPlugins(registry, restored)).toContainEqual(xgboost);
    expect(enabledStrategyPlugins(registry, ["freshness"])).not.toContainEqual(xgboost);
  });
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
