import source from "../../electron/main.cjs?raw";
import { describe, expect, it } from "vitest";
import { STRATEGY_FAMILY_BY_ID } from "./strategyFamilies";

describe("Electron Settings strategy registry", () => {
  it("exposes all renderer strategies and keeps XGBoost opt-in", () => {
    const start = source.indexOf("const strategyPlugins = [");
    const defaults = source.indexOf("const defaultStrategyPluginIds =", start);
    const end = source.indexOf(";", defaults) + 1;
    expect(start).toBeGreaterThan(-1);
    expect(defaults).toBeGreaterThan(start);
    const registry = new Function(
      source.slice(start, end) + "; return { plugins: strategyPlugins, defaults: defaultStrategyPluginIds };",
    )() as { plugins: { id: string; label: string }[]; defaults: string[] };
    expect(registry.plugins.map((plugin) => plugin.id).sort()).toEqual(Object.keys(STRATEGY_FAMILY_BY_ID).sort());
    expect(registry.plugins.find((plugin) => plugin.id === "xgboost_classifier")).toEqual({
      id: "xgboost_classifier", label: "XGBoost Classifier",
    });
    expect(registry.defaults).not.toContain("xgboost_classifier");
  });
});
