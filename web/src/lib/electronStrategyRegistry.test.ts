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
    expect(registry.plugins.find((plugin) => plugin.id === "xgboost_classifier_v2")).toEqual({
      id: "xgboost_classifier_v2", label: "XGBoost V2 (Experimental)",
    });
    expect(registry.defaults).not.toContain("xgboost_classifier_v2");
    expect(registry.defaults).not.toContain("xgboost_classifier");
    expect(registry.defaults).toContain("cdm");
    expect(registry.defaults).not.toContain("lstm");
    expect(registry.plugins.find((plugin) => plugin.id === "lstm")).toEqual({
      id: "lstm", label: "LSTM (Experimental)",
    });
    expect(registry.plugins.find((plugin) => plugin.id === "cdm")).toEqual({
      id: "cdm", label: "CDM",
    });
  });
});

describe("CDM saved strategy preferences", () => {
  it("enables CDM for fresh settings and preserves explicit saved choices", () => {
    const start = source.indexOf("function loadStrategyPreferences() {");
    const end = source.indexOf("function saveStrategyPreferences()", start);
    const legacyStart = source.indexOf("const legacyStrategyPluginIds = [");
    const registryEnd = source.indexOf("let enabledStrategyIds =", legacyStart);
    const load = (saved: string[]) => new Function(
      "fs", "strategyPreferencesPath",
      source.slice(legacyStart, registryEnd) + source.slice(start, end)
      + "; return loadStrategyPreferences();",
    )({ readFileSync: () => JSON.stringify({ enabledStrategies: saved }) }, () => "unused") as Set<string>;
    expect(load(["bayesian"])).toEqual(new Set(["bayesian"]));
    expect(load(["cdm", "bayesian"])).toEqual(new Set(["cdm", "bayesian"]));
    expect(load([])).toEqual(new Set());
    expect(load(["lstm"])).toEqual(new Set(["lstm"]));
    expect(load(["bayesian"]).has("lstm")).toBe(false);
    const legacy = ["proximity", "freshness", "emd", "randomness", "entropy", "markov100", "mkfr", "bayesian", "svc", "tbl"];
    expect(load(legacy).has("cdm")).toBe(false);
  });
});
