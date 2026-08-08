import { describe, expect, it } from "vitest";
import type { StrategyId } from "../types";
import {
  STRATEGY_FAMILIES,
  STRATEGY_FAMILY_BY_ID,
  groupStrategiesByFamily,
} from "./strategyFamilies";

const strategyIds: StrategyId[] = [
  "proximity",
  "freshness",
  "emd",
  "randomness",
  "fresh_random",
  "chi_square",
  "categorical_chi_square",
  "entropy",
  "markov100",
  "mkgsv",
  "mkfr",
  "mksp",
  "mknp",
  "mkrd",
  "bayesian",
  "predictive_grid",
  "co_occurrence",
  "doublet_triplet_markov",
  "mixed",
  "svc",
  "tbl",
  "sklearn_svm",
  "lag_logistic",
  "sparse_neural_ticket",
  "cis",
  "decision_tree_selector",
  "residual_coverage",
  "chained",
];

describe("prediction strategy families", () => {
  it("assigns every strategy to exactly one known family", () => {
    expect(Object.keys(STRATEGY_FAMILY_BY_ID).sort()).toEqual(
      [...strategyIds].sort(),
    );
    expect(
      Object.values(STRATEGY_FAMILY_BY_ID).every((familyId) =>
        STRATEGY_FAMILIES.some((family) => family.id === familyId),
      ),
    ).toBe(true);
  });

  it("uses the supplied family rank while preserving strategy efficacy rank", () => {
    const rankedStrategies = [
      { id: "fresh_random" as const, rank: 1 },
      { id: "mkfr" as const, rank: 2 },
      { id: "freshness" as const, rank: 3 },
      { id: "markov100" as const, rank: 4 },
      { id: "randomness" as const, rank: 5 },
    ];

    const groups = groupStrategiesByFamily(rankedStrategies, [
      "random-baselines",
      "markov-sequence",
      "frequency-recency",
    ]);

    expect(groups.map((group) => group.label)).toEqual([
      "Random Baselines",
      "Markov & Sequence",
      "Frequency & Recency",
    ]);
    expect(groups.map((group) => group.strategies.map((item) => item.rank))).toEqual([
      [1, 5],
      [2, 4],
      [3],
    ]);
  });

  it("uses canonical family order while preserving plugin order by default", () => {
    const plugins = [
      { id: "fresh_random" as const },
      { id: "mkfr" as const },
      { id: "freshness" as const },
      { id: "markov100" as const },
      { id: "randomness" as const },
    ];

    const groups = groupStrategiesByFamily(plugins);

    expect(groups.map((group) => group.id)).toEqual([
      "frequency-recency",
      "markov-sequence",
      "random-baselines",
    ]);
    expect(groups.map((group) => group.strategies.map((item) => item.id))).toEqual([
      ["freshness"],
      ["mkfr", "markov100"],
      ["fresh_random", "randomness"],
    ]);
  });

  it("omits empty families and appends unranked active families", () => {
    const groups = groupStrategiesByFamily(
      [
        { id: "freshness" as const },
        { id: "proximity" as const },
      ],
      ["shape-similarity"],
    );

    expect(groups.map((group) => group.id)).toEqual([
      "shape-similarity",
      "frequency-recency",
    ]);
  });
});
