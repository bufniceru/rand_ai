import { describe, expect, it } from "vitest";
import type { StrategyId } from "../types";
import {
  STRATEGY_CATEGORIES,
  STRATEGY_CATEGORY_BY_ID,
  groupStrategiesByCategory,
} from "./strategyCategories";

const strategyIds: StrategyId[] = [
  "proximity",
  "freshness",
  "emd",
  "randomness",
  "fresh_random",
  "chi_square",
  "entropy",
  "markov100",
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
  "residual_coverage",
  "chained",
];

describe("prediction strategy categories", () => {
  it("assigns every strategy to exactly one known category", () => {
    expect(Object.keys(STRATEGY_CATEGORY_BY_ID).sort()).toEqual(
      [...strategyIds].sort(),
    );
    expect(
      Object.values(STRATEGY_CATEGORY_BY_ID).every((categoryId) =>
        STRATEGY_CATEGORIES.some((category) => category.id === categoryId),
      ),
    ).toBe(true);
  });

  it("uses category order while preserving the input efficacy order", () => {
    const rankedStrategies = [
      { id: "fresh_random" as const, rank: 1 },
      { id: "mkfr" as const, rank: 2 },
      { id: "freshness" as const, rank: 3 },
      { id: "markov100" as const, rank: 4 },
      { id: "randomness" as const, rank: 5 },
    ];

    const groups = groupStrategiesByCategory(rankedStrategies);

    expect(groups.map((group) => group.label)).toEqual([
      "Frequency & Recency",
      "Markov & Sequence",
      "Random Baselines",
    ]);
    expect(groups.map((group) => group.strategies.map((item) => item.rank))).toEqual([
      [3],
      [2, 4],
      [1, 5],
    ]);
  });
});
