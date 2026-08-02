import { describe, expect, it } from "vitest";
import type { StrategyEfficacy, StrategyId } from "../types";
import {
  RANDOM_HITS_PER_EVALUATION,
  buildCategoryEfficacy,
} from "./categoryEfficacy";
import type { RandomBenchmarkSummary } from "./randomBenchmark";

function efficacy(strategyHits: number, evaluatedDraws = 2): StrategyEfficacy {
  const randomHits = evaluatedDraws * RANDOM_HITS_PER_EVALUATION;
  return {
    evaluatedDraws,
    strategyHits,
    randomHits,
    expectedRandomHits: randomHits,
    averageHitsPerDraw: strategyHits / evaluatedDraws,
    randomAverageHitsPerDraw: RANDOM_HITS_PER_EVALUATION,
    hitDifference: strategyHits - randomHits,
  };
}

function benchmark(meanHits: number): RandomBenchmarkSummary {
  return {
    simulations: 10_000,
    meanHits,
    lower95Hits: 0,
    upper95Hits: 8,
    sortedTotalHits: [0, 1, 2, 3, 4, 5, 6, 7, 8],
  };
}

describe("category efficacy", () => {
  it("pools enabled member results and uses a size-matched random benchmark", () => {
    const strategies = [
      { id: "freshness" as const },
      { id: "entropy" as const },
      { id: "proximity" as const },
    ];
    const efficacyByStrategy = new Map<StrategyId, StrategyEfficacy>([
      ["freshness", efficacy(3)],
      ["entropy", efficacy(1)],
      ["proximity", efficacy(2)],
    ]);
    const randomBenchmarks = new Map([
      [1, benchmark(1.5)],
      [2, benchmark(3)],
    ]);

    const rows = buildCategoryEfficacy(
      strategies,
      efficacyByStrategy,
      2,
      randomBenchmarks,
    );
    const frequency = rows.find((row) => row.id === "frequency-recency");

    expect(frequency).toMatchObject({
      strategyCount: 2,
      evaluatedDraws: 2,
      evaluations: 4,
      categoryHits: 4,
      hitsPerEvaluation: 1,
    });
    expect(frequency?.randomHits).toBeCloseTo(4 * RANDOM_HITS_PER_EVALUATION);
    expect(frequency?.randomHitsPerEvaluation).toBeCloseTo(
      RANDOM_HITS_PER_EVALUATION,
    );
    expect(frequency?.normalizedLift).toBeCloseTo(
      1 - RANDOM_HITS_PER_EVALUATION,
    );
    expect(frequency?.randomBenchmark).toBe(randomBenchmarks.get(2));
  });

  it("omits empty categories and excludes strategies not supplied as enabled", () => {
    const rows = buildCategoryEfficacy(
      [{ id: "freshness" }],
      new Map([["freshness", efficacy(2)]]),
      2,
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.strategyIds).toEqual(["freshness"]);
  });

  it("ranks by normalized efficacy and resolves ties by category name", () => {
    const strategies = [
      { id: "freshness" as const },
      { id: "entropy" as const },
      { id: "proximity" as const },
      { id: "markov100" as const },
    ];
    const efficacyByStrategy = new Map<StrategyId, StrategyEfficacy>([
      ["freshness", efficacy(2)],
      ["entropy", efficacy(2)],
      ["proximity", efficacy(2)],
      ["markov100", efficacy(1)],
    ]);

    const rows = buildCategoryEfficacy(strategies, efficacyByStrategy, 2);

    expect(rows.map((row) => row.label)).toEqual([
      "Frequency & Recency",
      "Shape & Similarity",
      "Markov & Sequence",
    ]);
  });

  it("rejects an invalid draw count", () => {
    expect(() => buildCategoryEfficacy([], new Map(), -1)).toThrow(RangeError);
  });
});
