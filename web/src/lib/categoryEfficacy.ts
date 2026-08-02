import type { StrategyEfficacy, StrategyId } from "../types";
import type { RandomBenchmarkSummary } from "./randomBenchmark";
import {
  groupStrategiesByCategory,
  type StrategyCategoryId,
} from "./strategyCategories";

export const RANDOM_HITS_PER_EVALUATION = 36 / 49;

export interface CategoryEfficacy {
  id: StrategyCategoryId;
  label: string;
  strategyIds: StrategyId[];
  strategyCount: number;
  evaluatedDraws: number;
  evaluations: number;
  categoryHits: number;
  randomHits: number;
  expectedRandomHits: number;
  hitsPerEvaluation: number;
  randomHitsPerEvaluation: number;
  normalizedLift: number;
  randomBenchmark: RandomBenchmarkSummary | null;
}

export function buildCategoryEfficacy(
  strategies: readonly { id: StrategyId }[],
  efficacyByStrategy: ReadonlyMap<StrategyId, StrategyEfficacy>,
  evaluatedDraws: number,
  randomBenchmarks: ReadonlyMap<number, RandomBenchmarkSummary> = new Map(),
): CategoryEfficacy[] {
  if (!Number.isInteger(evaluatedDraws) || evaluatedDraws < 0) {
    throw new RangeError("Evaluated draw count must be a non-negative integer");
  }

  return groupStrategiesByCategory(strategies)
    .map((group): CategoryEfficacy => {
      const strategyIds = group.strategies.map((strategy) => strategy.id);
      const strategyCount = strategyIds.length;
      const evaluations = strategyCount * evaluatedDraws;
      const categoryHits = strategyIds.reduce(
        (total, strategyId) =>
          total + (efficacyByStrategy.get(strategyId)?.strategyHits ?? 0),
        0,
      );
      const expectedRandomHits = evaluations * RANDOM_HITS_PER_EVALUATION;
      const randomBenchmark = randomBenchmarks.get(strategyCount) ?? null;
      const randomHits = expectedRandomHits;
      const hitsPerEvaluation = evaluations > 0 ? categoryHits / evaluations : 0;
      const randomHitsPerEvaluation =
        evaluations > 0 ? randomHits / evaluations : 0;

      return {
        id: group.id,
        label: group.label,
        strategyIds,
        strategyCount,
        evaluatedDraws,
        evaluations,
        categoryHits,
        randomHits,
        expectedRandomHits,
        hitsPerEvaluation,
        randomHitsPerEvaluation,
        normalizedLift: hitsPerEvaluation - RANDOM_HITS_PER_EVALUATION,
        randomBenchmark,
      };
    })
    .sort(
      (left, right) =>
        right.normalizedLift - left.normalizedLift ||
        right.hitsPerEvaluation - left.hitsPerEvaluation ||
        left.label.localeCompare(right.label),
    );
}
