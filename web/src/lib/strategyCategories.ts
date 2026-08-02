import type { StrategyId } from "../types";

export const STRATEGY_CATEGORIES = [
  { id: "frequency-recency", label: "Frequency & Recency" },
  { id: "shape-similarity", label: "Shape & Similarity" },
  { id: "markov-sequence", label: "Markov & Sequence" },
  {
    id: "relationships-machine-learning",
    label: "Relationships & Machine Learning",
  },
  { id: "ensembles-coverage", label: "Ensembles & Coverage" },
  { id: "random-baselines", label: "Random Baselines" },
] as const;

export type StrategyCategoryId = (typeof STRATEGY_CATEGORIES)[number]["id"];

export const STRATEGY_CATEGORY_BY_ID = {
  proximity: "shape-similarity",
  freshness: "frequency-recency",
  emd: "shape-similarity",
  randomness: "random-baselines",
  fresh_random: "random-baselines",
  chi_square: "frequency-recency",
  entropy: "frequency-recency",
  markov100: "markov-sequence",
  mkfr: "markov-sequence",
  mksp: "markov-sequence",
  mknp: "markov-sequence",
  mkrd: "markov-sequence",
  bayesian: "frequency-recency",
  predictive_grid: "shape-similarity",
  co_occurrence: "relationships-machine-learning",
  doublet_triplet_markov: "markov-sequence",
  mixed: "ensembles-coverage",
  svc: "relationships-machine-learning",
  tbl: "relationships-machine-learning",
  sklearn_svm: "relationships-machine-learning",
  lag_logistic: "relationships-machine-learning",
  sparse_neural_ticket: "relationships-machine-learning",
  cis: "ensembles-coverage",
  residual_coverage: "ensembles-coverage",
  chained: "ensembles-coverage",
} as const satisfies Record<StrategyId, StrategyCategoryId>;

export interface StrategyCategoryGroup<T> {
  id: StrategyCategoryId;
  label: string;
  strategies: T[];
}

export function groupStrategiesByCategory<T extends { id: StrategyId }>(
  strategies: readonly T[],
): StrategyCategoryGroup<T>[] {
  const strategiesByCategory = new Map<StrategyCategoryId, T[]>(
    STRATEGY_CATEGORIES.map((category) => [category.id, []]),
  );

  for (const strategy of strategies) {
    strategiesByCategory.get(STRATEGY_CATEGORY_BY_ID[strategy.id])?.push(strategy);
  }

  return STRATEGY_CATEGORIES.flatMap((category) => {
    const groupedStrategies = strategiesByCategory.get(category.id) ?? [];
    return groupedStrategies.length > 0
      ? [{ ...category, strategies: groupedStrategies }]
      : [];
  });
}
