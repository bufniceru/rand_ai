import type { StrategyId } from "../types";
import { registerColorTokens, themeColor } from "./colorTemplates";

export const STRATEGY_FAMILIES = [
  {
    id: "frequency-recency",
    label: "Frequency & Recency",
    color: "#FC9867",
  },
  {
    id: "shape-similarity",
    label: "Shape & Similarity",
    color: "#A9DC76",
  },
  {
    id: "markov-sequence",
    label: "Markov & Sequence",
    color: "#AB9DF2",
  },
  {
    id: "relationships-machine-learning",
    label: "Relationships & Machine Learning",
    color: "#78DCE8",
  },
  {
    id: "ensembles-coverage",
    label: "Ensembles & Coverage",
    color: "#FF6188",
  },
  {
    id: "border-space-groups",
    label: "Border Space Groups",
    color: "#C792EA",
  },
  {
    id: "random-baselines",
    label: "Random Baselines",
    color: "#FFD866",
  },
] as const;

export type StrategyFamilyId = (typeof STRATEGY_FAMILIES)[number]["id"];

registerColorTokens(
  STRATEGY_FAMILIES.map((family) => ({
    id: `strategy.family.${family.id}`,
    label: `${family.label} family`,
    group: "strategies" as const,
    defaultValue: family.color,
    description: `Default color for the ${family.label} strategy family.`,
  })),
);

export function configuredStrategyFamilyColor(
  familyId: StrategyFamilyId,
): string {
  const family = STRATEGY_FAMILIES.find((candidate) => candidate.id === familyId);
  return themeColor(
    `strategy.family.${familyId}`,
    family?.color ?? "#727072",
  );
}

export const STRATEGY_FAMILY_BY_ID = {
  proximity: "shape-similarity",
  freshness: "frequency-recency",
  emd: "shape-similarity",
  recurrence_dynamics: "shape-similarity",
  randomness: "random-baselines",
  fresh_random: "random-baselines",
  chi_square: "frequency-recency",
  categorical_chi_square: "frequency-recency",
  entropy: "frequency-recency",
  markov100: "markov-sequence",
  mkgsv: "markov-sequence",
  mkfr: "markov-sequence",
  mksp: "markov-sequence",
  mknp: "markov-sequence",
  mkrd: "markov-sequence",
  bayesian: "frequency-recency",
  predictive_grid: "shape-similarity",
  co_occurrence: "relationships-machine-learning",
  doublet_triplet_markov: "markov-sequence",
  mixed: "ensembles-coverage",
  svc_recurrence_hybrid: "ensembles-coverage",
  svc: "relationships-machine-learning",
  tbl: "relationships-machine-learning",
  sklearn_svm: "relationships-machine-learning",
  lag_logistic: "relationships-machine-learning",
  sparse_neural_ticket: "relationships-machine-learning",
  cis: "ensembles-coverage",
  decision_tree_selector: "ensembles-coverage",
  border_group_statistical: "border-space-groups",
  border_group_markov: "border-space-groups",
  border_group_bayesian: "border-space-groups",
  border_group_ml: "border-space-groups",
  border_group_hybrid: "border-space-groups",
  residual_coverage: "ensembles-coverage",
  chained: "ensembles-coverage",
} as const satisfies Record<StrategyId, StrategyFamilyId>;

export interface StrategyFamilyGroup<T> {
  id: StrategyFamilyId;
  label: string;
  color: string;
  strategies: T[];
}

export function groupStrategiesByFamily<T extends { id: StrategyId }>(
  strategies: readonly T[],
  rankedFamilyIds: readonly StrategyFamilyId[] = STRATEGY_FAMILIES.map(
    (family) => family.id,
  ),
): StrategyFamilyGroup<T>[] {
  const strategiesByFamily = new Map<StrategyFamilyId, T[]>(
    STRATEGY_FAMILIES.map((family) => [family.id, []]),
  );

  for (const strategy of strategies) {
    strategiesByFamily.get(STRATEGY_FAMILY_BY_ID[strategy.id])?.push(strategy);
  }

  const familyById = new Map(
    STRATEGY_FAMILIES.map((family) => [family.id, family]),
  );
  const uniqueRankedIds = [...new Set(rankedFamilyIds)];
  const remainingIds = STRATEGY_FAMILIES.map((family) => family.id).filter(
    (familyId) => !uniqueRankedIds.includes(familyId),
  );

  return [...uniqueRankedIds, ...remainingIds].flatMap((familyId) => {
    const family = familyById.get(familyId);
    const groupedStrategies = strategiesByFamily.get(familyId) ?? [];
    return family && groupedStrategies.length > 0
      ? [{ ...family, color: configuredStrategyFamilyColor(family.id), strategies: groupedStrategies }]
      : [];
  });
}
