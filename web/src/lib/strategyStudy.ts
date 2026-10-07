import presentation from "../../../reports/statistics_study_presentation.json";
import type { StrategyId } from "../types";

export const strategyStudyScope = presentation.scope;

export function primaryStudyStrategies<T extends { id: StrategyId }>(rows: T[]): T[] {
  const ids = new Set<string>(rows.map(row => row.id));
  const alternatives = new Set<string>(presentation.groups
    .filter(group => ids.has(group.representative))
    .flatMap(group => group.alternatives));
  return rows.filter(row => !alternatives.has(row.id));
}
