import type {
  FamilyDrawOutcome,
  FamilyEfficiencySnapshot,
  FamilyProbability,
  MetaDraw,
  MetaDrawHistory,
  MetaForecastEvaluation,
  MetaStrategyForecast,
  StrategyFamily,
  StrategyFamilyId,
  StrategyId,
} from "../types";
import { STRATEGY_FAMILIES } from "./strategyFamilies";

export const META_STRATEGY_IDS = [
  "family_ensemble",
  "long_term_strength",
  "recent_form",
  "winner_transition",
] as const;

export type MetaStrategyId = (typeof META_STRATEGY_IDS)[number];
export type MetaAccuracyAnchor = "first" | "latest";

export const META_STRATEGY_LABELS: Record<MetaStrategyId, string> = {
  family_ensemble: "Family Ensemble",
  long_term_strength: "Long-term Strength",
  recent_form: "Recent Family Form",
  winner_transition: "Previous-winner Transition",
};

export const META_STRATEGY_DESCRIPTIONS: Record<MetaStrategyId, string> = {
  family_ensemble: "40% long-term strength · 40% recent form · 20% winner transition",
  long_term_strength: "Bayesian-shrunk historical family mean with a 12-draw prior",
  recent_form: "Exponentially weighted recent family results with a 12-draw half-life",
  winner_transition: "Smoothed transitions from the previously prevailing family",
};

const FAMILY_COLORS = new Map<StrategyFamilyId, string>(
  STRATEGY_FAMILIES.map((family) => [family.id, family.color]),
);

export interface MetaFamilyEvidenceRow {
  family: StrategyFamily;
  color: string;
  enabledStrategyIds: StrategyId[];
  snapshot: FamilyEfficiencySnapshot;
  probability: FamilyProbability | null;
  outcome: FamilyDrawOutcome | null;
  benchmark: boolean;
}

export interface MetaFamilyModelRank {
  metaStrategyId: MetaStrategyId;
  label: string;
  rank: number | null;
  probability: number | null;
}

export interface MetaAccuracySummary {
  metaStrategyId: MetaStrategyId;
  label: string;
  evaluations: number;
  topPredictionHits: number;
  topPredictionHitRate: number;
  meanWinningProbabilityMass: number;
  meanReciprocalWinnerRank: number;
  meanBrierScore: number;
}

export function familyColor(familyId: StrategyFamilyId): string {
  return FAMILY_COLORS.get(familyId) ?? "#A9B7C6";
}

export function clampMetaRecordOffset(
  records: readonly MetaDraw[],
  offset: number,
): number {
  const maximum = Math.max(0, records.length - 1);
  if (!Number.isFinite(offset)) return 0;
  return Math.min(Math.max(Math.trunc(offset), 0), maximum);
}

export function metaRecordAtOffset(
  records: readonly MetaDraw[],
  offset: number,
): MetaDraw | null {
  if (records.length === 0) return null;
  return records[records.length - 1 - clampMetaRecordOffset(records, offset)] ?? null;
}

export function metaForecast(
  record: MetaDraw | null,
  metaStrategyId: MetaStrategyId,
): MetaStrategyForecast | null {
  return (
    record?.forecasts.find(
      (forecast) => forecast.metaStrategyId === metaStrategyId,
    ) ?? null
  );
}

export function rankedFamilyProbabilities(
  history: MetaDrawHistory,
  record: MetaDraw | null,
  metaStrategyId: MetaStrategyId,
): FamilyProbability[] {
  const familyOrder = new Map(
    history.families.map((family, index) => [family.id, index]),
  );
  return [...(metaForecast(record, metaStrategyId)?.familyProbabilities ?? [])].sort(
    (left, right) =>
      left.rank - right.rank ||
      (familyOrder.get(left.familyId) ?? Number.MAX_SAFE_INTEGER) -
        (familyOrder.get(right.familyId) ?? Number.MAX_SAFE_INTEGER),
  );
}

export function buildMetaFamilyEvidence(
  history: MetaDrawHistory,
  record: MetaDraw | null,
  metaStrategyId: MetaStrategyId,
): MetaFamilyEvidenceRow[] {
  if (!record) return [];
  const enabled = new Set(history.enabledStrategyIds);
  const snapshots = new Map(
    record.familySnapshots.map((snapshot) => [snapshot.familyId, snapshot]),
  );
  const probabilities = new Map(
    rankedFamilyProbabilities(history, record, metaStrategyId).map(
      (probability) => [probability.familyId, probability],
    ),
  );
  const outcomes = new Map(
    record.familyOutcomes.map((outcome) => [outcome.familyId, outcome]),
  );
  const catalogOrder = new Map(
    history.families.map((family, index) => [family.id, index]),
  );

  return history.families
    .flatMap((family): MetaFamilyEvidenceRow[] => {
      const snapshot = snapshots.get(family.id);
      if (!snapshot) return [];
      return [
        {
          family,
          color: familyColor(family.id),
          enabledStrategyIds: family.strategyIds.filter((strategyId) =>
            enabled.has(strategyId),
          ),
          snapshot,
          probability: probabilities.get(family.id) ?? null,
          outcome: outcomes.get(family.id) ?? null,
          benchmark: !family.predictive,
        },
      ];
    })
    .sort((left, right) => {
      if (left.benchmark !== right.benchmark) return left.benchmark ? 1 : -1;
      return (
        (left.probability?.rank ?? Number.MAX_SAFE_INTEGER) -
          (right.probability?.rank ?? Number.MAX_SAFE_INTEGER) ||
        (catalogOrder.get(left.family.id) ?? Number.MAX_SAFE_INTEGER) -
          (catalogOrder.get(right.family.id) ?? Number.MAX_SAFE_INTEGER)
      );
    });
}

export function familyModelRanks(
  record: MetaDraw | null,
  familyId: StrategyFamilyId,
): MetaFamilyModelRank[] {
  return META_STRATEGY_IDS.map((metaStrategyId) => {
    const probability = metaForecast(record, metaStrategyId)?.familyProbabilities.find(
      (item) => item.familyId === familyId,
    );
    return {
      metaStrategyId,
      label: META_STRATEGY_LABELS[metaStrategyId],
      rank: probability?.rank ?? null,
      probability: probability?.probability ?? null,
    };
  });
}

export function isUniformColdStart(record: MetaDraw | null): boolean {
  if (!record || record.forecasts.length === 0) return false;
  return record.familySnapshots.every((snapshot) => snapshot.evaluatedDraws === 0);
}

export function settledMetaRecordsThrough(
  records: readonly MetaDraw[],
  selectedRecord: MetaDraw | null,
): MetaDraw[] {
  if (!selectedRecord) return [];
  return records.filter(
    (record) =>
      record.settled &&
      record.referenceDrawNumber <= selectedRecord.referenceDrawNumber,
  );
}

export function selectMetaAccuracyWindow(
  records: readonly MetaDraw[],
  selectedRecord: MetaDraw | null,
  requestedCount: number,
  anchor: MetaAccuracyAnchor,
): MetaDraw[] {
  const available = settledMetaRecordsThrough(records, selectedRecord);
  if (available.length === 0) return [];
  const normalizedCount = Number.isFinite(requestedCount)
    ? Math.trunc(requestedCount)
    : available.length;
  const count = Math.min(Math.max(normalizedCount, 1), available.length);
  return anchor === "first" ? available.slice(0, count) : available.slice(-count);
}

function evaluationFor(
  record: MetaDraw,
  metaStrategyId: MetaStrategyId,
): MetaForecastEvaluation | null {
  return (
    record.forecastEvaluations.find(
      (evaluation) => evaluation.metaStrategyId === metaStrategyId,
    ) ?? null
  );
}

export function aggregateMetaAccuracy(
  records: readonly MetaDraw[],
): MetaAccuracySummary[] {
  return META_STRATEGY_IDS.map((metaStrategyId) => {
    const evaluations = records.flatMap((record) => {
      const evaluation = evaluationFor(record, metaStrategyId);
      return evaluation ? [evaluation] : [];
    });
    const count = evaluations.length;
    const total = (selector: (evaluation: MetaForecastEvaluation) => number) =>
      evaluations.reduce((sum, evaluation) => sum + selector(evaluation), 0);
    const mean = (selector: (evaluation: MetaForecastEvaluation) => number) =>
      count > 0 ? total(selector) / count : 0;
    const topPredictionHits = total((evaluation) =>
      evaluation.topPredictionHit ? 1 : 0,
    );
    return {
      metaStrategyId,
      label: META_STRATEGY_LABELS[metaStrategyId],
      evaluations: count,
      topPredictionHits,
      topPredictionHitRate: count > 0 ? topPredictionHits / count : 0,
      meanWinningProbabilityMass: mean(
        (evaluation) => evaluation.winningProbabilityMass,
      ),
      meanReciprocalWinnerRank: mean(
        (evaluation) => evaluation.reciprocalWinnerRank,
      ),
      meanBrierScore: mean((evaluation) => evaluation.brierScore),
    };
  });
}
