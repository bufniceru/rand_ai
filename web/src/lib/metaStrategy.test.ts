import { describe, expect, it } from "vitest";
import type {
  MetaDraw,
  MetaDrawHistory,
  MetaForecastEvaluation,
  MetaStrategyForecast,
} from "../types";
import {
  META_STRATEGY_IDS,
  aggregateMetaAccuracy,
  buildMetaFamilyEvidence,
  clampMetaRecordOffset,
  familyModelRanks,
  isUniformColdStart,
  metaRecordAtOffset,
  rankedFamilyProbabilities,
  selectMetaAccuracyWindow,
  settledMetaRecordsThrough,
} from "./metaStrategy";

function forecast(
  metaStrategyId: string,
  frequencyProbability: number,
): MetaStrategyForecast {
  const shapeProbability = 1 - frequencyProbability;
  const frequencyFirst = frequencyProbability >= shapeProbability;
  return {
    metaStrategyId,
    predictedFamilyId: frequencyFirst
      ? "frequency-recency"
      : "shape-similarity",
    familyProbabilities: [
      {
        familyId: "shape-similarity",
        rank: frequencyFirst ? 2 : 1,
        rawScore: shapeProbability,
        probability: shapeProbability,
      },
      {
        familyId: "frequency-recency",
        rank: frequencyFirst ? 1 : 2,
        rawScore: frequencyProbability,
        probability: frequencyProbability,
      },
    ],
  };
}

function evaluation(
  metaStrategyId: string,
  topPredictionHit: boolean,
  winningProbabilityMass: number,
): MetaForecastEvaluation {
  return {
    metaStrategyId,
    topPredictionHit,
    winningProbabilityMass,
    reciprocalWinnerRank: topPredictionHit ? 1 : 0.5,
    brierScore: topPredictionHit ? 0.18 : 0.68,
  };
}

function record(
  referenceDrawNumber: number,
  settled: boolean,
  frequencyProbability: number,
): MetaDraw {
  return {
    referenceDrawNumber,
    targetDrawNumber: referenceDrawNumber + 1,
    referenceDate: null,
    targetDate: settled ? `2026-01-0${referenceDrawNumber + 1}` : null,
    settled,
    familySnapshots: [
      {
        familyId: "frequency-recency",
        evaluatedDraws: referenceDrawNumber - 1,
        evaluations: referenceDrawNumber - 1,
        cumulativeHits: referenceDrawNumber - 1,
        meanHitsPerStrategy: referenceDrawNumber > 1 ? 1 : 0,
        recentEwmaHitsPerStrategy: 0.8,
        normalizedLift: 0.065,
        winShare: 0.5,
        volatility: 0.2,
        drawsSinceWin: 0,
      },
      {
        familyId: "shape-similarity",
        evaluatedDraws: referenceDrawNumber - 1,
        evaluations: referenceDrawNumber - 1,
        cumulativeHits: 0,
        meanHitsPerStrategy: 0,
        recentEwmaHitsPerStrategy: 0.6,
        normalizedLift: -0.135,
        winShare: 0.25,
        volatility: 0.1,
        drawsSinceWin: 1,
      },
      {
        familyId: "random-baselines",
        evaluatedDraws: referenceDrawNumber - 1,
        evaluations: referenceDrawNumber - 1,
        cumulativeHits: 1,
        meanHitsPerStrategy: 1,
        recentEwmaHitsPerStrategy: 0.75,
        normalizedLift: 0.265,
        winShare: 0,
        volatility: 0.3,
        drawsSinceWin: null,
      },
    ],
    forecasts: META_STRATEGY_IDS.map((id, index) =>
      forecast(id, Math.min(0.9, frequencyProbability + index * 0.01)),
    ),
    actualNumbers: settled ? [1, 2, 3, 4, 5, 6] : [],
    familyOutcomes: settled
      ? [
          {
            familyId: "frequency-recency",
            memberHits: { freshness: 2 },
            strategyCount: 1,
            totalHits: 2,
            meanHitsPerStrategy: 2,
            normalizedLift: 1.265,
            rank: 1,
            prevailing: true,
          },
          {
            familyId: "shape-similarity",
            memberHits: { proximity: 1 },
            strategyCount: 1,
            totalHits: 1,
            meanHitsPerStrategy: 1,
            normalizedLift: 0.265,
            rank: 2,
            prevailing: false,
          },
          {
            familyId: "random-baselines",
            memberHits: { randomness: 3 },
            strategyCount: 1,
            totalHits: 3,
            meanHitsPerStrategy: 3,
            normalizedLift: 2.265,
            rank: 0,
            prevailing: false,
          },
        ]
      : [],
    prevailingFamilyIds: settled ? ["frequency-recency"] : [],
    forecastEvaluations: settled
      ? META_STRATEGY_IDS.map((id, index) =>
          evaluation(id, index !== 2, 0.6 + index * 0.05),
        )
      : [],
  };
}

const history: MetaDrawHistory = {
  schemaVersion: 1,
  familyCatalogVersion: 1,
  strategySetFingerprint: "fingerprint",
  enabledStrategyIds: ["freshness", "proximity", "randomness"],
  families: [
    {
      id: "frequency-recency",
      label: "Frequency & Recency",
      strategyIds: ["freshness", "entropy"],
      predictive: true,
    },
    {
      id: "shape-similarity",
      label: "Shape & Similarity",
      strategyIds: ["proximity"],
      predictive: true,
    },
    {
      id: "random-baselines",
      label: "Random Baselines",
      strategyIds: ["randomness"],
      predictive: false,
    },
  ],
  records: [record(1, true, 0.6), record(2, true, 0.4), record(3, false, 0.7)],
};

describe("Meta Strategy view models", () => {
  it("clamps independent history navigation and defaults to the latest record", () => {
    expect(clampMetaRecordOffset(history.records, -3)).toBe(0);
    expect(clampMetaRecordOffset(history.records, 99)).toBe(2);
    expect(clampMetaRecordOffset(history.records, Number.NaN)).toBe(0);
    expect(metaRecordAtOffset(history.records, 0)?.referenceDrawNumber).toBe(3);
    expect(metaRecordAtOffset(history.records, 1)?.referenceDrawNumber).toBe(2);
    expect(metaRecordAtOffset([], 0)).toBeNull();
  });

  it("orders probabilities deterministically and keeps the benchmark out", () => {
    const probabilities = rankedFamilyProbabilities(
      history,
      history.records[0],
      "family_ensemble",
    );
    expect(probabilities.map((item) => item.familyId)).toEqual([
      "frequency-recency",
      "shape-similarity",
    ]);
    expect(probabilities.map((item) => item.probability)).toEqual([0.6, 0.4]);

    const evidence = buildMetaFamilyEvidence(
      history,
      history.records[0],
      "family_ensemble",
    );
    expect(evidence.map((row) => row.family.id)).toEqual([
      "frequency-recency",
      "shape-similarity",
      "random-baselines",
    ]);
    expect(evidence[0].enabledStrategyIds).toEqual(["freshness"]);
    expect(evidence[2].benchmark).toBe(true);
    expect(evidence[2].probability).toBeNull();
  });

  it("builds family rank comparisons for all four models", () => {
    const ranks = familyModelRanks(history.records[1], "shape-similarity");
    expect(ranks).toHaveLength(4);
    expect(ranks.every((rank) => rank.rank === 1)).toBe(true);
    expect(familyModelRanks(null, "shape-similarity").every((rank) => rank.rank === null)).toBe(
      true,
    );
  });

  it("recognizes cold starts and handles missing records", () => {
    expect(isUniformColdStart(history.records[0])).toBe(true);
    expect(isUniformColdStart(history.records[1])).toBe(false);
    expect(isUniformColdStart(null)).toBe(false);
    expect(buildMetaFamilyEvidence(history, null, "family_ensemble")).toEqual([]);
  });

  it("caps accuracy at the selected reference and supports both anchors", () => {
    const throughSecond = settledMetaRecordsThrough(history.records, history.records[1]);
    expect(throughSecond.map((item) => item.referenceDrawNumber)).toEqual([1, 2]);
    expect(
      selectMetaAccuracyWindow(history.records, history.records[2], 1, "first").map(
        (item) => item.referenceDrawNumber,
      ),
    ).toEqual([1]);
    expect(
      selectMetaAccuracyWindow(history.records, history.records[2], 1, "latest").map(
        (item) => item.referenceDrawNumber,
      ),
    ).toEqual([2]);
    expect(selectMetaAccuracyWindow(history.records, null, 2, "latest")).toEqual([]);
  });

  it("aggregates co-winner-aware backend evaluations without recomputing them", () => {
    const summaries = aggregateMetaAccuracy(history.records.slice(0, 2));
    const ensemble = summaries[0];
    const recent = summaries[2];
    expect(ensemble).toMatchObject({
      evaluations: 2,
      topPredictionHits: 2,
      topPredictionHitRate: 1,
      meanWinningProbabilityMass: 0.6,
      meanReciprocalWinnerRank: 1,
      meanBrierScore: 0.18,
    });
    expect(recent.topPredictionHitRate).toBe(0);
    expect(recent.meanReciprocalWinnerRank).toBe(0.5);
    expect(aggregateMetaAccuracy([]).every((summary) => summary.evaluations === 0)).toBe(
      true,
    );
  });
});
