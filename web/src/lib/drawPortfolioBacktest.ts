import {
  generateDrawPortfolio,
  PORTFOLIO_ALGORITHM_VERSION,
} from "./drawPortfolio";
import type {
  PortfolioBacktestAuditRow,
  PortfolioBacktestData,
  PortfolioBacktestProgress,
  PortfolioBacktestResult,
  PredictionSuite,
  StrategyEfficacy,
  StrategyId,
  StrategyPrediction,
} from "../types";

const NUMBER_COUNT = 49;
const PAIR_UNIVERSE = 49 * 48 / 2;
const RANDOM_HITS_PER_DRAW = 36 / 49;

function compareNumbers(left: readonly number[], right: readonly number[]): number {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return left.length - right.length;
}

function strategyEfficacy(
  strategyId: StrategyId,
  evaluatedDraws: number,
  hitsByStrategy: Map<StrategyId, number>,
): StrategyEfficacy {
  const strategyHits = hitsByStrategy.get(strategyId) ?? 0;
  return {
    evaluatedDraws,
    strategyHits,
    randomHits: 0,
    expectedRandomHits: evaluatedDraws * RANDOM_HITS_PER_DRAW,
    averageHitsPerDraw: evaluatedDraws > 0 ? strategyHits / evaluatedDraws : 0,
    randomAverageHitsPerDraw: 0,
    hitDifference: strategyHits,
  };
}

function predictionSuite(
  record: PortfolioBacktestData["records"][number],
  evaluatedDraws: number,
  hitsByStrategy: Map<StrategyId, number>,
): PredictionSuite {
  const strategies: StrategyPrediction[] = record.strategies.map((strategy) => ({
    id: strategy.id,
    name: strategy.id,
    description: "Compact full-history portfolio strategy",
    topNumbers: strategy.ranking.slice(0, 6),
    numbers: strategy.ranking.map((number, index) => ({
      number,
      rank: index + 1,
      score: (NUMBER_COUNT - index) / NUMBER_COUNT,
      gap: 0,
      details: [],
    })),
    efficacy: strategyEfficacy(strategy.id, evaluatedDraws, hitsByStrategy),
  }));
  return {
    referenceDrawNumber: record.referenceDrawNumber,
    targetDrawNumber: record.targetDrawNumber,
    actualNumbers: record.actualNumbers,
    strategies,
  };
}

function updatePairCounts(counts: Uint32Array, numbers: readonly number[]): void {
  for (let leftIndex = 0; leftIndex < numbers.length - 1; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < numbers.length; rightIndex += 1) {
      const left = Math.min(numbers[leftIndex], numbers[rightIndex]);
      const right = Math.max(numbers[leftIndex], numbers[rightIndex]);
      counts[left * 50 + right] += 1;
    }
  }
}

function updateStrategyHits(
  record: PortfolioBacktestData["records"][number],
  hitsByStrategy: Map<StrategyId, number>,
): void {
  const actual = new Set(record.actualNumbers);
  for (const strategy of record.strategies) {
    const hits = strategy.ranking
      .slice(0, 6)
      .filter((number) => actual.has(number)).length;
    hitsByStrategy.set(strategy.id, (hitsByStrategy.get(strategy.id) ?? 0) + hits);
  }
}

function buildBuckets(audit: PortfolioBacktestAuditRow[]) {
  const total = audit.length;
  return Array.from({ length: 7 }, (_value, hits) => {
    const exactCount = audit.filter((row) => row.bestHits === hits).length;
    const atLeastCount = audit.filter((row) => row.bestHits >= hits).length;
    return {
      hits,
      exactCount,
      exactRate: total > 0 ? exactCount / total : 0,
      atLeastCount,
      atLeastRate: total > 0 ? atLeastCount / total : 0,
    };
  });
}

export function portfolioBacktestCacheKey(
  sourceCacheKey: string,
  portfolioSize: number,
): string {
  return `${sourceCacheKey}-v${PORTFOLIO_ALGORITHM_VERSION}-p${portfolioSize}`;
}

export function runPortfolioBacktest(
  data: PortfolioBacktestData,
  requestedPortfolioSize: number,
  onProgress?: (progress: PortfolioBacktestProgress) => void,
): PortfolioBacktestResult {
  const portfolioSize = Math.min(Math.max(Math.trunc(requestedPortfolioSize || 10), 1), 100);
  const startedAt = performance.now();
  const audit: PortfolioBacktestAuditRow[] = [];
  const pairCounts = new Uint32Array(50 * 50);
  const hitsByStrategy = new Map<StrategyId, number>();
  let includedDrawCount = 0;
  let evaluatedDraws = 0;
  const total = data.records.length;

  for (let recordIndex = 0; recordIndex < data.records.length; recordIndex += 1) {
    const record = data.records[recordIndex];
    while (
      includedDrawCount < record.referenceDrawNumber &&
      includedDrawCount < data.draws.length
    ) {
      updatePairCounts(pairCounts, data.draws[includedDrawCount].numbers);
      includedDrawCount += 1;
    }
    const expectedPairCount = record.referenceDrawNumber * 15 / PAIR_UNIVERSE;
    const portfolio = generateDrawPortfolio(
      predictionSuite(record, evaluatedDraws, hitsByStrategy),
      (left, right) => {
        const first = Math.min(left, right);
        const second = Math.max(left, right);
        return expectedPairCount > 0
          ? pairCounts[first * 50 + second] / expectedPairCount
          : 0;
      },
      portfolioSize,
    );
    if (!portfolio) {
      throw new Error("No predictive strategies are available for portfolio simulation.");
    }
    const actual = new Set(record.actualNumbers);
    const evaluatedTickets = portfolio.draws.map((draw) => ({
      numbers: draw.numbers,
      hits: draw.numbers.filter((number) => actual.has(number)).length,
    }));
    const bestHits = Math.max(...evaluatedTickets.map((ticket) => ticket.hits));
    const tied = evaluatedTickets
      .filter((ticket) => ticket.hits === bestHits)
      .sort((left, right) => compareNumbers(left.numbers, right.numbers));
    audit.push({
      referenceDrawNumber: record.referenceDrawNumber,
      targetDrawNumber: record.targetDrawNumber,
      date: record.date,
      actualNumbers: [...record.actualNumbers].sort((left, right) => left - right),
      bestTicket: tied[0].numbers,
      bestHits,
      tiedBestCount: tied.length,
    });
    updateStrategyHits(record, hitsByStrategy);
    evaluatedDraws += 1;
    const processed = recordIndex + 1;
    onProgress?.({
      percent: total > 0 ? Math.round((processed / total) * 100) : 100,
      message: `Simulated target draw ${record.targetDrawNumber}`,
      processed,
      total,
    });
  }

  return {
    algorithmVersion: PORTFOLIO_ALGORITHM_VERSION,
    sourceCacheKey: data.cacheKey,
    portfolioSize,
    evaluatedTargets: audit.length,
    durationMs: performance.now() - startedAt,
    buckets: buildBuckets(audit),
    audit,
  };
}
