import { describe, expect, it } from "vitest";

import type {
  PredictionSuite,
  RelationshipEdge,
  StrategyId,
  StrategyPrediction,
} from "../types";
import { generateDrawPortfolio, portfolioPoolSize } from "./drawPortfolio";

function strategy(id: StrategyId, offset: number, averageHits = 1): StrategyPrediction {
  const ranking = Array.from(
    { length: 49 },
    (_value, index) => ((index + offset) % 49) + 1,
  );
  return {
    id,
    name: id,
    description: `${id} test strategy`,
    topNumbers: ranking.slice(0, 6),
    numbers: ranking.map((number, index) => ({
      number,
      rank: index + 1,
      score: (49 - index) / 49,
      gap: index,
      details: [],
    })),
    efficacy: {
      evaluatedDraws: 100,
      strategyHits: averageHits * 100,
      randomHits: 73,
      expectedRandomHits: 100 * 36 / 49,
      averageHitsPerDraw: averageHits,
      randomAverageHitsPerDraw: 0.73,
      hitDifference: Math.round(averageHits * 100) - 73,
    },
  };
}

function suite(strategies: StrategyPrediction[]): PredictionSuite {
  return {
    referenceDrawNumber: 100,
    targetDrawNumber: 101,
    actualNumbers: [],
    strategies,
  };
}

function relationships(): RelationshipEdge[] {
  const edges: RelationshipEdge[] = [];
  for (let left = 1; left < 49; left += 1) {
    for (let right = left + 1; right <= 49; right += 1) {
      edges.push({
        left,
        right,
        count: left + right,
        expected: 20,
        lift: 0.5 + ((left * right) % 20) / 10,
        residual: 0,
      });
    }
  }
  return edges;
}

describe("draw portfolio generation", () => {
  it("uses the adaptive 12-to-24-number pool boundaries", () => {
    expect(portfolioPoolSize(1)).toBe(12);
    expect(portfolioPoolSize(10)).toBe(20);
    expect(portfolioPoolSize(100)).toBe(24);
    expect(portfolioPoolSize(500)).toBe(24);
  });

  it("is deterministic and excludes random baselines", () => {
    const predictionSuite = suite([
      strategy("freshness", 0, 1.2),
      strategy("entropy", 7, 0.9),
      strategy("randomness", 19, 6),
      strategy("fresh_random", 31, 6),
    ]);
    const first = generateDrawPortfolio(predictionSuite, relationships(), 10);
    const second = generateDrawPortfolio(predictionSuite, relationships(), 10);

    expect(first).toEqual(second);
    expect(first?.contributingStrategyCount).toBe(2);
    expect(first?.draws).toHaveLength(10);
    expect(new Set(first?.draws.map((draw) => draw.numbers.join(","))).size).toBe(10);
    expect(first?.metrics.coveredNumbers).toBeGreaterThan(6);
  });

  it("gives more influence to the strategy with stronger walk-forward efficacy", () => {
    const freshnessLed = generateDrawPortfolio(
      suite([
        strategy("freshness", 0, 3),
        strategy("proximity", 24, 0),
      ]),
      [],
      1,
    );
    const proximityLed = generateDrawPortfolio(
      suite([
        strategy("freshness", 0, 0),
        strategy("proximity", 24, 3),
      ]),
      [],
      1,
    );

    expect(freshnessLed?.pool[0].number).toBe(1);
    expect(proximityLed?.pool[0].number).toBe(25);
  });

  it("returns valid portfolios at the supported count boundaries", () => {
    const predictionSuite = suite([
      strategy("freshness", 0, 1.1),
      strategy("proximity", 8, 0.8),
      strategy("entropy", 16, 1),
    ]);
    const one = generateDrawPortfolio(predictionSuite, [], 1);
    const hundred = generateDrawPortfolio(predictionSuite, [], 100);

    expect(one?.draws).toHaveLength(1);
    expect(hundred?.draws).toHaveLength(100);
    for (const draw of hundred?.draws ?? []) {
      expect(draw.numbers).toHaveLength(6);
      expect(new Set(draw.numbers).size).toBe(6);
      expect(draw.numbers).toEqual([...draw.numbers].sort((left, right) => left - right));
      expect(draw.numbers.every((number) => number >= 1 && number <= 49)).toBe(true);
    }
  }, 20_000);

  it("handles unavailable predictive strategies and relationship data", () => {
    expect(
      generateDrawPortfolio(suite([strategy("randomness", 0)]), [], 10),
    ).toBeNull();
    const result = generateDrawPortfolio(
      suite([strategy("freshness", 0)]),
      [],
      0,
    );
    expect(result?.draws).toHaveLength(1);
    expect(result?.metrics.pairCoverage).toBeGreaterThan(0);
  });

  it("produces identical output for edge arrays and equivalent lift providers", () => {
    const predictionSuite = suite([
      strategy("freshness", 0),
      strategy("entropy", 9),
    ]);
    const edges = relationships();
    const liftByPair = new Map(
      edges.map((edge) => [`${edge.left}-${edge.right}`, edge.lift]),
    );
    const fromEdges = generateDrawPortfolio(predictionSuite, edges, 5);
    const fromProvider = generateDrawPortfolio(
      predictionSuite,
      (left, right) =>
        liftByPair.get(`${Math.min(left, right)}-${Math.max(left, right)}`) ?? 0,
      5,
    );

    expect(fromProvider).toEqual(fromEdges);
  });
});
