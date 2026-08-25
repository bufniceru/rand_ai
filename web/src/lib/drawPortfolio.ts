import type {
  DrawPortfolioResultMetadata,
  PossibleDrawConstraints,
  PossibleDrawNumberState,
  PredictionSuite,
  RelationshipEdge,
  SpaceGroupAnalysis,
  SpaceGroupForecast,
  StrategyPrediction,
} from "../types";

const NUMBER_COUNT = 49;
const NUMBERS_PER_DRAW = 6;
const RANDOM_HITS_PER_DRAW = 36 / 49;
const EFFICACY_PRIOR_DRAWS = 24;
const MIN_POOL_SIZE = 12;
const MAX_POOL_SIZE = 24;
const CANDIDATE_PRIORITY_BONUS = 0.1;
const EXCLUDED_STRATEGIES = new Set(["randomness", "fresh_random"]);
export const PORTFOLIO_ALGORITHM_VERSION = 3;

export type RelationshipLiftSource =
  | RelationshipEdge[]
  | ((left: number, right: number) => number);

export interface PortfolioPoolNumber {
  number: number;
  score: number;
  topSixSupport: number;
  state?: PossibleDrawNumberState;
}

export interface PortfolioDraw {
  numbers: number[];
  modelScore: number;
  maximumOverlap: number;
}

export interface PortfolioMetrics {
  averageModelScore: number;
  coveredNumbers: number;
  numberCoverage: number;
  pairCoverage: number;
  tripleCoverage: number;
  maximumOverlap: number;
}

export interface DrawPortfolioResult {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  contributingStrategyCount: number;
  pool: PortfolioPoolNumber[];
  draws: PortfolioDraw[];
  metrics: PortfolioMetrics;
  metadata: DrawPortfolioResultMetadata;
}

interface CandidateDraw {
  indexes: number[];
  numbers: number[];
  mask: number;
  quality: number;
  pairIds: number[];
  tripleIds: number[];
}

interface CandidateTopology {
  indexes: number[];
  mask: number;
  pairIds: number[];
  tripleIds: number[];
}

const topologyCache = new Map<number, CandidateTopology[]>();

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function pairKey(left: number, right: number): string {
  return `${Math.min(left, right)}-${Math.max(left, right)}`;
}

function pairId(left: number, right: number): number {
  return left * MAX_POOL_SIZE + right;
}

function tripleId(first: number, second: number, third: number): number {
  return (first * MAX_POOL_SIZE + second) * MAX_POOL_SIZE + third;
}

function compareNumbers(left: readonly number[], right: readonly number[]): number {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const difference = left[index] - right[index];
    if (difference !== 0) return difference;
  }
  return left.length - right.length;
}

function popcount(value: number): number {
  let remaining = value >>> 0;
  let count = 0;
  while (remaining !== 0) {
    remaining &= remaining - 1;
    count += 1;
  }
  return count;
}

function strategyWeight(strategy: StrategyPrediction): number {
  const evaluatedDraws = strategy.efficacy?.evaluatedDraws ?? 0;
  const hits = strategy.efficacy?.strategyHits ??
    evaluatedDraws * RANDOM_HITS_PER_DRAW;
  const shrunkAverage =
    (hits + EFFICACY_PRIOR_DRAWS * RANDOM_HITS_PER_DRAW) /
    (evaluatedDraws + EFFICACY_PRIOR_DRAWS);
  return clamp(shrunkAverage / RANDOM_HITS_PER_DRAW, 0.5, 1.5);
}

export function portfolioPoolSize(drawCount: number): number {
  const normalizedCount = clamp(Math.trunc(drawCount || 1), 1, 100);
  return clamp(
    MIN_POOL_SIZE + 2 * Math.ceil(Math.log2(normalizedCount)),
    MIN_POOL_SIZE,
    MAX_POOL_SIZE,
  );
}

function scorePoolNumbers(
  strategies: StrategyPrediction[],
): PortfolioPoolNumber[] {
  const weights = new Map(
    strategies.map((strategy) => [strategy.id, strategyWeight(strategy)]),
  );
  const weightTotal = [...weights.values()].reduce((total, weight) => total + weight, 0);
  const scored = Array.from({ length: NUMBER_COUNT }, (_value, index) => {
    const number = index + 1;
    let weightedScore = 0;
    let topSixSupport = 0;
    for (const strategy of strategies) {
      const prediction = strategy.numbers.find((entry) => entry.number === number);
      if (!prediction) continue;
      const weight = weights.get(strategy.id) ?? 1;
      const rankStrength = (NUMBER_COUNT - prediction.rank) / (NUMBER_COUNT - 1);
      const inTopSix = prediction.rank <= NUMBERS_PER_DRAW ? 1 : 0;
      weightedScore += weight * (rankStrength * 0.8 + inTopSix * 0.2);
      topSixSupport += inTopSix;
    }
    return {
      number,
      score: weightTotal > 0 ? weightedScore / weightTotal : 0,
      topSixSupport,
    };
  });
  scored.sort(
    (left, right) =>
      right.score - left.score ||
      right.topSixSupport - left.topSixSupport ||
      left.number - right.number,
  );
  return scored;
}

export interface BorderGroupGuidance {
  borderSpace: number;
  analysis: SpaceGroupAnalysis;
}

function buildPool(
  strategies: StrategyPrediction[],
  drawCount: number,
): PortfolioPoolNumber[] {
  const scored = scorePoolNumbers(strategies);
  const topSixUnion = new Set(strategies.flatMap((strategy) => strategy.topNumbers));

  const targetSize = portfolioPoolSize(drawCount);
  const prioritized = scored.filter((entry) => topSixUnion.has(entry.number));
  const remaining = scored.filter((entry) => !topSixUnion.has(entry.number));
  return [...prioritized, ...remaining].slice(0, targetSize);
}

function normalizeConstraintNumbers(numbers: readonly number[]): number[] {
  return [...new Set(numbers)].filter(
    (number) => Number.isInteger(number) && number >= 1 && number <= NUMBER_COUNT,
  );
}

function buildGuidedPool(
  strategies: StrategyPrediction[],
  drawCount: number,
  constraints: PossibleDrawConstraints,
): {
  pool: PortfolioPoolNumber[];
  fixedNumbers: number[];
  candidateNumbers: number[];
  excludedNumbers: number[];
  omittedCandidates: number[];
} {
  const scored = scorePoolNumbers(strategies);
  const fixedNumbers = normalizeConstraintNumbers(constraints.fixedNumbers).slice(
    0,
    NUMBERS_PER_DRAW,
  );
  const fixed = new Set(fixedNumbers);
  const candidateNumbers = normalizeConstraintNumbers(constraints.candidateNumbers).filter(
    (number) => !fixed.has(number),
  );
  const candidates = new Set(candidateNumbers);
  const excludedNumbers = normalizeConstraintNumbers(constraints.excludedNumbers).filter(
    (number) => !fixed.has(number) && !candidates.has(number),
  );
  const excluded = new Set(excludedNumbers);
  const rankedFixed = scored.filter((entry) => fixed.has(entry.number));
  const rankedCandidates = scored.filter(
    (entry) => candidates.has(entry.number) && !excluded.has(entry.number),
  );
  const rankedNeutral = scored.filter(
    (entry) =>
      !fixed.has(entry.number) &&
      !candidates.has(entry.number) &&
      !excluded.has(entry.number),
  );
  const targetSize = Math.min(
    MAX_POOL_SIZE,
    Math.max(portfolioPoolSize(drawCount), rankedFixed.length + rankedCandidates.length),
  );
  const pool = [...rankedFixed, ...rankedCandidates, ...rankedNeutral]
    .slice(0, targetSize)
    .map((entry) => ({
      ...entry,
      state: fixed.has(entry.number)
        ? "fixed" as const
        : candidates.has(entry.number)
          ? "candidate" as const
          : "neutral" as const,
    }));
  const poolNumbers = new Set(pool.map((entry) => entry.number));
  return {
    pool,
    fixedNumbers,
    candidateNumbers,
    excludedNumbers,
    omittedCandidates: rankedCandidates
      .filter((entry) => !poolNumbers.has(entry.number))
      .map((entry) => entry.number),
  };
}

function candidateTopologies(poolSize: number): CandidateTopology[] {
  const cached = topologyCache.get(poolSize);
  if (cached) return cached;
  const topologies: CandidateTopology[] = [];
  const indexes = Array.from({ length: NUMBERS_PER_DRAW }, () => 0);

  function visit(depth: number, start: number): void {
    if (depth === NUMBERS_PER_DRAW) {
      const pairIds: number[] = [];
      const tripleIds: number[] = [];
      for (let left = 0; left < indexes.length - 1; left += 1) {
        for (let right = left + 1; right < indexes.length; right += 1) {
          const leftIndex = indexes[left];
          const rightIndex = indexes[right];
          pairIds.push(pairId(Math.min(leftIndex, rightIndex), Math.max(leftIndex, rightIndex)));
        }
      }
      for (let first = 0; first < indexes.length - 2; first += 1) {
        for (let second = first + 1; second < indexes.length - 1; second += 1) {
          for (let third = second + 1; third < indexes.length; third += 1) {
            const ordered = [indexes[first], indexes[second], indexes[third]].sort(
              (left, right) => left - right,
            );
            tripleIds.push(tripleId(ordered[0], ordered[1], ordered[2]));
          }
        }
      }
      topologies.push({
        indexes: [...indexes],
        mask: indexes.reduce((mask, index) => mask | (1 << index), 0),
        pairIds,
        tripleIds,
      });
      return;
    }
    const remaining = NUMBERS_PER_DRAW - depth;
    for (let index = start; index <= poolSize - remaining; index += 1) {
      indexes[depth] = index;
      visit(depth + 1, index + 1);
    }
  }

  visit(0, 0);
  topologyCache.set(poolSize, topologies);
  return topologies;
}

function enumerateCandidates(
  pool: PortfolioPoolNumber[],
  relationshipSource: RelationshipLiftSource,
  guided?: { fixedNumbers: number[]; candidateNumbers: number[] },
  groupGuidance?: BorderGroupGuidance,
): CandidateDraw[] {
  const relationshipByPair = Array.isArray(relationshipSource)
    ? new Map(
        relationshipSource.map((edge) => [pairKey(edge.left, edge.right), edge.lift]),
      )
    : null;
  const relationshipLift =
    typeof relationshipSource === "function"
      ? relationshipSource
      : (left: number, right: number) =>
          relationshipByPair?.get(pairKey(left, right)) ?? 0;
  const fixed = new Set(guided?.fixedNumbers ?? []);
  const candidates = new Set(guided?.candidateNumbers ?? []);
  const fixedIndexes = pool
    .map((entry, index) => fixed.has(entry.number) ? index : -1)
    .filter((index) => index >= 0);
  const topologies = guided
    ? candidateTopologies(pool.length).filter(
        (topology) =>
          fixedIndexes.every((index) => topology.indexes.includes(index)),
      )
    : candidateTopologies(pool.length);
  return topologies.map((topology) => {
    const selected = topology.indexes.map((index) => pool[index]);
    const numbers = selected.map((entry) => entry.number).sort((a, b) => a - b);
    const numberStrength =
      selected.reduce((total, entry) => total + entry.score, 0) /
      NUMBERS_PER_DRAW;
    const candidatePriority =
      selected.filter((entry) => candidates.has(entry.number)).length /
      NUMBERS_PER_DRAW;
    let relationshipStrength = 0;
    let relationshipCount = 0;
    for (let left = 0; left < topology.indexes.length - 1; left += 1) {
      for (let right = left + 1; right < topology.indexes.length; right += 1) {
        relationshipStrength += clamp(
          relationshipLift(
            pool[topology.indexes[left]].number,
            pool[topology.indexes[right]].number,
          ) / 2,
          0,
          1,
        );
        relationshipCount += 1;
      }
    }
    const baseQuality =
      numberStrength * 0.8 +
      (relationshipCount > 0 ? relationshipStrength / relationshipCount : 0) * 0.2 +
      candidatePriority * CANDIDATE_PRIORITY_BONUS;
    const groupStrength = groupGuidance
      ? normalizedGroupProbability(numbers, groupGuidance)
      : null;
    return {
      ...topology,
      numbers,
      quality: groupStrength === null
        ? baseQuality
        : baseQuality * 0.8 + groupStrength * 0.2,
    };
  });
}

function groupSignature(numbers: readonly number[], borderSpace: number): string {
  const ordered = [...numbers].sort((left, right) => left - right);
  const spaces = [
    ordered[0] - 1 + NUMBER_COUNT - ordered.at(-1)!,
    ...ordered.slice(1).map((number, index) => number - ordered[index] - 1),
  ];
  const separators = spaces
    .map((space, index) => space > borderSpace ? index : -1)
    .filter((index) => index >= 0);
  if (separators.length === 0) return String(NUMBERS_PER_DRAW);
  return separators
    .map((start, index) => {
      const next = separators[(index + 1) % separators.length];
      return (next - start + NUMBERS_PER_DRAW) % NUMBERS_PER_DRAW || NUMBERS_PER_DRAW;
    })
    .sort((left, right) => right - left)
    .join("+");
}

function selectedGroupForecast(guidance: BorderGroupGuidance): SpaceGroupForecast | null {
  const preferred = guidance.analysis.provisional
    ? "border_group_hybrid"
    : guidance.analysis.bestModelId ?? "border_group_hybrid";
  return guidance.analysis.forecasts.find((forecast) => forecast.modelId === preferred)
    ?? guidance.analysis.forecasts.find((forecast) => forecast.modelId === "border_group_hybrid")
    ?? null;
}

function normalizedGroupProbability(
  numbers: readonly number[],
  guidance: BorderGroupGuidance,
): number {
  const forecast = selectedGroupForecast(guidance);
  if (!forecast) return 0;
  const signature = groupSignature(numbers, guidance.borderSpace);
  const probability = forecast.probabilities.find((entry) => entry.signature === signature)?.probability ?? 0;
  const maximum = Math.max(...forecast.probabilities.map((entry) => entry.probability), Number.EPSILON);
  return probability / maximum;
}

function selectPortfolio(
  candidates: CandidateDraw[],
  pool: PortfolioPoolNumber[],
  drawCount: number,
  fixedMask = 0,
): CandidateDraw[] {
  candidates.sort(
    (left, right) =>
      right.quality - left.quality || compareNumbers(left.numbers, right.numbers),
  );
  const shortlistSize = Math.min(
    candidates.length,
    Math.max(5000, drawCount * 250),
  );
  const shortlist = candidates.slice(0, shortlistSize);
  const selected: CandidateDraw[] = [];
  const usedIndexes = new Set<number>();
  const coveredNumbers = new Set<number>();
  const coveredPairs = new Set<number>();
  const coveredTriples = new Set<number>();
  const totalNumberWeight = pool.reduce((total, entry) => total + entry.score, 0) || 1;
  let totalPairWeight = 0;
  let totalTripleWeight = 0;
  for (let left = 0; left < pool.length - 1; left += 1) {
    for (let right = left + 1; right < pool.length; right += 1) {
      totalPairWeight += pool[left].score * pool[right].score;
      for (let third = right + 1; third < pool.length; third += 1) {
        totalTripleWeight += pool[left].score * pool[right].score * pool[third].score;
      }
    }
  }
  totalPairWeight ||= 1;
  totalTripleWeight ||= 1;

  while (selected.length < drawCount && selected.length < shortlist.length) {
    let bestIndex = -1;
    let bestUtility = Number.NEGATIVE_INFINITY;
    for (let candidateIndex = 0; candidateIndex < shortlist.length; candidateIndex += 1) {
      if (usedIndexes.has(candidateIndex)) continue;
      const candidate = shortlist[candidateIndex];
      const newNumberWeight = candidate.indexes.reduce(
        (total, index) => total + (coveredNumbers.has(index) ? 0 : pool[index].score),
        0,
      );
      let newPairWeight = 0;
      let pairIndex = 0;
      for (let left = 0; left < candidate.indexes.length - 1; left += 1) {
        for (let right = left + 1; right < candidate.indexes.length; right += 1) {
          const id = candidate.pairIds[pairIndex];
          pairIndex += 1;
          if (!coveredPairs.has(id)) {
            newPairWeight +=
              pool[candidate.indexes[left]].score * pool[candidate.indexes[right]].score;
          }
        }
      }
      let newTripleWeight = 0;
      let tripleIndex = 0;
      for (let first = 0; first < candidate.indexes.length - 2; first += 1) {
        for (let second = first + 1; second < candidate.indexes.length - 1; second += 1) {
          for (let third = second + 1; third < candidate.indexes.length; third += 1) {
            const id = candidate.tripleIds[tripleIndex];
            tripleIndex += 1;
            if (!coveredTriples.has(id)) {
              newTripleWeight +=
                pool[candidate.indexes[first]].score *
                pool[candidate.indexes[second]].score *
                pool[candidate.indexes[third]].score;
            }
          }
        }
      }
      const maximumAvoidableOverlap = selected.reduce(
        (maximum, other) =>
          Math.max(maximum, popcount((candidate.mask & other.mask) & ~fixedMask)),
        0,
      );
      const avoidableSlots = Math.max(NUMBERS_PER_DRAW - popcount(fixedMask), 1);
      const overlapPenalty = 0.15 * (maximumAvoidableOverlap / avoidableSlots) ** 2;
      const utility =
        candidate.quality * 0.6 +
        (newNumberWeight / totalNumberWeight) * 0.15 +
        (newPairWeight / totalPairWeight) * 0.15 +
        (newTripleWeight / totalTripleWeight) * 0.1 -
        overlapPenalty;
      if (
        utility > bestUtility + Number.EPSILON ||
        (Math.abs(utility - bestUtility) <= Number.EPSILON &&
          (bestIndex < 0 ||
            compareNumbers(candidate.numbers, shortlist[bestIndex].numbers) < 0))
      ) {
        bestUtility = utility;
        bestIndex = candidateIndex;
      }
    }
    if (bestIndex < 0) break;
    const chosen = shortlist[bestIndex];
    usedIndexes.add(bestIndex);
    selected.push(chosen);
    chosen.indexes.forEach((index) => coveredNumbers.add(index));
    chosen.pairIds.forEach((id) => coveredPairs.add(id));
    chosen.tripleIds.forEach((id) => coveredTriples.add(id));
  }
  return selected;
}

function buildMetrics(
  selected: CandidateDraw[],
  pool: PortfolioPoolNumber[],
): PortfolioMetrics {
  const coveredIndexes = new Set(selected.flatMap((candidate) => candidate.indexes));
  const coveredPairs = new Set(selected.flatMap((candidate) => candidate.pairIds));
  const coveredTriples = new Set(selected.flatMap((candidate) => candidate.tripleIds));
  let pairWeight = 0;
  let coveredPairWeight = 0;
  let tripleWeight = 0;
  let coveredTripleWeight = 0;
  for (let left = 0; left < pool.length - 1; left += 1) {
    for (let right = left + 1; right < pool.length; right += 1) {
      const weight = pool[left].score * pool[right].score;
      pairWeight += weight;
      if (coveredPairs.has(pairId(left, right))) coveredPairWeight += weight;
      for (let third = right + 1; third < pool.length; third += 1) {
        const tripleWeightValue = weight * pool[third].score;
        tripleWeight += tripleWeightValue;
        if (coveredTriples.has(tripleId(left, right, third))) {
          coveredTripleWeight += tripleWeightValue;
        }
      }
    }
  }
  let maximumOverlap = 0;
  for (let left = 0; left < selected.length - 1; left += 1) {
    for (let right = left + 1; right < selected.length; right += 1) {
      maximumOverlap = Math.max(
        maximumOverlap,
        popcount(selected[left].mask & selected[right].mask),
      );
    }
  }
  return {
    averageModelScore:
      selected.length > 0
        ? selected.reduce((total, candidate) => total + candidate.quality, 0) /
          selected.length
        : 0,
    coveredNumbers: coveredIndexes.size,
    numberCoverage: pool.length > 0 ? coveredIndexes.size / pool.length : 0,
    pairCoverage: pairWeight > 0 ? coveredPairWeight / pairWeight : 0,
    tripleCoverage: tripleWeight > 0 ? coveredTripleWeight / tripleWeight : 0,
    maximumOverlap,
  };
}

export function generateDrawPortfolio(
  suite: PredictionSuite,
  relationshipSource: RelationshipLiftSource,
  requestedDrawCount: number,
  constraints?: PossibleDrawConstraints,
  groupGuidance?: BorderGroupGuidance,
): DrawPortfolioResult | null {
  const strategies = suite.strategies.filter(
    (strategy) => !EXCLUDED_STRATEGIES.has(strategy.id),
  );
  if (strategies.length === 0) return null;
  const drawCount = clamp(Math.trunc(requestedDrawCount || 1), 1, 100);
  const mode = constraints?.mode ?? "classic";
  const guided = mode === "guided" && constraints
    ? buildGuidedPool(strategies, drawCount, constraints)
    : null;
  const displayFixedNumbers = guided?.fixedNumbers ??
    normalizeConstraintNumbers(constraints?.fixedNumbers ?? []).slice(0, NUMBERS_PER_DRAW);
  const displayFixed = new Set(displayFixedNumbers);
  const displayCandidateNumbers = guided?.candidateNumbers ??
    normalizeConstraintNumbers(constraints?.candidateNumbers ?? []).filter(
      (number) => !displayFixed.has(number),
    );
  const displayCandidates = new Set(displayCandidateNumbers);
  const displayExcludedNumbers = guided?.excludedNumbers ??
    normalizeConstraintNumbers(constraints?.excludedNumbers ?? []).filter(
      (number) => !displayFixed.has(number) && !displayCandidates.has(number),
    );
  const displayExcluded = new Set(displayExcludedNumbers);
  const classicPool = guided ? [] : buildPool(strategies, drawCount);
  const pool = guided?.pool ?? (constraints
    ? classicPool.map((entry) => ({
        ...entry,
        state: displayFixed.has(entry.number)
          ? "fixed" as const
          : displayCandidates.has(entry.number)
            ? "candidate" as const
            : displayExcluded.has(entry.number)
              ? "excluded" as const
              : "neutral" as const,
      }))
    : classicPool);
  const candidates = enumerateCandidates(
    pool,
    relationshipSource,
    guided
      ? {
          fixedNumbers: guided.fixedNumbers,
          candidateNumbers: guided.candidateNumbers,
        }
      : undefined,
    groupGuidance,
  );
  const fixedSet = new Set(guided?.fixedNumbers ?? []);
  const fixedMask = pool.reduce(
    (mask, entry, index) => fixedSet.has(entry.number) ? mask | (1 << index) : mask,
    0,
  );
  const selected = selectPortfolio(candidates, pool, drawCount, fixedMask);
  const metrics = buildMetrics(selected, pool);
  const omittedCandidates = guided?.omittedCandidates ?? [];
  const constraintLimited = selected.length < drawCount || omittedCandidates.length > 0;
  const messages: string[] = [];
  if (omittedCandidates.length > 0) {
    messages.push(
      `${omittedCandidates.length} lower-ranked Candidate${omittedCandidates.length === 1 ? " was" : "s were"} omitted by the 24-number pool limit`,
    );
  }
  if (selected.length < drawCount) {
    messages.push(
      `the active constraints allow ${candidates.length} unique ticket${candidates.length === 1 ? "" : "s"}, so ${selected.length} of ${drawCount} requested draws were generated`,
    );
  }
  return {
    referenceDrawNumber: suite.referenceDrawNumber,
    targetDrawNumber: suite.targetDrawNumber,
    contributingStrategyCount: strategies.length,
    pool,
    draws: selected.map((candidate) => ({
      numbers: candidate.numbers,
      modelScore: candidate.quality,
      maximumOverlap: selected.reduce(
        (maximum, other) =>
          other === candidate
            ? maximum
            : Math.max(maximum, popcount(candidate.mask & other.mask)),
        0,
      ),
    })),
    metrics,
    metadata: {
      mode,
      requestedDrawCount: drawCount,
      generatedDrawCount: selected.length,
      availableUniqueCount: candidates.length,
      fixedNumbers: displayFixedNumbers,
      candidateNumbers: displayCandidateNumbers,
      excludedNumbers: displayExcludedNumbers,
      omittedCandidates,
      constraintLimited,
      constraintMessage: messages.length > 0 ? `${messages.join("; ")}.` : undefined,
      borderSpace: groupGuidance?.borderSpace,
      groupModelId: groupGuidance ? selectedGroupForecast(groupGuidance)?.modelId : undefined,
      provisionalGroupModel: groupGuidance?.analysis.provisional,
    },
  };
}
