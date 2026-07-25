const LOTTERY_NUMBER_COUNT = 49;
const TICKET_SIZE = 6;
const RANDOM_BUFFER_LENGTH = 16_384;
const UINT32_RANGE = 0x100000000;

export const RANDOM_BENCHMARK_SIMULATIONS = 10_000;

export interface RandomBenchmarkSummary {
  simulations: number;
  meanHits: number;
  lower95Hits: number;
  upper95Hits: number;
  sortedTotalHits: number[];
}

function combinations(total: number, selected: number): number {
  const smallerSelection = Math.min(selected, total - selected);
  let result = 1;
  for (let index = 1; index <= smallerSelection; index += 1) {
    result = result * (total - smallerSelection + index) / index;
  }
  return result;
}

function buildTicketHitCdf(): number[] {
  const possibleTickets = combinations(LOTTERY_NUMBER_COUNT, TICKET_SIZE);
  let cumulativeProbability = 0;
  return Array.from({ length: TICKET_SIZE + 1 }, (_value, hits) => {
    cumulativeProbability += (
      combinations(TICKET_SIZE, hits)
      * combinations(LOTTERY_NUMBER_COUNT - TICKET_SIZE, TICKET_SIZE - hits)
      / possibleTickets
    );
    return cumulativeProbability;
  });
}

const TICKET_HIT_CDF = buildTicketHitCdf();

function createCryptoRandomUnit(): () => number {
  const values = new Uint32Array(RANDOM_BUFFER_LENGTH);
  let nextIndex = values.length;
  return () => {
    if (nextIndex === values.length) {
      crypto.getRandomValues(values);
      nextIndex = 0;
    }
    const value = values[nextIndex] ?? 0;
    nextIndex += 1;
    return value / UINT32_RANGE;
  };
}

function sampleTicketHits(randomUnit: number): number {
  for (let hits = 0; hits < TICKET_HIT_CDF.length; hits += 1) {
    if (randomUnit < (TICKET_HIT_CDF[hits] ?? 1)) return hits;
  }
  return TICKET_SIZE;
}

export function runRandomBenchmark(
  drawCount: number,
  simulations = RANDOM_BENCHMARK_SIMULATIONS,
): RandomBenchmarkSummary {
  if (!Number.isInteger(drawCount) || drawCount < 1) {
    throw new RangeError("Draw count must be a positive integer");
  }
  if (!Number.isInteger(simulations) || simulations < 1) {
    throw new RangeError("Simulation count must be a positive integer");
  }

  const randomUnit = createCryptoRandomUnit();
  const totalHits = new Uint32Array(simulations);
  let hitSum = 0;
  for (let simulation = 0; simulation < simulations; simulation += 1) {
    let simulationHits = 0;
    for (let draw = 0; draw < drawCount; draw += 1) {
      simulationHits += sampleTicketHits(randomUnit());
    }
    totalHits[simulation] = simulationHits;
    hitSum += simulationHits;
  }

  const sortedTotalHits = Array.from(totalHits).sort((left, right) => left - right);
  const lowerIndex = Math.floor((simulations - 1) * 0.025);
  const upperIndex = Math.ceil((simulations - 1) * 0.975);
  return {
    simulations,
    meanHits: hitSum / simulations,
    lower95Hits: sortedTotalHits[lowerIndex] ?? 0,
    upper95Hits: sortedTotalHits[upperIndex] ?? 0,
    sortedTotalHits,
  };
}

export function randomTailProbability(
  summary: RandomBenchmarkSummary,
  strategyHits: number,
): number {
  const values = summary.sortedTotalHits;
  let lower = 0;
  let upper = values.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if ((values[middle] ?? 0) < strategyHits) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  const atLeastStrategy = values.length - lower;
  return (atLeastStrategy + 1) / (values.length + 1);
}
