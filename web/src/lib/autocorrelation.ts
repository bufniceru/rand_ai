import type {
  AutocorrelationBand,
  AnalysisHistoryDraw,
  AutocorrelationLagSummary,
  AutocorrelationModel,
  AutocorrelationNumberSummary,
} from "../types";
import { themeColor } from "./colorTemplates";

const numberCount = 49;
const numbersPerDraw = 6;
const maxAnalyzedLag = 24;
const lowHighSplit = 24;

export const autocorrelationBands: AutocorrelationBand[] = [
  {
    id: "strongNegative",
    label: "Strong negative",
    description: "Correlation is below -0.10.",
    color: "#78dce8",
  },
  {
    id: "mildNegative",
    label: "Mild negative",
    description: "Correlation sits between -0.10 and -0.04.",
    color: "#ab9df2",
  },
  {
    id: "neutral",
    label: "Neutral",
    description: "Correlation is close to zero.",
    color: "#a9dc76",
  },
  {
    id: "mildPositive",
    label: "Mild positive",
    description: "Correlation sits between 0.04 and 0.10.",
    color: "#ffd866",
  },
  {
    id: "strongPositive",
    label: "Strong positive",
    description: "Correlation is above 0.10.",
    color: "#ff6188",
  },
];

function refreshAutocorrelationBandColors(): void {
  const ids = [
    "predictions.autocorrelationStrongNegative",
    "predictions.autocorrelationMildNegative",
    "predictions.autocorrelationNeutral",
    "predictions.autocorrelationMildPositive",
    "predictions.autocorrelationStrongPositive",
  ];
  autocorrelationBands.forEach((band, index) => {
    band.color = themeColor(ids[index]!);
  });
}

const bandById = new Map(
  autocorrelationBands.map((band) => [band.id, band]),
);

function correlation(leftValues: number[], rightValues: number[]): number {
  const count = Math.min(leftValues.length, rightValues.length);
  if (count < 3) return 0;

  const leftMean =
    leftValues.slice(0, count).reduce((total, value) => total + value, 0) /
    count;
  const rightMean =
    rightValues.slice(0, count).reduce((total, value) => total + value, 0) /
    count;
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;

  for (let index = 0; index < count; index += 1) {
    const leftDelta = (leftValues[index] ?? 0) - leftMean;
    const rightDelta = (rightValues[index] ?? 0) - rightMean;
    covariance += leftDelta * rightDelta;
    leftVariance += leftDelta * leftDelta;
    rightVariance += rightDelta * rightDelta;
  }

  const denominator = Math.sqrt(leftVariance * rightVariance);
  return denominator === 0 ? 0 : covariance / denominator;
}

function bandForCorrelation(value: number): AutocorrelationBand {
  if (value <= -0.1) {
    return bandById.get("strongNegative") ?? autocorrelationBands[0]!;
  }
  if (value <= -0.04) {
    return bandById.get("mildNegative") ?? autocorrelationBands[1]!;
  }
  if (value >= 0.1) {
    return bandById.get("strongPositive") ?? autocorrelationBands[4]!;
  }
  if (value >= 0.04) {
    return bandById.get("mildPositive") ?? autocorrelationBands[3]!;
  }
  return bandById.get("neutral") ?? autocorrelationBands[2]!;
}

function choose(count: number, picks: number): number {
  if (picks < 0 || count < picks) return 0;
  if (picks === 0) return 1;

  let numerator = 1;
  let denominator = 1;
  for (let index = 1; index <= picks; index += 1) {
    numerator *= count - picks + index;
    denominator *= index;
  }
  return numerator / denominator;
}

function countOverlap(left: Set<number>, right: Set<number>): number {
  let count = 0;
  for (const number of left) {
    if (right.has(number)) count += 1;
  }
  return count;
}

function presenceSeries(
  history: AnalysisHistoryDraw[],
  number: number,
): number[] {
  return history.map((draw) => (draw.numbers.includes(number) ? 1 : 0));
}

function laggedCorrelation(values: number[], lag: number): number {
  return correlation(values.slice(0, -lag), values.slice(lag));
}

function interpretationFor(
  strongestLag: AutocorrelationLagSummary | null,
): string {
  if (strongestLag === null || strongestLag.score < 0.04) {
    return "No meaningful lag relationship stands out across the analyzed draw history.";
  }
  if (strongestLag.score >= 0.1) {
    return `The strongest lag is ${strongestLag.lag}, but this is still descriptive autocorrelation, not lottery probability.`;
  }
  return `Lag ${strongestLag.lag} is the most visible pattern, with only mild autocorrelation strength.`;
}

export function buildAutocorrelationModel(
  history: AnalysisHistoryDraw[],
): AutocorrelationModel {
  refreshAutocorrelationBandColors();
  const drawCount = history.length;
  const maxLag = Math.min(maxAnalyzedLag, Math.max(drawCount - 2, 1));
  const expectedOverlap = (numbersPerDraw * numbersPerDraw) / numberCount;
  const expectedDoublets =
    (choose(numbersPerDraw, 2) * choose(numbersPerDraw, 2)) /
    choose(numberCount, 2);
  const expectedTriplets =
    (choose(numbersPerDraw, 3) * choose(numbersPerDraw, 3)) /
    choose(numberCount, 3);
  const sets = history.map((draw) => new Set(draw.numbers));
  const sums = history.map((draw) =>
    draw.numbers.reduce((total, number) => total + number, 0),
  );
  const oddCounts = history.map(
    (draw) => draw.numbers.filter((number) => number % 2 === 1).length,
  );
  const lowCounts = history.map(
    (draw) => draw.numbers.filter((number) => number <= lowHighSplit).length,
  );
  const numberSeries = new Map<number, number[]>();

  for (let number = 1; number <= numberCount; number += 1) {
    numberSeries.set(number, presenceSeries(history, number));
  }

  const lagSummaries: AutocorrelationLagSummary[] = [];
  for (let lag = 1; lag <= maxLag; lag += 1) {
    const pairCount = Math.max(drawCount - lag, 0);
    let overlapTotal = 0;
    let doubletTotal = 0;
    let tripletTotal = 0;
    let numberCorrelationTotal = 0;

    for (let index = 0; index < pairCount; index += 1) {
      const overlap = countOverlap(sets[index] ?? new Set(), sets[index + lag] ?? new Set());
      overlapTotal += overlap;
      doubletTotal += choose(overlap, 2);
      tripletTotal += choose(overlap, 3);
    }
    for (let number = 1; number <= numberCount; number += 1) {
      numberCorrelationTotal += laggedCorrelation(
        numberSeries.get(number) ?? [],
        lag,
      );
    }

    const averageOverlap = pairCount === 0 ? 0 : overlapTotal / pairCount;
    const averageDoublets = pairCount === 0 ? 0 : doubletTotal / pairCount;
    const averageTriplets = pairCount === 0 ? 0 : tripletTotal / pairCount;
    const numberPresenceCorrelation = numberCorrelationTotal / numberCount;
    const sumCorrelation = laggedCorrelation(sums, lag);
    const oddCountCorrelation = laggedCorrelation(oddCounts, lag);
    const lowCountCorrelation = laggedCorrelation(lowCounts, lag);
    const score = Math.max(
      Math.abs(numberPresenceCorrelation),
      Math.abs(sumCorrelation),
      Math.abs(oddCountCorrelation),
      Math.abs(lowCountCorrelation),
      Math.abs((averageOverlap - expectedOverlap) / numbersPerDraw),
      Math.abs(
        (averageDoublets - expectedDoublets) /
          Math.max(expectedDoublets, 0.001),
      ),
      Math.abs(
        (averageTriplets - expectedTriplets) /
          Math.max(expectedTriplets, 0.001),
      ),
    );
    const direction =
      Math.abs(numberPresenceCorrelation) >= Math.abs(sumCorrelation)
        ? numberPresenceCorrelation
        : sumCorrelation;
    const band = bandForCorrelation(direction);

    lagSummaries.push({
      lag,
      pairCount,
      averageOverlap,
      expectedOverlap,
      overlapDelta: averageOverlap - expectedOverlap,
      overlapRate: averageOverlap / numbersPerDraw,
      averageDoublets,
      expectedDoublets,
      doubletDelta: averageDoublets - expectedDoublets,
      averageTriplets,
      expectedTriplets,
      tripletDelta: averageTriplets - expectedTriplets,
      numberPresenceCorrelation,
      sumCorrelation,
      oddCountCorrelation,
      lowCountCorrelation,
      score,
      bandId: band.id,
      label: band.label,
    });
  }

  const numberSummaries: AutocorrelationNumberSummary[] = Array.from(
    { length: numberCount },
    (_value, index) => {
      const number = index + 1;
      const series = numberSeries.get(number) ?? [];
      const appearances = series.reduce((total, value) => total + value, 0);
      let strongestLag = 1;
      let strongestCorrelation = 0;

      for (let lag = 1; lag <= maxLag; lag += 1) {
        const value = laggedCorrelation(series, lag);
        if (Math.abs(value) > Math.abs(strongestCorrelation)) {
          strongestLag = lag;
          strongestCorrelation = value;
        }
      }

      const band = bandForCorrelation(strongestCorrelation);
      return {
        number,
        appearances,
        strongestLag,
        strongestCorrelation,
        score: Math.abs(strongestCorrelation),
        bandId: band.id,
        label: band.label,
        rank: 0,
      };
    },
  )
    .sort(
      (left, right) =>
        right.score - left.score ||
        Math.abs(right.strongestCorrelation) -
          Math.abs(left.strongestCorrelation) ||
        left.number - right.number,
    )
    .map((summary, index) => ({ ...summary, rank: index + 1 }));

  const strongestLag =
    [...lagSummaries].sort(
      (left, right) => right.score - left.score || left.lag - right.lag,
    )[0] ?? null;
  const strongestPositiveLag =
    lagSummaries
      .filter((summary) => summary.numberPresenceCorrelation > 0)
      .sort(
        (left, right) =>
          right.numberPresenceCorrelation - left.numberPresenceCorrelation ||
          left.lag - right.lag,
      )[0] ?? null;
  const strongestNegativeLag =
    lagSummaries
      .filter((summary) => summary.numberPresenceCorrelation < 0)
      .sort(
        (left, right) =>
          left.numberPresenceCorrelation - right.numberPresenceCorrelation ||
          left.lag - right.lag,
      )[0] ?? null;
  const summariesByNumber = new Map(
    numberSummaries.map((summary) => [summary.number, summary]),
  );
  const latestDraw = history.at(-1) ?? null;
  const latestNumbers = (latestDraw?.numbers ?? [])
    .map((number) => summariesByNumber.get(number))
    .filter(
      (summary): summary is AutocorrelationNumberSummary =>
        summary !== undefined,
    )
    .sort(
      (left, right) => right.score - left.score || left.number - right.number,
    );

  return {
    bands: autocorrelationBands,
    drawCount,
    maxLag,
    expectedOverlap,
    expectedDoublets,
    expectedTriplets,
    lagSummaries,
    numberSummaries,
    strongestLag,
    strongestPositiveLag,
    strongestNegativeLag,
    latestProfile: {
      date: latestDraw?.date ?? null,
      signature:
        latestNumbers.length === 0
          ? "n/a"
          : latestNumbers
              .map(
                (summary) =>
                  `${summary.number}: lag ${summary.strongestLag}`,
              )
              .join(" | "),
      numbers: latestNumbers,
    },
    interpretation: interpretationFor(strongestLag),
  };
}
