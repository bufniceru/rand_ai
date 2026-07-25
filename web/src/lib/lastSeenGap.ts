import type { HistoryDraw } from "../types";

export interface LastSeenGapPoint {
  gap: number;
  gapGap: number;
  drawIndex: number;
  highlighted: boolean;
}

export interface LastSeenGapModel {
  draws: HistoryDraw[];
  points: LastSeenGapPoint[];
  drawCount: number;
  maxGap: number;
  referenceGaps: number[];
  referenceGapNumbers: Record<number, number[]>;
  maxReferenceOffset: number;
  referenceDrawIndex: number | null;
  referenceDrawNumber: number | null;
}

function refreshDrawGaps(draws: HistoryDraw[]): HistoryDraw[] {
  const lastSeen = new Map<number, number | null>();
  for (let number = 1; number <= 49; number += 1) lastSeen.set(number, null);

  return draws.map((draw, drawIndex) => {
    const numbers = draw.numbers.map((number) => {
      const previousIndex = lastSeen.get(number.value) ?? null;
      const gap = previousIndex === null ? drawIndex : drawIndex - previousIndex - 1;
      lastSeen.set(number.value, drawIndex);
      return {
        value: number.value,
        gap,
        leftSpace: number.leftSpace,
        rightSpace: number.rightSpace,
      };
    });
    return { drawNumber: draw.drawNumber, date: draw.date, numbers };
  });
}

function currentGapsForReference(
  history: HistoryDraw[],
  referenceDrawOffset: number,
): Map<number, number> {
  const referenceDrawIndex = Math.max(0, history.length - 1 - referenceDrawOffset);
  const lastSeen = new Map<number, number | null>();
  for (let number = 1; number <= 49; number += 1) lastSeen.set(number, null);

  for (let drawIndex = 0; drawIndex <= referenceDrawIndex; drawIndex += 1) {
    for (const number of history[drawIndex].numbers) {
      lastSeen.set(number.value, drawIndex);
    }
  }

  const gaps = new Map<number, number>();
  for (let number = 1; number <= 49; number += 1) {
    const seenAt = lastSeen.get(number) ?? null;
    gaps.set(number, seenAt === null ? referenceDrawIndex + 1 : referenceDrawIndex - seenAt);
  }
  return gaps;
}

function groupNumbersByGap(gapsByNumber: Map<number, number>): Record<number, number[]> {
  const gapNumbers: Record<number, number[]> = {};
  for (const [number, gap] of gapsByNumber) {
    gapNumbers[gap] = [...(gapNumbers[gap] ?? []), number].sort(
      (left, right) => left - right,
    );
  }
  return gapNumbers;
}

function lastSeenGapIndices(
  draws: HistoryDraw[],
  referenceDrawIndex: number,
): Map<number, number> {
  const lastSeen = new Map<number, number>();
  for (let drawIndex = 0; drawIndex <= referenceDrawIndex; drawIndex += 1) {
    for (const number of draws[drawIndex].numbers) {
      lastSeen.set(number.gap, drawIndex);
    }
  }
  return lastSeen;
}

function gapGapsByDraw(draws: HistoryDraw[]): Map<number, number>[] {
  const lastSeen = new Map<number, number>();
  return draws.map((draw, drawIndex) => {
    const gapGaps = new Map<number, number>();
    const gapsInDraw = new Set(draw.numbers.map((number) => number.gap));
    for (const gap of gapsInDraw) {
      const previousIndex = lastSeen.get(gap);
      gapGaps.set(
        gap,
        previousIndex === undefined ? drawIndex : drawIndex - previousIndex - 1,
      );
      lastSeen.set(gap, drawIndex);
    }
    return gapGaps;
  });
}

export function buildLastSeenGapModel(
  history: HistoryDraw[],
  requestedCount: number,
  requestedReferenceOffset: number,
): LastSeenGapModel {
  const count = Math.min(Math.max(Math.trunc(requestedCount), 1), history.length);
  const draws = refreshDrawGaps(history.slice(-count));
  if (draws.length === 0) {
    return {
      draws: [],
      points: [],
      drawCount: 0,
      maxGap: 0,
      referenceGaps: [],
      referenceGapNumbers: {},
      maxReferenceOffset: 0,
      referenceDrawIndex: null,
      referenceDrawNumber: null,
    };
  }

  const maxReferenceOffset = draws.length - 1;
  const referenceDrawOffset = Math.min(
    Math.max(Math.trunc(requestedReferenceOffset), 0),
    maxReferenceOffset,
  );
  const referenceDrawIndex = draws.length - 1 - referenceDrawOffset;
  const referenceGapsByNumber = currentGapsForReference(history, referenceDrawOffset);
  const referenceGaps = [...referenceGapsByNumber.values()];
  const maxDrawGap = Math.max(
    ...draws.flatMap((draw) => draw.numbers.map((number) => number.gap)),
    0,
  );
  const maxGap = Math.max(maxDrawGap, ...referenceGaps);
  const lastSeen = lastSeenGapIndices(draws, referenceDrawIndex);
  const gapGaps = gapGapsByDraw(draws);
  const points: LastSeenGapPoint[] = [];

  draws.forEach((draw, drawIndex) => {
    for (const number of draw.numbers) {
      points.push({
        gap: number.gap,
        gapGap: gapGaps[drawIndex].get(number.gap) ?? 0,
        drawIndex,
        highlighted: lastSeen.get(number.gap) === drawIndex,
      });
    }
  });

  return {
    draws,
    points,
    drawCount: draws.length,
    maxGap,
    referenceGaps,
    referenceGapNumbers: groupNumbersByGap(referenceGapsByNumber),
    maxReferenceOffset,
    referenceDrawIndex,
    referenceDrawNumber: draws[referenceDrawIndex]?.drawNumber ?? null,
  };
}
