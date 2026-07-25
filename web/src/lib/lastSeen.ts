import type { HistoryDraw } from "../types";

export interface LastSeenPoint {
  number: number;
  gap: number;
  leftSpace: number;
  rightSpace: number;
  drawIndex: number;
  drawNumber: number;
  highlighted: boolean;
}

export interface LastSeenModel {
  draws: HistoryDraw[];
  points: LastSeenPoint[];
  drawCount: number;
  maxReferenceOffset: number;
  referenceDrawIndex: number | null;
  referenceDrawNumber: number | null;
}

export function buildLastSeenModel(
  history: HistoryDraw[],
  requestedCount: number,
  requestedReferenceOffset: number,
): LastSeenModel {
  const count = Math.min(Math.max(Math.trunc(requestedCount), 1), history.length);
  const draws = history.slice(-count);
  if (draws.length === 0) {
    return {
      draws: [],
      points: [],
      drawCount: 0,
      maxReferenceOffset: 0,
      referenceDrawIndex: null,
      referenceDrawNumber: null,
    };
  }

  const maxReferenceOffset = draws.length - 1;
  const referenceOffset = Math.min(
    Math.max(Math.trunc(requestedReferenceOffset), 0),
    maxReferenceOffset,
  );
  const referenceDrawIndex = draws.length - 1 - referenceOffset;
  const lastSeen = new Map<number, number | null>();
  for (let number = 1; number <= 49; number += 1) lastSeen.set(number, null);

  for (let drawIndex = 0; drawIndex <= referenceDrawIndex; drawIndex += 1) {
    for (const number of draws[drawIndex].numbers) {
      lastSeen.set(number.value, drawIndex);
    }
  }

  const points: LastSeenPoint[] = [];
  draws.forEach((draw, drawIndex) => {
    for (const number of draw.numbers) {
      points.push({
        number: number.value,
        gap: number.gap,
        leftSpace: number.leftSpace,
        rightSpace: number.rightSpace,
        drawIndex,
        drawNumber: draw.drawNumber,
        highlighted: lastSeen.get(number.value) === drawIndex,
      });
    }
  });

  return {
    draws,
    points,
    drawCount: draws.length,
    maxReferenceOffset,
    referenceDrawIndex,
    referenceDrawNumber: draws[referenceDrawIndex]?.drawNumber ?? null,
  };
}
