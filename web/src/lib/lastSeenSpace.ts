import type { HistoryDraw } from "../types";

export interface LastSeenSpacePoint {
  space: number;
  drawIndex: number;
  highlighted: boolean;
}

export interface LastSeenSpaceModel {
  draws: HistoryDraw[];
  points: LastSeenSpacePoint[];
  drawCount: number;
  maxSpace: number;
  maxReferenceOffset: number;
  referenceDrawIndex: number | null;
  referenceDrawNumber: number | null;
}

export function internalSpacesForDraw(draw: HistoryDraw): number[] {
  const sorted = draw.numbers
    .slice()
    .sort((left, right) => left.value - right.value);
  return sorted.slice(0, -1).map((number) => number.rightSpace);
}

export function buildLastSeenSpaceModel(
  history: HistoryDraw[],
  requestedCount: number,
  requestedReferenceOffset: number,
): LastSeenSpaceModel {
  const count = Math.min(
    Math.max(Math.trunc(requestedCount), 1),
    history.length,
  );
  const draws = history.slice(-count);
  if (draws.length === 0) {
    return {
      draws: [],
      points: [],
      drawCount: 0,
      maxSpace: 0,
      maxReferenceOffset: 0,
      referenceDrawIndex: null,
      referenceDrawNumber: null,
    };
  }

  const spacesByDraw = draws.map(internalSpacesForDraw);
  const maxSpace = Math.max(...spacesByDraw.flat(), 0);
  const maxReferenceOffset = draws.length - 1;
  const referenceOffset = Math.min(
    Math.max(Math.trunc(requestedReferenceOffset), 0),
    maxReferenceOffset,
  );
  const referenceDrawIndex = draws.length - 1 - referenceOffset;
  const lastSeen = new Map<number, number | null>();
  for (let space = 0; space <= maxSpace; space += 1) {
    lastSeen.set(space, null);
  }
  for (let drawIndex = 0; drawIndex <= referenceDrawIndex; drawIndex += 1) {
    for (const space of spacesByDraw[drawIndex]!) {
      lastSeen.set(space, drawIndex);
    }
  }

  const points: LastSeenSpacePoint[] = [];
  spacesByDraw.forEach((spaces, drawIndex) => {
    for (const space of spaces) {
      points.push({
        space,
        drawIndex,
        highlighted: lastSeen.get(space) === drawIndex,
      });
    }
  });

  return {
    draws,
    points,
    drawCount: draws.length,
    maxSpace,
    maxReferenceOffset,
    referenceDrawIndex,
    referenceDrawNumber: draws[referenceDrawIndex]?.drawNumber ?? null,
  };
}
