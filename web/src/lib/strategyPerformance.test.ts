import { expect, it } from "vitest";
import type { StrategyEfficacyRecord } from "../types";
import { strategyHitCurve, strategyPerformance } from "./strategyPerformance";

it("excludes unavailable forecasts while retaining actual zero-hit predictions", () => {
  const records = [{ strategyHits: {} }, { strategyHits: { svc: 0 } },
    { strategyHits: {} }, { strategyHits: { svc: 4 } }] as StrategyEfficacyRecord[];
  expect(strategyPerformance(records, "svc")).toEqual({ drawCount: 2, hits: 4, average: 2,
    distribution: [1, 0, 0, 0, 1, 0, 0] });
  expect(strategyHitCurve(records, "svc")).toEqual([null, 0, null, 2]);
  expect(strategyHitCurve(records, "svc", 2)).toEqual([null, 0, null, 4]);
  expect(strategyPerformance(records, "tbl").drawCount).toBe(0);
});
