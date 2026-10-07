import { expect, it } from "vitest";
import { primaryStudyStrategies } from "./strategyStudy";
import type { StrategyId } from "../types";

it("groups duplicates only when their representative is available without changing saved selections", () => {
  const rows = (["mkgsv", "markov100", "svc"] as StrategyId[]).map(id => ({ id }));
  expect(primaryStudyStrategies(rows).map(row => row.id)).toEqual(["markov100", "svc"]);
  expect(rows.map(row => row.id)).toEqual(["mkgsv", "markov100", "svc"]);
  expect(primaryStudyStrategies([rows[0]])).toEqual([rows[0]]);
  expect(primaryStudyStrategies(rows)[0]).toBe(rows[1]);
});
