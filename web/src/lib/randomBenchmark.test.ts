import { describe, expect, it } from "vitest";
import {
  runPooledRandomBenchmarks,
  runRandomBenchmark,
} from "./randomBenchmark";

describe("random benchmarks", () => {
  it("builds pooled summaries for each requested strategy count", () => {
    const summaries = runPooledRandomBenchmarks(4, [3, 1, 3], 128);
    const single = summaries.get(1);
    const pooled = summaries.get(3);

    expect([...summaries.keys()]).toEqual([1, 3]);
    expect(single?.simulations).toBe(128);
    expect(single?.sortedTotalHits).toHaveLength(128);
    expect(pooled?.sortedTotalHits).toHaveLength(128);
    expect(pooled?.meanHits ?? 0).toBeGreaterThanOrEqual(single?.meanHits ?? 0);
    expect(pooled?.lower95Hits ?? 0).toBeLessThanOrEqual(
      pooled?.upper95Hits ?? 0,
    );
  });

  it("preserves the single-strategy benchmark API", () => {
    const summary = runRandomBenchmark(2, 32);

    expect(summary.simulations).toBe(32);
    expect(summary.sortedTotalHits).toHaveLength(32);
  });

  it("rejects invalid pooled benchmark inputs", () => {
    expect(() => runPooledRandomBenchmarks(0, [1], 10)).toThrow(RangeError);
    expect(() => runPooledRandomBenchmarks(1, [], 10)).toThrow(RangeError);
    expect(() => runPooledRandomBenchmarks(1, [0], 10)).toThrow(RangeError);
    expect(() => runPooledRandomBenchmarks(1, [1.5], 10)).toThrow(RangeError);
  });
});
