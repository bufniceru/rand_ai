import { describe, expect, it, vi } from "vitest";
import type { StatisticsCommandPayload } from "../types";
import {
  applicationCommands,
  filterApplicationCommands,
} from "./commands";

function payload(): StatisticsCommandPayload {
  return {
    id: "statistics.number-frequency",
    datasetName: "draws.pkl",
    drawCount: 3,
    table: {
      columns: ["number", "count", "expected_count"],
      rows: [
        { number: 1, count: 2, expected_count: 18 / 49 },
        { number: 2, count: 1, expected_count: 18 / 49 },
      ],
    },
  };
}

describe("application command registry", () => {
  it("filters commands by category, title, keywords, and palette prefix", () => {
    expect(filterApplicationCommands(applicationCommands, "frequency")).toHaveLength(1);
    expect(filterApplicationCommands(applicationCommands, "> statistics")).toHaveLength(1);
    expect(filterApplicationCommands(applicationCommands, "appearances expected")).toHaveLength(1);
    expect(filterApplicationCommands(applicationCommands, "unrelated")).toEqual([]);
  });

  it("explains dataset-dependent availability", () => {
    const command = applicationCommands[0];
    expect(command.disabledReason({ hasDataset: false })).toBe(
      "Analyze a dataset first",
    );
    expect(command.disabledReason({ hasDataset: true })).toBeNull();
  });

  it("executes Number Frequency and builds observed and expected traces", async () => {
    const runStatisticsCommand = vi.fn(async () => payload());
    const result = await applicationCommands[0].execute({
      hasDataset: true,
      runStatisticsCommand,
    });

    expect(runStatisticsCommand).toHaveBeenCalledWith("statistics.number-frequency");
    expect(result.title).toBe("Statistics: Number Frequency");
    expect(result.subtitle).toContain("3 draws");
    expect(result.figure.data).toHaveLength(2);
    expect(result.figure.data[0].x).toEqual([1, 2]);
    expect(result.figure.data[0].y).toEqual([2, 1]);
    expect(result.figure.data[1].y).toEqual([18 / 49, 18 / 49]);
  });
});
