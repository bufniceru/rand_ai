import { describe, expect, it, vi } from "vitest";
import type {
  StatisticsCommandPayload,
  StatisticsCommandRequest,
} from "../types";
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

function groupPayload(): StatisticsCommandPayload {
  return {
    id: "statistics.group-frequency",
    datasetName: "draws.pkl",
    drawCount: 3,
    borderSpace: 7,
    table: {
      columns: ["group_count", "count"],
      rows: [
        { group_count: 1, count: 1 },
        { group_count: 2, count: 0 },
        { group_count: 3, count: 1 },
        { group_count: 4, count: 0 },
        { group_count: 5, count: 1 },
        { group_count: 6, count: 0 },
      ],
    },
  };
}

function command(id: string) {
  const value = applicationCommands.find((entry) => entry.id === id);
  if (!value) throw new Error(`Missing command: ${id}`);
  return value;
}

describe("application command registry", () => {
  it("filters commands by category, title, keywords, and palette prefix", () => {
    expect(filterApplicationCommands(applicationCommands, "frequency")).toHaveLength(2);
    expect(filterApplicationCommands(applicationCommands, "> statistics")).toHaveLength(2);
    expect(filterApplicationCommands(applicationCommands, "appearances expected")).toHaveLength(1);
    expect(filterApplicationCommands(applicationCommands, "border count").map((item) => item.id)).toEqual([
      "statistics.group-frequency",
    ]);
    expect(filterApplicationCommands(applicationCommands, "unrelated")).toEqual([]);
  });

  it("explains dataset-dependent availability", () => {
    for (const entry of applicationCommands) {
      expect(entry.disabledReason({ hasDataset: false })).toBe(
        "Analyze a dataset first",
      );
      expect(entry.disabledReason({ hasDataset: true })).toBeNull();
    }
  });

  it("executes Number Frequency and builds observed and expected traces", async () => {
    const runStatisticsCommand = vi.fn(async (
      _request: StatisticsCommandRequest,
    ) => payload());
    const result = await command("statistics.number-frequency").execute({
      hasDataset: true,
      borderSpace: 7,
      runStatisticsCommand,
    });

    expect(runStatisticsCommand).toHaveBeenCalledWith({ id: "statistics.number-frequency" });
    expect(result.title).toBe("Statistics: Number Frequency");
    expect(result.subtitle).toContain("3 draws");
    expect(result.figure.data).toHaveLength(2);
    expect(result.figure.data[0].x).toEqual([1, 2]);
    expect(result.figure.data[0].y).toEqual([2, 1]);
    expect(result.figure.data[1].y).toEqual([18 / 49, 18 / 49]);
  });

  it("executes Group Frequency with the current border and one count trace", async () => {
    const runStatisticsCommand = vi.fn(async (
      _request: StatisticsCommandRequest,
    ) => groupPayload());
    const result = await command("statistics.group-frequency").execute({
      hasDataset: true,
      borderSpace: 7,
      runStatisticsCommand,
    });

    expect(runStatisticsCommand).toHaveBeenCalledWith({
      id: "statistics.group-frequency",
      borderSpace: 7,
    });
    expect(result.title).toBe("Statistics: Group Frequency");
    expect(result.subtitle).toContain("Border space 7");
    expect(result.figure.data).toHaveLength(1);
    expect(result.figure.data[0].x).toEqual([1, 2, 3, 4, 5, 6]);
    expect(result.figure.data[0].y).toEqual([1, 0, 1, 0, 1, 0]);
    expect(result.figure.layout).toMatchObject({
      xaxis: { tickvals: [1, 2, 3, 4, 5, 6] },
      yaxis: { title: { text: "Draws" }, rangemode: "tozero" },
    });
  });
});
