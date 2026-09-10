import { freshnessGapDistribution, freshnessGapHitRate, groupFrequencyFigure, numberFrequencyFigure } from "./figureBuilders";
import type {
  FigureSpec,
  StatisticsCommandPayload,
  StatisticsCommandRequest,
  TablePayload,
} from "../types";

export interface CommandAvailabilityContext {
  hasDataset: boolean;
}

export interface CommandExecutionContext extends CommandAvailabilityContext {
  borderSpace: number;
  runStatisticsCommand(
    request: StatisticsCommandRequest,
  ): Promise<StatisticsCommandPayload>;
}

export interface FigureCommandResult {
  kind: "figure";
  commandId: string;
  title: string;
  subtitle: string;
  figure: FigureSpec;
}

export interface GapStatisticsCommandResult {
  kind: "gap-statistics";
  commandId: string;
  title: string;
  subtitle: string;
  table: TablePayload;
  figures: Record<string, FigureSpec>;
}

export type AppCommandResult = FigureCommandResult | GapStatisticsCommandResult;

export interface AppCommand {
  id: string;
  title: string;
  category: string;
  keywords: readonly string[];
  disabledReason(context: CommandAvailabilityContext): string | null;
  execute(context: CommandExecutionContext): Promise<AppCommandResult>;
}

export type CommandResultOverlayState =
  | { status: "loading"; title: string }
  | { status: "ready"; result: AppCommandResult }
  | { status: "error"; title: string; message: string };

export const applicationCommands: readonly AppCommand[] = [
  {
    id: "statistics.gap-statistics",
    title: "Gap Statistics vs Fair Randomness",
    category: "Statistics",
    keywords: ["gaps", "freshness", "hits", "opportunities", "random", "baseline", "waiting"],
    disabledReason: ({ hasDataset }) =>
      hasDataset ? null : "Analyze a dataset first",
    execute: async (context) => {
      const payload = await context.runStatisticsCommand({ id: "statistics.gap-statistics" });
      if (payload.id !== "statistics.gap-statistics") {
        throw new Error("Unexpected Gap Statistics response.");
      }
      return {
        kind: "gap-statistics",
        commandId: payload.id,
        title: "Statistics: Gap Statistics vs Fair Randomness",
        subtitle: `${payload.datasetName} · ${payload.drawCount.toLocaleString()} draws`,
        table: payload.table,
        figures: {
          freshness_gap_distribution: freshnessGapDistribution(payload.table),
          freshness_gap_hit_rate: freshnessGapHitRate(payload.table),
        },
      };
    },
  },
  {
    id: "statistics.number-frequency",
    title: "Number Frequency",
    category: "Statistics",
    keywords: ["numbers", "frequency", "appearances", "expected"],
    disabledReason: ({ hasDataset }) =>
      hasDataset ? null : "Analyze a dataset first",
    execute: async (context) => {
      const payload = await context.runStatisticsCommand(
        { id: "statistics.number-frequency" },
      );
      if (payload.id !== "statistics.number-frequency") {
        throw new Error("Unexpected Number Frequency response.");
      }
      return {
        kind: "figure",
        commandId: payload.id,
        title: "Statistics: Number Frequency",
        subtitle: `${payload.datasetName} · ${payload.drawCount.toLocaleString()} draws`,
        figure: numberFrequencyFigure(payload.table),
      };
    },
  },
  {
    id: "statistics.group-frequency",
    title: "Group Frequency",
    category: "Statistics",
    keywords: ["groups", "border", "frequency", "count"],
    disabledReason: ({ hasDataset }) =>
      hasDataset ? null : "Analyze a dataset first",
    execute: async (context) => {
      const payload = await context.runStatisticsCommand({
        id: "statistics.group-frequency",
        borderSpace: context.borderSpace,
      });
      if (payload.id !== "statistics.group-frequency") {
        throw new Error("Unexpected Group Frequency response.");
      }
      return {
        kind: "figure",
        commandId: payload.id,
        title: "Statistics: Group Frequency",
        subtitle: `${payload.datasetName} · ${payload.drawCount.toLocaleString()} draws · Border space ${payload.borderSpace}`,
        figure: groupFrequencyFigure(payload.table, {
          datasetName: payload.datasetName,
          borderSpace: payload.borderSpace,
        }),
      };
    },
  },
];

function fuzzyIncludes(value: string, query: string): boolean {
  let queryIndex = 0;
  for (const character of value) {
    if (character === query[queryIndex]) queryIndex += 1;
    if (queryIndex === query.length) return true;
  }
  return query.length === 0;
}

export function filterApplicationCommands(
  commands: readonly AppCommand[],
  query: string,
): AppCommand[] {
  const normalized = query.trim().replace(/^>\s*/, "").toLowerCase();
  if (!normalized) return [...commands];
  const tokens = normalized.split(/\s+/);
  return commands.filter((command) => {
    const searchable = [
      command.category,
      command.title,
      ...command.keywords,
    ].join(" ").toLowerCase();
    return tokens.every(
      (token) => searchable.includes(token) || fuzzyIncludes(searchable, token),
    );
  });
}
