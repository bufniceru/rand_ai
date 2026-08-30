import { groupFrequencyFigure, numberFrequencyFigure } from "./figureBuilders";
import type {
  FigureSpec,
  StatisticsCommandPayload,
  StatisticsCommandRequest,
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

export type AppCommandResult = FigureCommandResult;

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
        figure: groupFrequencyFigure(payload.table),
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
