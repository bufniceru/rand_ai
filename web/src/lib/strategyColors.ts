import type { StrategyId } from "../types";
import {
  STRATEGY_FAMILIES,
  STRATEGY_FAMILY_BY_ID,
  type StrategyFamilyId,
} from "./strategyFamilies";

export const STRATEGY_TONE_POSITION_BY_ID = {
  freshness: 0,
  chi_square: 1,
  categorical_chi_square: 2,
  entropy: 3,
  bayesian: 4,
  proximity: 0,
  emd: 1,
  predictive_grid: 2,
  markov100: 0,
  mkgsv: 1,
  mkfr: 2,
  mksp: 3,
  mknp: 4,
  mkrd: 5,
  doublet_triplet_markov: 6,
  co_occurrence: 0,
  svc: 1,
  tbl: 2,
  sklearn_svm: 3,
  lag_logistic: 4,
  sparse_neural_ticket: 5,
  mixed: 0,
  cis: 1,
  decision_tree_selector: 2,
  residual_coverage: 3,
  chained: 4,
  randomness: 0,
  fresh_random: 1,
} as const satisfies Record<StrategyId, number>;

const DARK_MIX_COLOR = "#221F22";
const LIGHT_MIX_COLOR = "#FFFFFF";
const MAX_DARK_MIX = 0.24;
const MAX_LIGHT_MIX = 0.2;

const familyById = new Map(
  STRATEGY_FAMILIES.map((family) => [family.id, family]),
);

const familyStrategyCounts = Object.values(STRATEGY_FAMILY_BY_ID).reduce(
  (counts, familyId) => {
    counts[familyId] += 1;
    return counts;
  },
  Object.fromEntries(
    STRATEGY_FAMILIES.map((family) => [family.id, 0]),
  ) as Record<StrategyFamilyId, number>,
);

function parseHexColor(color: string): [number, number, number] {
  const normalized = color.replace(/^#/, "");
  return [0, 2, 4].map((offset) =>
    Number.parseInt(normalized.slice(offset, offset + 2), 16),
  ) as [number, number, number];
}

function mixHexColors(source: string, target: string, amount: number): string {
  const sourceChannels = parseHexColor(source);
  const targetChannels = parseHexColor(target);
  const channels = sourceChannels.map((channel, index) =>
    Math.round(channel + (targetChannels[index] - channel) * amount),
  );
  return `#${channels
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")}`.toUpperCase();
}

export function strategyFamilyColor(familyId: StrategyFamilyId): string {
  return familyById.get(familyId)?.color ?? "#727072";
}

export function strategyColor(strategyId: StrategyId): string {
  const familyId = STRATEGY_FAMILY_BY_ID[strategyId];
  const baseColor = strategyFamilyColor(familyId);
  const familySize = familyStrategyCounts[familyId];
  if (familySize <= 1) return baseColor;

  const position = STRATEGY_TONE_POSITION_BY_ID[strategyId];
  const normalizedPosition = (position / (familySize - 1)) * 2 - 1;
  return normalizedPosition < 0
    ? mixHexColors(
        baseColor,
        DARK_MIX_COLOR,
        Math.abs(normalizedPosition) * MAX_DARK_MIX,
      )
    : mixHexColors(
        baseColor,
        LIGHT_MIX_COLOR,
        normalizedPosition * MAX_LIGHT_MIX,
      );
}
