import type { StrategyId, StrategyPlugin } from "../types";

export function enabledStrategyPlugins(
  plugins: readonly StrategyPlugin[],
  enabledIds: ReadonlySet<StrategyId> | readonly StrategyId[],
): StrategyPlugin[] {
  const enabled = enabledIds instanceof Set ? enabledIds : new Set(enabledIds);
  return plugins.filter((plugin) => enabled.has(plugin.id));
}

export function orderedStrategySelection(
  plugins: readonly StrategyPlugin[],
  selectedIds: ReadonlySet<StrategyId> | readonly StrategyId[],
): StrategyId[] {
  const selected =
    selectedIds instanceof Set ? selectedIds : new Set(selectedIds);
  return plugins
    .map((plugin) => plugin.id)
    .filter((strategyId) => selected.has(strategyId));
}

export function strategySelectionsEqual(
  plugins: readonly StrategyPlugin[],
  left: ReadonlySet<StrategyId> | readonly StrategyId[],
  right: ReadonlySet<StrategyId> | readonly StrategyId[],
): boolean {
  const orderedLeft = orderedStrategySelection(plugins, left);
  const orderedRight = orderedStrategySelection(plugins, right);
  return (
    orderedLeft.length === orderedRight.length &&
    orderedLeft.every((strategyId, index) => strategyId === orderedRight[index])
  );
}
