<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { groupStrategiesByFamily } from "../lib/strategyFamilies";
import {
  orderedStrategySelection,
  strategySelectionsEqual,
} from "../lib/strategySelection";
import type { StrategyId, StrategyPlugin } from "../types";

const props = defineProps<{
  plugins: StrategyPlugin[];
  enabledStrategyIds: StrategyId[];
  busy: boolean;
}>();

const emit = defineEmits<{
  apply: [strategyIds: StrategyId[]];
}>();

const selectedStrategyIds = ref<Set<StrategyId>>(
  new Set(props.enabledStrategyIds),
);

const groupedPlugins = computed(() => groupStrategiesByFamily(props.plugins));
const selectedCount = computed(() => selectedStrategyIds.value.size);
const hasChanges = computed(
  () =>
    !strategySelectionsEqual(
      props.plugins,
      props.enabledStrategyIds,
      selectedStrategyIds.value,
    ),
);

watch(
  () => props.enabledStrategyIds,
  (enabledStrategyIds) => {
    selectedStrategyIds.value = new Set(enabledStrategyIds);
  },
);

function setStrategyEnabled(strategyId: StrategyId, enabled: boolean): void {
  const next = new Set(selectedStrategyIds.value);
  if (enabled) next.add(strategyId);
  else next.delete(strategyId);
  selectedStrategyIds.value = next;
}

function enableAll(): void {
  selectedStrategyIds.value = new Set(
    props.plugins.map((plugin) => plugin.id),
  );
}

function disableAll(): void {
  selectedStrategyIds.value = new Set();
}

function resetSelection(): void {
  selectedStrategyIds.value = new Set(props.enabledStrategyIds);
}

function applySelection(): void {
  if (!hasChanges.value || props.busy) return;
  emit(
    "apply",
    orderedStrategySelection(props.plugins, selectedStrategyIds.value),
  );
}
</script>

<template>
  <section class="strategy-selection-panel" aria-label="Prediction strategy selection">
    <header>
      <div>
        <strong>Calculated strategies</strong>
        <small>Unchecked strategies are excluded after reanalysis.</small>
      </div>
      <b>{{ selectedCount }} of {{ plugins.length }}</b>
    </header>

    <div class="strategy-selection-actions" aria-label="Strategy selection actions">
      <button
        type="button"
        :disabled="busy || selectedCount === plugins.length"
        @click="enableAll"
      >
        Enable all
      </button>
      <button
        type="button"
        :disabled="busy || selectedCount === 0"
        @click="disableAll"
      >
        Disable all
      </button>
      <button type="button" :disabled="busy || !hasChanges" @click="resetSelection">
        Reset
      </button>
    </div>

    <div class="strategy-selection-families">
      <section
        v-for="family in groupedPlugins"
        :key="family.id"
        class="strategy-selection-family"
        :aria-labelledby="`strategy-selection-family-${family.id}`"
      >
        <h3 :id="`strategy-selection-family-${family.id}`">
          {{ family.label }}
        </h3>
        <label
          v-for="plugin in family.strategies"
          :key="plugin.id"
          :class="{ enabled: selectedStrategyIds.has(plugin.id) }"
        >
          <input
            type="checkbox"
            :checked="selectedStrategyIds.has(plugin.id)"
            :disabled="busy"
            @change="
              setStrategyEnabled(
                plugin.id,
                ($event.target as HTMLInputElement).checked,
              )
            "
          />
          <span>{{ plugin.label }}</span>
        </label>
      </section>
    </div>

    <footer>
      <small v-if="hasChanges">Changes are not applied yet.</small>
      <small v-else>Selection is up to date.</small>
      <button
        type="button"
        :disabled="busy || !hasChanges"
        @click="applySelection"
      >
        {{ busy ? "Applying…" : "Apply selection" }}
      </button>
    </footer>
  </section>
</template>

<style scoped>
.strategy-selection-panel {
  min-width: 0;
  display: grid;
  gap: 7px;
}

.strategy-selection-panel > header,
.strategy-selection-panel > footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.strategy-selection-panel > header > div {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.strategy-selection-panel > header strong,
.strategy-selection-panel > header b {
  color: var(--monokai-fg, #173d64);
  font-size: 11px;
}

.strategy-selection-panel small {
  color: var(--monokai-muted, #687d92);
  font-size: 9px;
  font-weight: 750;
}

.strategy-selection-actions {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 3px;
}

.strategy-selection-panel button {
  min-height: 30px;
  padding: 5px 7px;
  border: 1px solid var(--monokai-border, #c8d7e7);
  border-radius: 0;
  color: var(--monokai-fg, #38536f);
  background: var(--monokai-raised, #eef5fb);
  font-size: 9px;
  font-weight: 900;
  cursor: pointer;
}

.strategy-selection-panel button:hover:not(:disabled) {
  border-color: var(--monokai-cyan, #5f91c2);
}

.strategy-selection-panel button:focus-visible,
.strategy-selection-family label:has(input:focus-visible) {
  outline: 2px solid var(--monokai-yellow, #3264ad);
  outline-offset: 2px;
}

.strategy-selection-panel button:disabled {
  cursor: default;
  opacity: 0.5;
}

.strategy-selection-families {
  display: grid;
  gap: 5px;
}

.strategy-selection-family {
  min-width: 0;
  display: grid;
  gap: 3px;
  break-inside: avoid;
}

.strategy-selection-family h3 {
  margin: 0;
  padding: 5px 7px 3px;
  border-bottom: 1px solid var(--monokai-border, #c8d8e7);
  color: var(--monokai-muted, #496984);
  font-size: 8px;
  font-weight: 950;
  letter-spacing: 0.07em;
  line-height: 1.25;
  text-transform: uppercase;
}

.strategy-selection-family label {
  min-height: 31px;
  display: grid;
  grid-template-columns: 17px minmax(0, 1fr);
  align-items: center;
  gap: 7px;
  padding: 4px 7px;
  border: 1px solid var(--monokai-border, #d8e4ef);
  color: var(--monokai-muted, #5b7188);
  background: var(--monokai-surface, #f5f9fd);
  font-size: 9px;
  font-weight: 850;
  cursor: pointer;
}

.strategy-selection-family label.enabled {
  border-color: var(--monokai-cyan, #86acd4);
  color: var(--monokai-fg, #173d64);
  background: color-mix(
    in srgb,
    var(--monokai-cyan, #d9ebfb) 14%,
    var(--monokai-surface, #f5f9fd)
  );
}

.strategy-selection-family input {
  width: 15px;
  height: 15px;
  margin: 0;
  accent-color: var(--monokai-yellow, #3479b2);
}

.strategy-selection-panel > footer {
  position: sticky;
  bottom: 0;
  padding-top: 7px;
  border-top: 1px solid var(--monokai-border, #d8e2ec);
  background: var(--monokai-surface, rgba(255, 255, 255, 0.97));
}

.strategy-selection-panel > footer button {
  color: var(--monokai-deep, #ffffff);
  border-color: var(--monokai-yellow, #5f91c2);
  background: var(--monokai-yellow, #3479b2);
}

@media print {
  .strategy-selection-actions,
  .strategy-selection-panel > footer {
    display: none !important;
  }
}
</style>
