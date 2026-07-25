<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import type { StrategyId, StrategyPlugin } from "../types";

const props = defineProps<{
  plugins: StrategyPlugin[];
  enabledStrategyIds: StrategyId[];
  saving: boolean;
}>();

const emit = defineEmits<{
  cancel: [];
  save: [strategyIds: StrategyId[]];
}>();

const dialog = ref<HTMLElement | null>(null);
const selectedStrategyIds = ref<Set<StrategyId>>(
  new Set(props.enabledStrategyIds),
);

const strategyDescriptions: Record<StrategyId, string> = {
  proximity: "Nearest-neighbor spacing profile.",
  freshness: "Gap recency and historical hit-rate model.",
  emd: "Earth-mover similarity to historical draw vectors.",
  randomness: "Deterministic random comparison baseline.",
  fresh_random: "Seeded random ranking guided 35% by freshness.",
  chi_square: "Signed frequency deviation from uniform random expectation.",
  entropy: "Gap-entropy history with overdue adjustment.",
  markov100: "Recency-weighted gap-state Markov model.",
  mkfr: "Variable-order D/!D context transition lift.",
  bayesian: "Bayesian gap-state posterior ranking.",
  predictive_grid: "Six-component Markov, transition, and history score grid.",
  mixed: "Weighted consensus of four complementary strategies.",
  svc: "Online support-vector classification model.",
  tbl: "Temporal behavior learning ensemble.",
  cis: "Online learner combining ten strategy experts.",
};

const selectedCount = computed(() => selectedStrategyIds.value.size);
const selectedIdsInPluginOrder = computed(() =>
  props.plugins
    .map((plugin) => plugin.id)
    .filter((strategyId) => selectedStrategyIds.value.has(strategyId)),
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

onMounted(() => dialog.value?.focus());
</script>

<template>
  <div
    class="modal-backdrop"
    role="presentation"
    @click.self="!saving && emit('cancel')"
  >
    <section
      ref="dialog"
      class="settings-dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
      tabindex="-1"
      @keydown.esc="!saving && emit('cancel')"
    >
      <header class="settings-dialog-heading">
        <div>
          <p class="eyebrow">Application settings</p>
          <h2 id="settings-title">Prediction strategies</h2>
          <p>
            Choose which strategy models are calculated when a dataset is
            analyzed.
          </p>
        </div>
        <strong>{{ selectedCount }} of {{ plugins.length }} enabled</strong>
      </header>

      <div class="settings-selection-actions">
        <button
          class="button secondary"
          type="button"
          :disabled="saving || selectedCount === plugins.length"
          @click="enableAll"
        >
          Enable all
        </button>
        <button
          class="button secondary"
          type="button"
          :disabled="saving || selectedCount === 0"
          @click="disableAll"
        >
          Disable all
        </button>
      </div>

      <div class="strategy-settings-grid">
        <label
          v-for="plugin in plugins"
          :key="plugin.id"
          class="strategy-setting-option"
          :class="{ enabled: selectedStrategyIds.has(plugin.id) }"
        >
          <input
            type="checkbox"
            :checked="selectedStrategyIds.has(plugin.id)"
            :disabled="saving"
            @change="
              setStrategyEnabled(
                plugin.id,
                ($event.target as HTMLInputElement).checked,
              )
            "
          >
          <span class="strategy-setting-check" aria-hidden="true" />
          <span>
            <strong>{{ plugin.label }}</strong>
            <small>{{ strategyDescriptions[plugin.id] }}</small>
          </span>
        </label>
      </div>

      <p class="settings-dialog-note">
        Saving changes reanalyzes the active dataset so prediction windows use
        the new strategy selection.
      </p>

      <footer class="dialog-actions">
        <button
          class="button secondary"
          type="button"
          :disabled="saving"
          @click="emit('cancel')"
        >
          Cancel
        </button>
        <button
          class="button primary"
          type="button"
          :disabled="saving"
          @click="emit('save', selectedIdsInPluginOrder)"
        >
          {{ saving ? "Saving…" : "Save settings" }}
        </button>
      </footer>
    </section>
  </div>
</template>
