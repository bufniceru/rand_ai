<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import type { StrategyId, StrategyPlugin } from "../types";

const props = defineProps<{
  plugins: StrategyPlugin[];
  enabledStrategyIds: StrategyId[];
  lastSeenDrawCount: number;
  maxLastSeenDrawCount: number;
  saving: boolean;
}>();

const emit = defineEmits<{
  cancel: [];
  save: [strategyIds: StrategyId[], lastSeenDrawCount: number];
}>();

const dialog = ref<HTMLElement | null>(null);
const selectedStrategyIds = ref<Set<StrategyId>>(
  new Set(props.enabledStrategyIds),
);
const selectedLastSeenDrawCount = ref(props.lastSeenDrawCount);

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
  mksp: "Order-20 space analogues decoded into complete valid draws.",
  mknp: "Order-20 normalized-position analogues decoded into valid translated draws.",
  mkrd: "Order-20 relative-shape and dispersion analogues decoded into valid translated draws.",
  bayesian: "Hierarchically shrunk Bayesian model of gap and recent-number posteriors.",
  predictive_grid: "Seven-component history grid with earth-mover similarity.",
  co_occurrence: "Pair counts stabilized with candidate-adjusted recent lift.",
  doublet_triplet_markov:
    "Consecutive doublet/triplet recurrence with first-order next-draw Markov transitions.",
  mixed: "Weighted consensus of four complementary strategies.",
  svc: "Online support-vector classification model.",
  tbl: "Temporal behavior learning ensemble.",
  sklearn_svm:
    "Scikit-learn online linear SVM with temporal, expert-rank, and prior-efficacy inputs.",
  lag_logistic:
    "Compact probability model combining exact three-draw lags with gap and frequency context.",
  cis: "Online learner combining ten strategy experts.",
  residual_coverage:
    "Diversity-first complement covering numbers outside every base Top-6.",
  chained:
    "Sequential effectiveness-weighted consensus, relationships, shape, and residual coverage.",
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

function normalizedLastSeenDrawCount(): number {
  return Math.min(
    Math.max(Math.trunc(selectedLastSeenDrawCount.value || 1), 1),
    props.maxLastSeenDrawCount,
  );
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
          <h2 id="settings-title">Settings</h2>
          <p>
            Configure the Last Seen displays and choose which prediction
            strategies are calculated.
          </p>
        </div>
        <strong>{{ selectedCount }} of {{ plugins.length }} enabled</strong>
      </header>

      <section class="settings-display-section">
        <div>
          <strong>Last Seen views</strong>
          <small>One draw-window size is shared by Numbers, Gaps, and Spaces.</small>
        </div>
        <label class="settings-number-field">
          <span>Draw count</span>
          <input
            v-model.number="selectedLastSeenDrawCount"
            :max="maxLastSeenDrawCount"
            min="1"
            type="number"
            :disabled="saving"
            @change="selectedLastSeenDrawCount = normalizedLastSeenDrawCount()"
          >
        </label>
      </section>

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
        Draw-count changes apply immediately. Strategy changes reanalyze the
        active dataset so prediction windows use the new selection.
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
          @click="emit('save', selectedIdsInPluginOrder, normalizedLastSeenDrawCount())"
        >
          {{ saving ? "Saving…" : "Save settings" }}
        </button>
      </footer>
    </section>
  </div>
</template>
