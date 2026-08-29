<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { groupStrategiesByFamily } from "../lib/strategyFamilies";
import type { StrategyId, StrategyPlugin } from "../types";

const props = defineProps<{
  plugins: StrategyPlugin[];
  enabledStrategyIds: StrategyId[];
  lastSeenDrawCount: number;
  maxLastSeenDrawCount: number;
  borderSpace: number;
  targetGroupCount: number | null;
  saving: boolean;
}>();

const emit = defineEmits<{
  cancel: [];
  appearance: [];
  save: [
    strategyIds: StrategyId[],
    lastSeenDrawCount: number,
    borderSpace: number,
    targetGroupCount: number | null,
  ];
}>();

const dialog = ref<HTMLElement | null>(null);
const selectedStrategyIds = ref<Set<StrategyId>>(
  new Set(props.enabledStrategyIds),
);
const selectedLastSeenDrawCount = ref(props.lastSeenDrawCount);
const selectedBorderSpace = ref(props.borderSpace);
const selectedTargetGroupCount = ref<number | null>(props.targetGroupCount);
const groupCountChoices = [1, 2, 3, 4, 5, 6];

const strategyDescriptions: Record<StrategyId, string> = {
  proximity: "Nearest-neighbor spacing profile.",
  freshness: "Gap recency and historical hit-rate model.",
  emd: "Earth-mover similarity to historical draw vectors.",
  recurrence_dynamics:
    "Experimental V2 three-draw recurrence using eight causal value analogues.",
  randomness: "Deterministic random comparison baseline.",
  fresh_random: "Seeded random ranking guided 35% by freshness.",
  chi_square: "Signed frequency deviation from uniform random expectation.",
  categorical_chi_square:
    "Per-number exact gap and left/right-space probability with hierarchical chi-square backoff.",
  entropy: "Gap-entropy history with overdue adjustment.",
  markov100: "Recency-weighted gap-state Markov model.",
  mkgsv:
    "Experimental ticket-level gap-space motif research; its v3 promotion gate failed, so output exactly matches Markov 100.",
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
  svc_recurrence_hybrid:
    "Experimental leakage-free rank blend of SVC and Recurrence Dynamics, weighted by cumulative walk-forward effectiveness.",
  svc_recurrence_proximity_hybrid:
    "Experimental rank blend reserving 25% for Proximity and adaptively splitting 75% between SVC and Recurrence Dynamics.",
  srph_residual_diversity_hybrid:
    "Experimental guarded residual blend retaining SRPH unless a fixed 10% candidate blend has higher cumulative walk-forward quality.",
  srph_minimax_regret_hybrid:
    "Experimental minimax-regret selector over 503 guarded SRPH and residual blends using completed 40-draw blocks.",
  tbl: "Temporal behavior learning ensemble.",
  sklearn_svm:
    "Scikit-learn online linear SVM with temporal, expert-rank, and prior-efficacy inputs.",
  lag_logistic:
    "Compact probability model combining exact three-draw lags with gap and frequency context.",
  sparse_neural_ticket:
    "Experimental frozen five-seed neural ticket; its historical promotion gate failed.",
  cis: "Online learner combining ten strategy experts.",
  decision_tree_selector:
    "Leakage-safe decision tree selecting one stable expert for the next draw.",
  border_group_statistical:
    "Smoothed historical frequencies of circular border-group signatures.",
  border_group_markov:
    "Next-signature transitions with statistical backoff.",
  border_group_bayesian:
    "Bayesian posterior over signatures from recent circular-group context.",
  border_group_ml:
    "Online multinomial model of recent spaces, signatures, and trends.",
  border_group_hybrid:
    "Walk-forward log-loss-weighted blend of all border-group models.",
  residual_coverage:
    "Diversity-first complement covering numbers outside every base Top-6.",
  chained:
    "Sequential effectiveness-weighted consensus, relationships, shape, and residual coverage.",
};

const selectedCount = computed(() => selectedStrategyIds.value.size);
const groupedPlugins = computed(() => groupStrategiesByFamily(props.plugins));
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

function normalizedBorderSpace(): number {
  return Math.min(
    Math.max(Math.trunc(selectedBorderSpace.value || 0), 0),
    43,
  );
}

function groupCountFeasible(groupCount: number): boolean {
  return groupCount === 1 || groupCount * (normalizedBorderSpace() + 1) <= 43;
}

function normalizedTargetGroupCount(): number | null {
  const value = selectedTargetGroupCount.value;
  return value !== null && Number.isInteger(value) && groupCountFeasible(value)
    ? value
    : null;
}

function applyBorderSpace(): void {
  selectedBorderSpace.value = normalizedBorderSpace();
  if (normalizedTargetGroupCount() === null) {
    selectedTargetGroupCount.value = null;
  }
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
            Configure shared analysis behavior, Last Seen displays, and the
            prediction strategies that are calculated.
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

      <section class="settings-display-section">
        <div>
          <strong>Border groups</strong>
          <small>
            Spaces up to this inclusive value connect numbers; larger spaces
            separate groups. This affects analysis, predictions, exports, and portfolios.
          </small>
        </div>
        <div class="settings-border-fields">
          <label class="settings-number-field">
            <span>Border space</span>
            <input
              v-model.number="selectedBorderSpace"
              min="0"
              max="43"
              type="number"
              :disabled="saving"
              @change="applyBorderSpace"
            >
          </label>
          <label class="settings-number-field">
            <span>Predicted groups</span>
            <select v-model="selectedTargetGroupCount" :disabled="saving">
              <option :value="null">Automatic</option>
              <option
                v-for="groupCount in groupCountChoices"
                :key="groupCount"
                :value="groupCount"
                :disabled="!groupCountFeasible(groupCount)"
              >
                {{ groupCount }} {{ groupCount === 1 ? "group" : "groups" }}
              </option>
            </select>
          </label>
        </div>
      </section>

      <section class="settings-display-section settings-appearance-section">
        <div>
          <strong>Appearance</strong>
          <small>Customize every application, chart, strategy, and component color.</small>
        </div>
        <button
          class="button secondary"
          type="button"
          :disabled="saving"
          @click="emit('appearance')"
        >
          Color templates…
        </button>
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

      <div class="strategy-settings-families">
        <section
          v-for="family in groupedPlugins"
          :key="family.id"
          class="settings-strategy-family"
          :style="{ '--family-color': family.color }"
          :aria-labelledby="`settings-strategy-family-${family.id}`"
        >
          <h3 :id="`settings-strategy-family-${family.id}`">
            {{ family.label }}
          </h3>
          <div>
            <label
              v-for="plugin in family.strategies"
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
        </section>
      </div>

      <p class="settings-dialog-note">
        Draw-count changes apply immediately. Border-space and strategy changes
        reanalyze the active dataset using the new settings.
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
          @click="emit(
            'save',
            selectedIdsInPluginOrder,
            normalizedLastSeenDrawCount(),
            normalizedBorderSpace(),
            normalizedTargetGroupCount(),
          )"
        >
          {{ saving ? "Saving…" : "Save settings" }}
        </button>
      </footer>
    </section>
  </div>
</template>
