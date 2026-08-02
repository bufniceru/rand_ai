<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import EfficacyComparisonChart from "../components/EfficacyComparisonChart.vue";
import type { EfficacyChartRow } from "../lib/efficacyChart";
import {
  META_STRATEGY_DESCRIPTIONS,
  META_STRATEGY_IDS,
  META_STRATEGY_LABELS,
  aggregateMetaAccuracy,
  buildMetaFamilyEvidence,
  clampMetaRecordOffset,
  familyColor,
  familyModelRanks,
  isUniformColdStart,
  metaForecast,
  metaRecordAtOffset,
  rankedFamilyProbabilities,
  selectMetaAccuracyWindow,
  settledMetaRecordsThrough,
  type MetaAccuracyAnchor,
  type MetaStrategyId,
} from "../lib/metaStrategy";
import type {
  MetaDrawHistory,
  StrategyFamilyId,
  StrategyId,
  StrategyPlugin,
} from "../types";

const props = defineProps<{
  metaHistory: MetaDrawHistory;
  strategyPlugins: StrategyPlugin[];
}>();

const recordOffset = ref(0);
const activeMetaStrategyId = ref<MetaStrategyId>("family_ensemble");
const selectedFamilyId = ref<StrategyFamilyId | null>(null);
const accuracyCount = ref(0);
const accuracyAnchor = ref<MetaAccuracyAnchor>("latest");
const isEvidenceExpanded = ref(true);
const isOutcomeExpanded = ref(true);
const isAccuracyExpanded = ref(true);

const maximumOffset = computed(() => Math.max(0, props.metaHistory.records.length - 1));
const selectedRecord = computed(() =>
  metaRecordAtOffset(props.metaHistory.records, recordOffset.value),
);
const activeForecast = computed(() =>
  metaForecast(selectedRecord.value, activeMetaStrategyId.value),
);
const probabilities = computed(() =>
  rankedFamilyProbabilities(
    props.metaHistory,
    selectedRecord.value,
    activeMetaStrategyId.value,
  ),
);
const familyCatalog = computed(
  () => new Map(props.metaHistory.families.map((family) => [family.id, family])),
);
const strategyNames = computed(
  () => new Map(props.strategyPlugins.map((plugin) => [plugin.id, plugin.label])),
);
const evidenceRows = computed(() =>
  buildMetaFamilyEvidence(
    props.metaHistory,
    selectedRecord.value,
    activeMetaStrategyId.value,
  ),
);
const selectedEvidence = computed(
  () =>
    evidenceRows.value.find((row) => row.family.id === selectedFamilyId.value) ??
    null,
);
const selectedModelRanks = computed(() =>
  selectedFamilyId.value
    ? familyModelRanks(selectedRecord.value, selectedFamilyId.value)
    : [],
);
const winnerFamily = computed(() =>
  activeForecast.value
    ? familyCatalog.value.get(activeForecast.value.predictedFamilyId) ?? null
    : null,
);
const winnerProbability = computed(
  () => probabilities.value.find((probability) => probability.rank === 1) ?? null,
);
const hasPredictiveFamilies = computed(() => {
  const enabled = new Set(props.metaHistory.enabledStrategyIds);
  return props.metaHistory.families.some(
    (family) =>
      family.predictive && family.strategyIds.some((strategyId) => enabled.has(strategyId)),
  );
});
const coldStart = computed(() => isUniformColdStart(selectedRecord.value));
const familyEfficacyChartRows = computed<EfficacyChartRow[]>(() =>
  evidenceRows.value.map((row) => ({
    id: row.family.id,
    label: row.family.label,
    rate: row.snapshot.meanHitsPerStrategy,
    normalizedLift: row.snapshot.normalizedLift,
    detail: row.benchmark
      ? `${row.snapshot.evaluations} benchmark strategy-draw evaluations`
      : `${row.enabledStrategyIds.length} enabled strategies · ${row.snapshot.evaluations} strategy-draw evaluations`,
  })),
);
const outcomeRows = computed(() => {
  const catalogOrder = new Map(
    props.metaHistory.families.map((family, index) => [family.id, index]),
  );
  return [...(selectedRecord.value?.familyOutcomes ?? [])].sort((left, right) => {
    const leftFamily = familyCatalog.value.get(left.familyId);
    const rightFamily = familyCatalog.value.get(right.familyId);
    if (leftFamily?.predictive !== rightFamily?.predictive) {
      return leftFamily?.predictive ? -1 : 1;
    }
    return (
      left.rank - right.rank ||
      (catalogOrder.get(left.familyId) ?? Number.MAX_SAFE_INTEGER) -
        (catalogOrder.get(right.familyId) ?? Number.MAX_SAFE_INTEGER)
    );
  });
});
const prevailingFamilies = computed(() =>
  (selectedRecord.value?.prevailingFamilyIds ?? []).flatMap((familyId) => {
    const family = familyCatalog.value.get(familyId);
    return family ? [family] : [];
  }),
);
const availableAccuracyRecords = computed(() =>
  settledMetaRecordsThrough(props.metaHistory.records, selectedRecord.value),
);
const maximumAccuracyCount = computed(() => availableAccuracyRecords.value.length);
const appliedAccuracyCount = computed(() => {
  if (maximumAccuracyCount.value === 0) return 0;
  const requested = Number.isFinite(accuracyCount.value)
    ? Math.trunc(accuracyCount.value)
    : maximumAccuracyCount.value;
  return Math.min(Math.max(requested, 1), maximumAccuracyCount.value);
});
const accuracyWindow = computed(() =>
  selectMetaAccuracyWindow(
    props.metaHistory.records,
    selectedRecord.value,
    appliedAccuracyCount.value,
    accuracyAnchor.value,
  ),
);
const accuracySummaries = computed(() => aggregateMetaAccuracy(accuracyWindow.value));
const activeAccuracy = computed(
  () =>
    accuracySummaries.value.find(
      (summary) => summary.metaStrategyId === activeMetaStrategyId.value,
    ) ?? null,
);
const accuracyRangeLabel = computed(() => {
  if (accuracyWindow.value.length === 0) return "No settled forecasts available";
  return `Target draws ${accuracyWindow.value[0].targetDrawNumber}–${accuracyWindow.value.at(-1)?.targetDrawNumber}`;
});

watch(
  () => props.metaHistory.records.length,
  () => {
    recordOffset.value = clampMetaRecordOffset(
      props.metaHistory.records,
      recordOffset.value,
    );
  },
);

watch(
  [selectedRecord, activeMetaStrategyId],
  () => {
    selectedFamilyId.value =
      activeForecast.value?.predictedFamilyId ?? evidenceRows.value[0]?.family.id ?? null;
  },
  { immediate: true },
);

watch(
  maximumAccuracyCount,
  (maximum, previousMaximum) => {
    if (accuracyCount.value === 0 || accuracyCount.value === previousMaximum) {
      accuracyCount.value = maximum;
      return;
    }
    accuracyCount.value = Math.min(accuracyCount.value, maximum);
  },
  { immediate: true },
);

function selectMetaStrategy(metaStrategyId: MetaStrategyId): void {
  activeMetaStrategyId.value = metaStrategyId;
}

function selectMetaStrategyAndFocus(metaStrategyId: MetaStrategyId): void {
  selectMetaStrategy(metaStrategyId);
  nextTick(() => {
    document.getElementById(`meta-strategy-tab-${metaStrategyId}`)?.focus();
  });
}

function selectAdjacentMetaStrategy(offset: number): void {
  const currentIndex = META_STRATEGY_IDS.indexOf(activeMetaStrategyId.value);
  const nextIndex =
    (currentIndex + offset + META_STRATEGY_IDS.length) % META_STRATEGY_IDS.length;
  selectMetaStrategyAndFocus(META_STRATEGY_IDS[nextIndex]);
}

function normalizeAccuracyCount(): void {
  accuracyCount.value = appliedAccuracyCount.value;
}

function probabilityWidth(probability: number): string {
  return `${Math.max(0, Math.min(100, probability * 100))}%`;
}

function percentage(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function decimal(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

function signedDecimal(value: number): string {
  const formatted = value.toFixed(3);
  return value > 0 ? `+${formatted}` : formatted;
}

function strategyName(strategyId: StrategyId): string {
  return strategyNames.value.get(strategyId) ?? strategyId;
}

function familyName(familyId: StrategyFamilyId): string {
  return familyCatalog.value.get(familyId)?.label ?? familyId;
}

function memberHitsLabel(memberHits: Partial<Record<StrategyId, number>>): string {
  return Object.entries(memberHits)
    .map(([strategyId, hits]) => `${strategyName(strategyId as StrategyId)}: ${hits}`)
    .join(" · ");
}
</script>

<template>
  <section v-if="selectedRecord" class="meta-strategy-view">
    <Teleport to="#meta-strategy-toolbar-navigation">
      <div class="prediction-reference-toolbar">
        <div class="reference-buttons" aria-label="Meta Strategy history navigation">
          <button
            type="button"
            :disabled="recordOffset >= maximumOffset"
            aria-label="First MetaDraw forecast"
            title="First"
            @click="recordOffset = maximumOffset"
          >
            ⏮
          </button>
          <button
            type="button"
            :disabled="recordOffset >= maximumOffset"
            aria-label="Previous MetaDraw forecast"
            title="Previous"
            @click="recordOffset += 1"
          >
            ◀
          </button>
          <output
            class="prediction-reference-summary"
            :aria-label="`Reference draw ${selectedRecord.referenceDrawNumber}, target draw ${selectedRecord.targetDrawNumber}`"
          >
            <strong>{{ selectedRecord.referenceDrawNumber }}</strong>
            <span>→ {{ selectedRecord.targetDrawNumber }}</span>
          </output>
          <button
            type="button"
            :disabled="recordOffset === 0"
            aria-label="Next MetaDraw forecast"
            title="Next"
            @click="recordOffset -= 1"
          >
            ▶
          </button>
          <button
            type="button"
            :disabled="recordOffset === 0"
            aria-label="Latest MetaDraw forecast"
            title="Latest"
            @click="recordOffset = 0"
          >
            ⏭
          </button>
        </div>
      </div>
    </Teleport>

    <header class="meta-strategy-heading">
      <div>
        <p class="eyebrow">Family-level forecast</p>
        <h1>Meta Strategy</h1>
        <p>
          Reference draw {{ selectedRecord.referenceDrawNumber }} predicts target draw
          {{ selectedRecord.targetDrawNumber }}.
          <span v-if="selectedRecord.referenceDate">Reference date {{ selectedRecord.referenceDate }}.</span>
        </p>
      </div>
      <span class="meta-status" :class="selectedRecord.settled ? 'settled' : 'pending'">
        {{ selectedRecord.settled ? "Settled" : "Pending next draw" }}
      </span>
    </header>

    <nav
      class="meta-model-tabs"
      role="tablist"
      aria-label="Meta prediction strategy"
      @keydown.left.prevent="selectAdjacentMetaStrategy(-1)"
      @keydown.right.prevent="selectAdjacentMetaStrategy(1)"
      @keydown.home.prevent="selectMetaStrategyAndFocus(META_STRATEGY_IDS[0])"
      @keydown.end.prevent="selectMetaStrategyAndFocus(META_STRATEGY_IDS.at(-1)!)"
    >
      <button
        v-for="metaStrategyId in META_STRATEGY_IDS"
        :id="`meta-strategy-tab-${metaStrategyId}`"
        :key="metaStrategyId"
        type="button"
        role="tab"
        :aria-selected="activeMetaStrategyId === metaStrategyId"
        :tabindex="activeMetaStrategyId === metaStrategyId ? 0 : -1"
        :class="{ active: activeMetaStrategyId === metaStrategyId }"
        @click="selectMetaStrategy(metaStrategyId)"
      >
        <strong>{{ META_STRATEGY_LABELS[metaStrategyId] }}</strong>
        <small>{{ META_STRATEGY_DESCRIPTIONS[metaStrategyId] }}</small>
      </button>
    </nav>

    <p v-if="coldStart" class="meta-notice">
      No settled family evidence was available for this forecast, so predictive families
      begin with uniform confidence.
    </p>

    <section class="meta-forecast-layout" aria-label="Active family forecast">
      <article v-if="activeForecast && winnerFamily" class="meta-winner-card">
        <span>Predicted prevailing family</span>
        <strong :style="{ '--family-color': familyColor(winnerFamily.id) }">
          {{ winnerFamily.label }}
        </strong>
        <b>{{ percentage(winnerProbability?.probability) }}</b>
        <small>{{ META_STRATEGY_LABELS[activeMetaStrategyId] }}</small>
      </article>
      <article v-else class="meta-winner-card empty">
        <span>No predictive family forecast</span>
        <strong>
          {{ hasPredictiveFamilies ? "No model forecast is available" : "Enable a non-random strategy" }}
        </strong>
        <small>Random Baselines remain available below as a benchmark.</small>
      </article>

      <div class="meta-probability-panel">
        <header>
          <div>
            <h2>Ranked family confidence</h2>
            <p>Normalized model confidence across eligible predictive families.</p>
          </div>
          <small>Not a lottery-winning probability</small>
        </header>
        <ol v-if="probabilities.length > 0" class="meta-probability-bars">
          <li v-for="probability in probabilities" :key="probability.familyId">
            <button
              type="button"
              :class="{ selected: selectedFamilyId === probability.familyId }"
              :style="{ '--family-color': familyColor(probability.familyId) }"
              :aria-pressed="selectedFamilyId === probability.familyId"
              :aria-label="`Rank ${probability.rank}, ${familyName(probability.familyId)}, ${percentage(probability.probability)} confidence`"
              @click="selectedFamilyId = probability.familyId"
            >
              <span class="meta-probability-label">
                <b>#{{ probability.rank }}</b>
                {{ familyName(probability.familyId) }}
              </span>
              <span class="meta-probability-track" aria-hidden="true">
                <i :style="{ width: probabilityWidth(probability.probability) }"></i>
              </span>
              <strong>{{ percentage(probability.probability) }}</strong>
            </button>
          </li>
        </ol>
        <p v-else>No predictive family probabilities are available.</p>
      </div>
    </section>

    <section v-if="selectedEvidence" class="meta-family-detail">
      <header :style="{ '--family-color': selectedEvidence.color }">
        <div>
          <span>{{ selectedEvidence.benchmark ? "Benchmark family" : "Selected family" }}</span>
          <h2>{{ selectedEvidence.family.label }}</h2>
        </div>
        <b v-if="selectedEvidence.benchmark">Benchmark</b>
        <b v-else>
          #{{ selectedEvidence.probability?.rank ?? "—" }} ·
          {{ percentage(selectedEvidence.probability?.probability) }}
        </b>
      </header>
      <div class="meta-member-list">
        <span>Enabled member strategies</span>
        <ul>
          <li v-for="strategyId in selectedEvidence.enabledStrategyIds" :key="strategyId">
            {{ strategyName(strategyId) }}
          </li>
        </ul>
      </div>
      <dl class="meta-family-metrics">
        <div><dt>Evaluated draws</dt><dd>{{ selectedEvidence.snapshot.evaluatedDraws }}</dd></div>
        <div><dt>Evaluations</dt><dd>{{ selectedEvidence.snapshot.evaluations }}</dd></div>
        <div><dt>Historical mean</dt><dd>{{ decimal(selectedEvidence.snapshot.meanHitsPerStrategy) }}</dd></div>
        <div><dt>Recent EWMA</dt><dd>{{ decimal(selectedEvidence.snapshot.recentEwmaHitsPerStrategy) }}</dd></div>
        <div><dt>Lift vs random</dt><dd>{{ signedDecimal(selectedEvidence.snapshot.normalizedLift) }}</dd></div>
        <div><dt>Win share</dt><dd>{{ percentage(selectedEvidence.snapshot.winShare) }}</dd></div>
        <div><dt>Volatility</dt><dd>{{ decimal(selectedEvidence.snapshot.volatility) }}</dd></div>
        <div>
          <dt>Draws since win</dt>
          <dd>{{ selectedEvidence.snapshot.drawsSinceWin ?? "Never" }}</dd>
        </div>
      </dl>
      <div class="meta-model-ranks">
        <article v-for="rank in selectedModelRanks" :key="rank.metaStrategyId">
          <span>{{ rank.label }}</span>
          <strong>{{ rank.rank === null ? "Benchmark" : `#${rank.rank}` }}</strong>
          <b>{{ percentage(rank.probability) }}</b>
        </article>
      </div>
      <div
        v-if="selectedRecord.settled && selectedEvidence.outcome"
        class="meta-selected-outcome"
        :class="{ prevailing: selectedEvidence.outcome.prevailing }"
      >
        <span>Selected family target outcome</span>
        <strong>
          {{
            selectedEvidence.outcome.rank === 0
              ? "Benchmark"
              : `Rank #${selectedEvidence.outcome.rank}`
          }}
          ·
          {{ selectedEvidence.outcome.prevailing ? "Prevailing family" : "Did not prevail" }}
        </strong>
        <small>
          {{ selectedEvidence.outcome.totalHits }} total hits ·
          {{ decimal(selectedEvidence.outcome.meanHitsPerStrategy) }} mean ·
          {{ memberHitsLabel(selectedEvidence.outcome.memberHits) }}
        </small>
      </div>
    </section>

    <section class="meta-collapsible" :class="{ collapsed: !isEvidenceExpanded }">
      <header>
        <div>
          <strong>Historical family efficiency at this reference</strong>
          <small>Evidence available before target draw {{ selectedRecord.targetDrawNumber }}</small>
        </div>
        <button
          type="button"
          :aria-expanded="isEvidenceExpanded"
          aria-controls="meta-family-evidence"
          @click="isEvidenceExpanded = !isEvidenceExpanded"
        >
          {{ isEvidenceExpanded ? "Collapse" : "Expand" }}
        </button>
      </header>
      <div v-show="isEvidenceExpanded" id="meta-family-evidence">
        <div class="meta-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Family</th><th>Strategies</th><th>Draws</th><th>Evaluations</th>
                <th>Mean hits</th><th>Recent EWMA</th><th>Lift</th><th>Win share</th>
                <th>Volatility</th><th>Since win</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in evidenceRows"
                :key="row.family.id"
                :class="{ benchmark: row.benchmark, selected: row.family.id === selectedFamilyId }"
              >
                <th scope="row" :style="{ '--family-color': row.color }">
                  <button type="button" @click="selectedFamilyId = row.family.id">
                    <span v-if="row.probability">#{{ row.probability.rank }}</span>
                    {{ row.family.label }}
                    <b v-if="row.benchmark">Benchmark</b>
                  </button>
                </th>
                <td>{{ row.enabledStrategyIds.length }}</td>
                <td>{{ row.snapshot.evaluatedDraws }}</td>
                <td>{{ row.snapshot.evaluations }}</td>
                <td>{{ decimal(row.snapshot.meanHitsPerStrategy) }}</td>
                <td>{{ decimal(row.snapshot.recentEwmaHitsPerStrategy) }}</td>
                <td>{{ signedDecimal(row.snapshot.normalizedLift) }}</td>
                <td>{{ percentage(row.snapshot.winShare) }}</td>
                <td>{{ decimal(row.snapshot.volatility) }}</td>
                <td>{{ row.snapshot.drawsSinceWin ?? "Never" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <EfficacyComparisonChart
          id="meta-family-efficiency-rate"
          title="Family historical mean hits per strategy-draw"
          :rows="familyEfficacyChartRows"
          mode="rate"
          rate-unit="strategy-draw"
        />
        <EfficacyComparisonChart
          id="meta-family-efficiency-lift"
          title="Family normalized lift from random"
          :rows="familyEfficacyChartRows"
          mode="lift"
          rate-unit="strategy-draw"
        />
      </div>
    </section>

    <section class="meta-collapsible" :class="{ collapsed: !isOutcomeExpanded }">
      <header>
        <div>
          <strong>Target draw outcome</strong>
          <small>Target draw {{ selectedRecord.targetDrawNumber }}</small>
        </div>
        <button
          type="button"
          :aria-expanded="isOutcomeExpanded"
          aria-controls="meta-target-outcome"
          @click="isOutcomeExpanded = !isOutcomeExpanded"
        >
          {{ isOutcomeExpanded ? "Collapse" : "Expand" }}
        </button>
      </header>
      <div v-show="isOutcomeExpanded" id="meta-target-outcome">
        <div v-if="selectedRecord.settled" class="meta-outcome-content">
          <div class="meta-prevailing">
            <span>Prevailing {{ prevailingFamilies.length === 1 ? "family" : "co-winners" }}</span>
            <strong
              v-for="family in prevailingFamilies"
              :key="family.id"
              :style="{ '--family-color': familyColor(family.id) }"
            >
              {{ family.label }}
            </strong>
          </div>
          <div class="meta-table-wrap">
            <table>
              <thead>
                <tr><th>Family</th><th>Rank</th><th>Strategies</th><th>Total hits</th><th>Mean hits</th><th>Member hits</th></tr>
              </thead>
              <tbody>
                <tr v-for="outcome in outcomeRows" :key="outcome.familyId" :class="{ prevailing: outcome.prevailing }">
                  <th scope="row">{{ familyName(outcome.familyId) }}</th>
                  <td>{{ outcome.rank === 0 ? "Benchmark" : `#${outcome.rank}` }}</td>
                  <td>{{ outcome.strategyCount }}</td>
                  <td>{{ outcome.totalHits }}</td>
                  <td>{{ decimal(outcome.meanHitsPerStrategy) }}</td>
                  <td :title="memberHitsLabel(outcome.memberHits)">{{ memberHitsLabel(outcome.memberHits) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <p v-else class="meta-pending-outcome">
          Awaiting target draw {{ selectedRecord.targetDrawNumber }}. The forecast is fixed and will be settled when that real draw is added.
        </p>
      </div>
    </section>

    <section class="meta-collapsible" :class="{ collapsed: !isAccuracyExpanded }">
      <header>
        <div>
          <strong>Historical meta-strategy accuracy</strong>
          <small>Only settled forecasts through reference draw {{ selectedRecord.referenceDrawNumber }}</small>
        </div>
        <button
          type="button"
          :aria-expanded="isAccuracyExpanded"
          aria-controls="meta-accuracy"
          @click="isAccuracyExpanded = !isAccuracyExpanded"
        >
          {{ isAccuracyExpanded ? "Collapse" : "Expand" }}
        </button>
      </header>
      <div v-show="isAccuracyExpanded" id="meta-accuracy">
        <div class="meta-accuracy-controls">
          <label>
            <span>Settled forecasts to compare</span>
            <input
              v-model.number="accuracyCount"
              type="number"
              min="1"
              :max="maximumAccuracyCount"
              :disabled="maximumAccuracyCount === 0"
              @change="normalizeAccuracyCount"
            />
            <small>Maximum {{ maximumAccuracyCount }}</small>
          </label>
          <fieldset>
            <legend>Take records from</legend>
            <label><input v-model="accuracyAnchor" type="radio" value="first" /> First N</label>
            <label><input v-model="accuracyAnchor" type="radio" value="latest" /> Latest N</label>
          </fieldset>
          <output><strong>{{ appliedAccuracyCount }} forecasts</strong><span>{{ accuracyRangeLabel }}</span></output>
        </div>
        <article v-if="activeAccuracy && activeAccuracy.evaluations > 0" class="meta-accuracy-lead">
          <span>{{ activeAccuracy.label }} top-family hit rate</span>
          <strong>{{ percentage(activeAccuracy.topPredictionHitRate) }}</strong>
          <small>{{ activeAccuracy.topPredictionHits }} of {{ activeAccuracy.evaluations }} settled forecasts</small>
        </article>
        <p v-else class="meta-pending-outcome">No settled forecast evaluations are available in this range.</p>
        <div class="meta-table-wrap">
          <table>
            <thead>
              <tr><th>Meta strategy</th><th>Evaluations</th><th>Top hits</th><th>Hit rate</th><th>Winner probability mass</th><th>Reciprocal winner rank</th><th>Brier score</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in accuracySummaries" :key="summary.metaStrategyId" :class="{ selected: summary.metaStrategyId === activeMetaStrategyId }">
                <th scope="row">{{ summary.label }}</th>
                <td>{{ summary.evaluations }}</td>
                <td>{{ summary.topPredictionHits }}</td>
                <td>{{ percentage(summary.topPredictionHitRate) }}</td>
                <td>{{ percentage(summary.meanWinningProbabilityMass) }}</td>
                <td>{{ decimal(summary.meanReciprocalWinnerRank) }}</td>
                <td>{{ decimal(summary.meanBrierScore) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </section>

  <section v-else class="meta-strategy-empty">
    <strong>Meta Strategy analysis is unavailable</strong>
    <p>
      Enable the Meta Strategy report and at least one predictive strategy, then reanalyze the dataset.
    </p>
  </section>
</template>

<style scoped>
.meta-strategy-view,
.meta-strategy-empty {
  width: 100%;
  min-height: 100%;
  color: var(--monokai-fg);
  background: var(--monokai-bg);
}

.meta-strategy-view {
  display: grid;
  align-content: start;
  gap: 10px;
  padding: 12px;
}

.meta-strategy-heading,
.meta-model-tabs,
.meta-forecast-layout,
.meta-family-detail,
.meta-collapsible,
.meta-strategy-empty {
  border: 1px solid var(--monokai-border);
  background: var(--monokai-surface);
}

.meta-strategy-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  padding: 14px 16px;
}

.meta-strategy-heading h1,
.meta-strategy-heading p,
.meta-probability-panel h2,
.meta-probability-panel p,
.meta-family-detail h2 {
  margin: 0;
}

.meta-strategy-heading h1 {
  font-size: 24px;
}

.meta-strategy-heading > div > p:last-child {
  margin-top: 5px;
  color: var(--monokai-muted);
  font-size: 12px;
}

.meta-status {
  padding: 6px 10px;
  border: 1px solid currentColor;
  font-size: 11px;
  font-weight: 900;
  text-transform: uppercase;
}

.meta-status.pending {
  color: var(--monokai-yellow);
}

.meta-status.settled {
  color: var(--monokai-green);
}

.meta-model-tabs {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px;
  padding: 5px;
}

.meta-model-tabs button {
  min-width: 0;
  min-height: 58px;
  display: grid;
  align-content: center;
  gap: 3px;
  padding: 7px 9px;
  border: 1px solid var(--monokai-border);
  color: var(--monokai-fg);
  background: var(--monokai-raised);
  text-align: left;
  cursor: pointer;
}

.meta-model-tabs button.active {
  border-color: var(--monokai-yellow);
  background: color-mix(in srgb, var(--monokai-yellow) 14%, var(--monokai-surface));
}

.meta-model-tabs strong {
  font-size: 12px;
}

.meta-model-tabs small {
  overflow: hidden;
  color: var(--monokai-muted);
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta-notice,
.meta-pending-outcome {
  margin: 0;
  padding: 10px 12px;
  border: 1px solid var(--monokai-yellow);
  color: var(--monokai-yellow);
  background: color-mix(in srgb, var(--monokai-yellow) 9%, var(--monokai-surface));
  font-size: 11px;
  font-weight: 750;
}

.meta-forecast-layout {
  display: grid;
  grid-template-columns: minmax(210px, 0.7fr) minmax(420px, 2fr);
  gap: 12px;
  padding: 12px;
}

.meta-winner-card {
  min-height: 170px;
  display: grid;
  align-content: center;
  gap: 7px;
  padding: 18px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-deep);
}

.meta-winner-card > span,
.meta-winner-card small,
.meta-probability-panel header p,
.meta-probability-panel header small {
  color: var(--monokai-muted);
  font-size: 10px;
  font-weight: 800;
}

.meta-winner-card > strong {
  padding-left: 10px;
  border-left: 5px solid var(--family-color);
  font-size: 19px;
}

.meta-winner-card > b {
  color: var(--monokai-yellow);
  font-size: 34px;
  font-variant-numeric: tabular-nums;
}

.meta-winner-card.empty > strong {
  border-left-color: var(--monokai-muted);
}

.meta-probability-panel {
  min-width: 0;
}

.meta-probability-panel > header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 9px;
}

.meta-probability-panel h2 {
  font-size: 15px;
}

.meta-probability-bars {
  display: grid;
  gap: 5px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.meta-probability-bars button {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(160px, 0.8fr) minmax(180px, 2fr) 58px;
  align-items: center;
  gap: 9px;
  min-height: 34px;
  padding: 5px 7px;
  border: 1px solid transparent;
  color: var(--monokai-fg);
  background: transparent;
  cursor: pointer;
}

.meta-probability-bars button:hover,
.meta-probability-bars button.selected {
  border-color: var(--family-color);
  background: color-mix(in srgb, var(--family-color) 10%, transparent);
}

.meta-probability-label {
  overflow: hidden;
  font-size: 11px;
  font-weight: 850;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta-probability-label b {
  display: inline-block;
  width: 28px;
  color: var(--family-color);
}

.meta-probability-track {
  height: 17px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-deep);
}

.meta-probability-track i {
  display: block;
  height: 100%;
  background: var(--family-color);
}

.meta-probability-bars button > strong {
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.meta-family-detail {
  display: grid;
  grid-template-columns: minmax(210px, 0.8fr) minmax(410px, 1.4fr);
  gap: 10px 14px;
  padding: 12px;
}

.meta-family-detail > header {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 3px 0 7px 10px;
  border-bottom: 1px solid var(--monokai-border);
  border-left: 5px solid var(--family-color);
}

.meta-family-detail header span,
.meta-member-list > span,
.meta-family-metrics dt,
.meta-model-ranks span {
  color: var(--monokai-muted);
  font-size: 9px;
  font-weight: 850;
  text-transform: uppercase;
}

.meta-family-detail header h2 {
  margin-top: 2px;
  font-size: 17px;
}

.meta-family-detail header > b {
  color: var(--family-color);
  font-size: 12px;
}

.meta-member-list ul {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin: 7px 0 0;
  padding: 0;
  list-style: none;
}

.meta-member-list li {
  padding: 4px 6px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-raised);
  font-size: 10px;
}

.meta-family-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(90px, 1fr));
  gap: 5px;
  margin: 0;
}

.meta-family-metrics > div {
  padding: 6px 7px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-deep);
}

.meta-family-metrics dt,
.meta-family-metrics dd {
  margin: 0;
}

.meta-family-metrics dd {
  margin-top: 3px;
  font-size: 13px;
  font-weight: 900;
  font-variant-numeric: tabular-nums;
}

.meta-model-ranks {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 5px;
}

.meta-selected-outcome {
  grid-column: 1 / -1;
  display: grid;
  gap: 3px;
  padding: 8px 10px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-deep);
}

.meta-selected-outcome.prevailing {
  border-color: var(--monokai-green);
  background: color-mix(in srgb, var(--monokai-green) 8%, var(--monokai-surface));
}

.meta-selected-outcome span,
.meta-selected-outcome small {
  color: var(--monokai-muted);
  font-size: 9px;
  font-weight: 800;
}

.meta-selected-outcome strong {
  font-size: 11px;
}

.meta-model-ranks article {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px 8px;
  padding: 7px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-raised);
}

.meta-model-ranks span {
  grid-column: 1 / -1;
}

.meta-model-ranks strong,
.meta-model-ranks b {
  font-size: 11px;
}

.meta-model-ranks b {
  color: var(--monokai-yellow);
}

.meta-collapsible {
  padding: 10px 12px 12px;
}

.meta-collapsible.collapsed {
  padding-bottom: 10px;
}

.meta-collapsible > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 31px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--monokai-border);
}

.meta-collapsible.collapsed > header {
  padding-bottom: 0;
  border-bottom: 0;
}

.meta-collapsible > header > div {
  display: grid;
  gap: 2px;
}

.meta-collapsible > header strong {
  font-size: 14px;
}

.meta-collapsible > header small {
  color: var(--monokai-muted);
  font-size: 10px;
}

.meta-collapsible > header button {
  padding: 5px 9px;
  border: 1px solid var(--monokai-border);
  color: var(--monokai-fg);
  background: var(--monokai-raised);
  font-size: 10px;
  font-weight: 850;
  cursor: pointer;
}

.meta-table-wrap {
  width: 100%;
  overflow: auto;
  margin-top: 9px;
  border: 1px solid var(--monokai-border);
}

.meta-table-wrap table {
  width: 100%;
  min-width: 920px;
  border-collapse: collapse;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.meta-table-wrap th,
.meta-table-wrap td {
  padding: 6px 7px;
  border: 1px solid var(--monokai-border);
  text-align: right;
  white-space: nowrap;
}

.meta-table-wrap th:first-child,
.meta-table-wrap td:first-child,
.meta-table-wrap td:last-child {
  text-align: left;
}

.meta-table-wrap thead th {
  color: var(--monokai-yellow);
  background: var(--monokai-deep);
}

.meta-table-wrap tbody tr:nth-child(even) {
  background: var(--monokai-raised);
}

.meta-table-wrap tbody tr.selected,
.meta-table-wrap tbody tr.prevailing {
  background: color-mix(in srgb, var(--monokai-green) 12%, var(--monokai-surface));
}

.meta-table-wrap tbody tr.benchmark {
  background: color-mix(in srgb, var(--monokai-yellow) 8%, var(--monokai-surface));
}

.meta-table-wrap tbody th {
  border-left: 4px solid var(--family-color, var(--monokai-border));
}

.meta-table-wrap tbody th button {
  padding: 0;
  border: 0;
  color: var(--monokai-fg);
  background: transparent;
  font: inherit;
  font-weight: 800;
  cursor: pointer;
}

.meta-table-wrap tbody th button:hover {
  color: var(--family-color);
}

.meta-table-wrap tbody th span {
  color: var(--family-color);
  margin-right: 4px;
}

.meta-table-wrap tbody th b {
  margin-left: 5px;
  color: var(--monokai-yellow);
  font-size: 8px;
  text-transform: uppercase;
}

.meta-prevailing {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 9px;
}

.meta-prevailing > span {
  color: var(--monokai-muted);
  font-size: 10px;
  font-weight: 850;
  text-transform: uppercase;
}

.meta-prevailing > strong {
  padding: 5px 8px;
  border: 1px solid var(--family-color);
  color: var(--family-color);
  font-size: 10px;
}

.meta-pending-outcome {
  margin-top: 9px;
}

.meta-accuracy-controls {
  display: grid;
  grid-template-columns: minmax(180px, 0.8fr) minmax(220px, 1fr) minmax(190px, 1fr);
  gap: 7px;
  margin-top: 9px;
}

.meta-accuracy-controls > label,
.meta-accuracy-controls fieldset,
.meta-accuracy-controls output {
  min-width: 0;
  margin: 0;
  padding: 7px 9px;
  border: 1px solid var(--monokai-border);
  background: var(--monokai-deep);
}

.meta-accuracy-controls > label {
  display: grid;
  grid-template-columns: 1fr 70px;
  gap: 3px 7px;
}

.meta-accuracy-controls > label span,
.meta-accuracy-controls legend,
.meta-accuracy-controls output span,
.meta-accuracy-controls small {
  color: var(--monokai-muted);
  font-size: 9px;
  font-weight: 800;
}

.meta-accuracy-controls input[type="number"] {
  grid-row: 1 / span 2;
  grid-column: 2;
  width: 70px;
  padding: 3px 5px;
  border: 1px solid var(--monokai-border);
  color: var(--monokai-fg);
  background: var(--monokai-surface);
}

.meta-accuracy-controls fieldset {
  display: flex;
  align-items: center;
  gap: 10px;
}

.meta-accuracy-controls fieldset label {
  color: var(--monokai-fg);
  font-size: 10px;
}

.meta-accuracy-controls output {
  display: grid;
  gap: 2px;
}

.meta-accuracy-lead {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 3px 12px;
  margin-top: 9px;
  padding: 10px 12px;
  border: 1px solid var(--monokai-green);
  background: color-mix(in srgb, var(--monokai-green) 9%, var(--monokai-surface));
}

.meta-accuracy-lead span,
.meta-accuracy-lead small {
  color: var(--monokai-muted);
  font-size: 10px;
  font-weight: 800;
}

.meta-accuracy-lead strong {
  grid-row: 1 / span 2;
  grid-column: 2;
  color: var(--monokai-green);
  font-size: 27px;
}

.meta-strategy-empty {
  display: grid;
  place-content: center;
  gap: 6px;
  padding: 40px;
  text-align: center;
}

.meta-strategy-empty p {
  max-width: 520px;
  margin: 0;
  color: var(--monokai-muted);
  font-size: 12px;
}

@media (max-width: 980px) {
  .meta-model-tabs,
  .meta-model-ranks,
  .meta-family-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .meta-forecast-layout,
  .meta-family-detail {
    grid-template-columns: 1fr;
  }

  .meta-family-detail > header,
  .meta-model-ranks,
  .meta-selected-outcome {
    grid-column: 1;
  }

  .meta-accuracy-controls {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 620px) {
  .meta-strategy-view {
    padding: 7px;
  }

  .meta-strategy-heading {
    flex-direction: column;
  }

  .meta-model-tabs,
  .meta-model-ranks,
  .meta-family-metrics {
    grid-template-columns: 1fr;
  }

  .meta-probability-bars button {
    grid-template-columns: minmax(120px, 1fr) 50px;
  }

  .meta-probability-track {
    grid-column: 1 / -1;
    grid-row: 2;
  }
}
</style>
