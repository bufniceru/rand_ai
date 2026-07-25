<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type {
  PredictionSuite,
  StrategyNumberPrediction,
  StrategyPrediction,
} from "../types";

const props = defineProps<{
  predictionSuites: PredictionSuite[];
}>();

const referenceOffset = ref(0);
const selectedStrategyId = ref("all");

const maximumOffset = computed(() => Math.max(0, props.predictionSuites.length - 1));
const selectedIndex = computed(() =>
  Math.max(0, props.predictionSuites.length - 1 - referenceOffset.value),
);
const selectedPrediction = computed(() => props.predictionSuites[selectedIndex.value] ?? null);
const selectedStrategy = computed(
  () =>
    selectedStrategyId.value === "all"
      ? null
      :
    selectedPrediction.value?.strategies.find(
      (strategy) => strategy.id === selectedStrategyId.value,
    ) ??
    selectedPrediction.value?.strategies[0] ??
    null,
);
const actualNumbers = computed(
  () => new Set(selectedPrediction.value?.actualNumbers ?? []),
);
const allReportCells = computed(() =>
  Array.from({ length: 49 }, (_value, index) => {
    const number = index + 1;
    return {
      number,
      reports: (selectedPrediction.value?.strategies ?? [])
        .filter((strategy) => strategy.topNumbers.includes(number))
        .map((strategy) => ({
          id: strategy.id,
          name: strategy.name,
          color: strategyColor(strategy.id),
        })),
    };
  }),
);
const coverageReport = computed(() => {
  const strategyCount = selectedPrediction.value?.strategies.length ?? 0;
  return Array.from({ length: strategyCount }, (_value, index) => {
    const threshold = index + 1;
    return {
      threshold,
      numberCount: allReportCells.value.filter(
        (cell) => cell.reports.length >= threshold,
      ).length,
    };
  });
});

watch(
  () => props.predictionSuites.length,
  () => {
    referenceOffset.value = Math.min(referenceOffset.value, maximumOffset.value);
  },
);

watch(selectedPrediction, (prediction) => {
  if (
    prediction &&
    selectedStrategyId.value !== "all" &&
    !prediction.strategies.some((strategy) => strategy.id === selectedStrategyId.value)
  ) {
    selectedStrategyId.value = prediction.strategies[0]?.id ?? "";
  }
});

function scoreLabel(entry: StrategyNumberPrediction): string {
  return `${(entry.score * 100).toFixed(2)}%`;
}

function strategyColor(strategyId: string): string {
  return {
    proximity: "#efb23e",
    freshness: "#f47f5b",
    emd: "#55a873",
    randomness: "#3264ad",
    entropy: "#c95d42",
    markov100: "#e8c238",
    bayesian: "#d477b8",
    svc: "#9567e8",
    tbl: "#1695a8",
  }[strategyId] ?? "#6e8195";
}

function cellTitle(
  entry: StrategyNumberPrediction,
  strategy: StrategyPrediction,
): string {
  const outcome = actualNumbers.value.has(entry.number) ? " — drawn next" : "";
  const details = entry.details.length > 0 ? ` — ${entry.details.join(" · ")}` : "";
  return `Number ${entry.number}: rank ${entry.rank}, ${strategy.name} score ${scoreLabel(entry)}${details}${outcome}`;
}

function allCellTitle(cell: (typeof allReportCells.value)[number]): string {
  const reports = cell.reports.map((report) => report.name).join(", ");
  const outcome = actualNumbers.value.has(cell.number) ? " — drawn next" : "";
  const consensus =
    cell.reports.length >= 3
      ? ` — high agreement (${cell.reports.length} strategies)`
      : "";
  return `Number ${cell.number}${outcome}${consensus} — predicted by ${reports || "no report"}`;
}

function sectorStyle(cell: (typeof allReportCells.value)[number]): Record<string, string> {
  if (cell.reports.length === 0) return {};
  if (cell.reports.length === 1) {
    return { background: cell.reports[0].color };
  }
  const sectorSize = 100 / cell.reports.length;
  const sectors = cell.reports.map(
    (report, index) =>
      `${report.color} ${(index * sectorSize).toFixed(3)}% ${((index + 1) * sectorSize).toFixed(3)}%`,
  );
  return { background: `conic-gradient(${sectors.join(", ")})` };
}
</script>

<template>
  <section
    v-if="selectedPrediction && selectedPrediction.strategies.length > 0"
    class="combined-prediction-view"
  >
    <header class="prediction-reference-toolbar">
      <div class="reference-buttons" aria-label="Prediction history navigation">
        <button
          type="button"
          :disabled="referenceOffset >= maximumOffset"
          @click="referenceOffset = maximumOffset"
        >
          First
        </button>
        <button
          type="button"
          :disabled="referenceOffset >= maximumOffset"
          @click="referenceOffset += 1"
        >
          Previous
        </button>
        <button
          type="button"
          :disabled="referenceOffset === 0"
          @click="referenceOffset -= 1"
        >
          Next
        </button>
        <button
          type="button"
          :disabled="referenceOffset === 0"
          @click="referenceOffset = 0"
        >
          Latest
        </button>
      </div>
      <div class="prediction-reference-summary">
        <span>Prediction after draw</span>
        <strong>{{ selectedPrediction.referenceDrawNumber }}</strong>
        <span aria-hidden="true">→</span>
        <span>draw {{ selectedPrediction.targetDrawNumber }}</span>
      </div>
    </header>

    <div class="prediction-workspace-layout">
      <div class="prediction-strategy-tabs" role="tablist" aria-label="Prediction strategy">
        <button
          type="button"
          role="tab"
          :aria-selected="selectedStrategyId === 'all'"
          :class="{ active: selectedStrategyId === 'all' }"
          @click="selectedStrategyId = 'all'"
        >
          All
        </button>
        <button
          v-for="strategy in selectedPrediction.strategies"
          :key="strategy.id"
          type="button"
          role="tab"
          :aria-selected="strategy.id === selectedStrategyId"
          :class="{ active: strategy.id === selectedStrategyId }"
          @click="selectedStrategyId = strategy.id"
        >
          {{ strategy.name }}
        </button>
      </div>

      <div class="prediction-workspace-content">
        <div v-if="selectedStrategy" class="prediction-strategy-heading">
          <div>
            <span>Prediction</span>
            <h1>{{ selectedStrategy.name }}</h1>
          </div>
          <p>{{ selectedStrategy.description }}</p>
          <strong>
            Top 6: {{ selectedStrategy.topNumbers.join(", ") }}
          </strong>
        </div>

        <div
          v-if="selectedStrategyId === 'all'"
          class="all-predictions-wrapper"
        >
          <section class="prediction-coverage-report" aria-label="Strategy coverage report">
            <header>
              <strong>Strategy coverage report</strong>
              <span>Cumulative coverage across all 49 numbers</span>
            </header>
            <ul>
              <li
                v-for="entry in coverageReport"
                :key="entry.threshold"
              >
                <span>
                  At least {{ entry.threshold }}
                  {{ entry.threshold === 1 ? "strategy" : "strategies" }}
                </span>
                <strong>{{ entry.numberCount }} numbers</strong>
              </li>
            </ul>
          </section>
          <div class="prediction-color-legend" aria-label="Prediction report colors">
            <span
              v-for="strategy in selectedPrediction.strategies"
              :key="strategy.id"
            >
              <i :style="{ background: strategyColor(strategy.id) }"></i>
              {{ strategy.name }}
            </span>
          </div>
          <div
          class="all-predictions-grid"
          role="grid"
          aria-label="All prediction reports by number from 1 through 49"
          >
            <article
              v-for="cell in allReportCells"
              :key="cell.number"
              class="all-predictions-cell"
              :class="{
                'has-prediction': cell.reports.length > 0,
                'is-high-consensus': cell.reports.length >= 3,
                'is-drawn': actualNumbers.has(cell.number),
              }"
              :style="sectorStyle(cell)"
              :title="allCellTitle(cell)"
              role="gridcell"
            >
              <strong>{{ cell.number }}</strong>
            </article>
          </div>
        </div>

        <div v-else-if="selectedStrategy" class="prediction-report-stack">
          <section class="prediction-report">
            <div
              class="combined-score-grid"
              role="grid"
              :aria-label="`${selectedStrategy.name} numbers by rank`"
            >
              <div
                v-for="entry in selectedStrategy.numbers"
                :key="entry.number"
                class="combined-score-cell"
                :class="{ 'is-drawn': actualNumbers.has(entry.number) }"
                :title="cellTitle(entry, selectedStrategy)"
                role="gridcell"
              >
                <span class="prediction-rank">#{{ entry.rank }}</span>
                <strong>{{ entry.number }}</strong>
                <small>{{ scoreLabel(entry) }}</small>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  </section>

  <section v-else class="dialog-empty-state">
    <strong>No prediction strategies are active</strong>
    <p>
      {{
        selectedPrediction
          ? "Enable at least one item in the Strategies menu."
          : "The imported dataset does not contain any draws."
      }}
    </p>
  </section>
</template>
