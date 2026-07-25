<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import PredictionWorkspaceNavigation from "../components/PredictionWorkspaceNavigation.vue";
import {
  RANDOM_BENCHMARK_SIMULATIONS,
  randomTailProbability,
  runRandomBenchmark,
  type RandomBenchmarkSummary,
} from "../lib/randomBenchmark";
import type {
  PossibleDrawNumberState,
  PredictionSuite,
  StrategyEfficacy,
  StrategyEfficacyRecord,
  StrategyNumberPrediction,
  StrategyPrediction,
} from "../types";

const props = defineProps<{
  predictionSuites: PredictionSuite[];
  efficacyHistory: StrategyEfficacyRecord[];
}>();

const referenceOffset = ref(0);
const selectedStrategyId = ref("all");
const selectedCoverageThreshold = ref<number | null>(null);
const mutedStrategyIds = ref<Set<string>>(new Set());
const efficacyDrawCount = ref(0);
const efficacyRangeAnchor = ref<"first" | "latest">("first");
const efficacyRandomBenchmark = shallowRef<RandomBenchmarkSummary | null>(null);
const efficacyRetestCount = ref(0);
const isEfficacyRetesting = ref(false);
let numberActionTimer: ReturnType<typeof setTimeout> | null = null;

const maximumOffset = computed(() => Math.max(0, props.predictionSuites.length - 1));
const selectedIndex = computed(() =>
  Math.max(0, props.predictionSuites.length - 1 - referenceOffset.value),
);
const selectedPrediction = computed(() => props.predictionSuites[selectedIndex.value] ?? null);
const selectedStrategy = computed(
  () =>
    selectedStrategyId.value === "all"
      ? null
      : selectedPrediction.value?.strategies.find(
          (strategy) => strategy.id === selectedStrategyId.value,
        ) ??
        selectedPrediction.value?.strategies[0] ??
        null,
);
const allStrategiesVisible = computed(
  () =>
    (selectedPrediction.value?.strategies.length ?? 0) > 0 &&
    selectedPrediction.value?.strategies.every(
      (strategy) => !mutedStrategyIds.value.has(strategy.id),
    ),
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
          color: strategyDisplayColor(strategy.id),
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
const availableEfficacyHistory = computed(() => {
  const maximumReference =
    selectedPrediction.value?.referenceDrawNumber ?? Number.MAX_SAFE_INTEGER;
  return props.efficacyHistory.filter(
    (record) => record.referenceDrawNumber <= maximumReference,
  );
});
const maximumEfficacyDraws = computed(() => availableEfficacyHistory.value.length);
const appliedEfficacyDrawCount = computed(() => {
  if (maximumEfficacyDraws.value === 0) return 0;
  const requested = Number.isFinite(efficacyDrawCount.value)
    ? Math.trunc(efficacyDrawCount.value)
    : maximumEfficacyDraws.value;
  return Math.min(Math.max(requested, 1), maximumEfficacyDraws.value);
});
const selectedEfficacyHistory = computed(() => {
  const count = appliedEfficacyDrawCount.value;
  if (count === 0) return [];
  return efficacyRangeAnchor.value === "first"
    ? availableEfficacyHistory.value.slice(0, count)
    : availableEfficacyHistory.value.slice(-count);
});
const efficacySelectionKey = computed(() =>
  selectedEfficacyHistory.value
    .map((record) => record.targetDrawNumber)
    .join(","),
);
const efficacyByStrategy = computed(() => {
  const records = selectedEfficacyHistory.value;
  const evaluatedDraws = records.length;
  const expectedRandomHits = evaluatedDraws * 36 / 49;
  const randomHits =
    efficacyRandomBenchmark.value?.meanHits ?? expectedRandomHits;
  return new Map(
    (selectedPrediction.value?.strategies ?? []).map((strategy) => {
      const strategyHits =
        strategy.id === "randomness"
          ? randomHits
          : records.reduce(
              (total, record) => total + (record.strategyHits[strategy.id] ?? 0),
              0,
            );
      const efficacy: StrategyEfficacy = {
        evaluatedDraws,
        strategyHits,
        randomHits,
        expectedRandomHits,
        averageHitsPerDraw:
          evaluatedDraws > 0 ? strategyHits / evaluatedDraws : 0,
        randomAverageHitsPerDraw:
          evaluatedDraws > 0 ? randomHits / evaluatedDraws : 0,
        hitDifference: strategyHits - randomHits,
      };
      return [strategy.id, efficacy] as const;
    }),
  );
});
const selectedStrategyEfficacy = computed(() =>
  selectedStrategy.value
    ? efficacyByStrategy.value.get(selectedStrategy.value.id) ?? null
    : null,
);
const efficacyRangeLabel = computed(() => {
  const records = selectedEfficacyHistory.value;
  if (records.length === 0) return "No completed draws available";
  return `Target draws ${records[0].targetDrawNumber}–${records.at(-1)?.targetDrawNumber}`;
});
const efficacyRandomDatasetLabel = computed(() =>
  efficacyRandomBenchmark.value
    ? `${RANDOM_BENCHMARK_SIMULATIONS.toLocaleString()} datasets · run #${efficacyRetestCount.value}`
    : "Exact mean · run for 95% interval",
);
const efficacyRandomRangeLabel = computed(() => {
  const benchmark = efficacyRandomBenchmark.value;
  return benchmark
    ? `${benchmark.lower95Hits}–${benchmark.upper95Hits}`
    : "Run Retest";
});
const efficacyRanking = computed(() =>
  [...(selectedPrediction.value?.strategies ?? [])].sort((left, right) => {
    const leftEfficacy = efficacyByStrategy.value.get(left.id);
    const rightEfficacy = efficacyByStrategy.value.get(right.id);
    const difference =
      (rightEfficacy?.hitDifference ?? -Number.MAX_SAFE_INTEGER) -
      (leftEfficacy?.hitDifference ?? -Number.MAX_SAFE_INTEGER);
    if (difference !== 0) return difference;
    const hitDifference =
      (rightEfficacy?.strategyHits ?? 0) - (leftEfficacy?.strategyHits ?? 0);
    if (hitDifference !== 0) return hitDifference;
    return left.name.localeCompare(right.name);
  }),
);

watch(
  () => props.predictionSuites.length,
  () => {
    referenceOffset.value = Math.min(referenceOffset.value, maximumOffset.value);
  },
);

watch(
  maximumEfficacyDraws,
  (maximum, previousMaximum) => {
    if (
      efficacyDrawCount.value === 0 ||
      efficacyDrawCount.value === previousMaximum
    ) {
      efficacyDrawCount.value = maximum;
      return;
    }
    efficacyDrawCount.value = Math.min(efficacyDrawCount.value, maximum);
  },
  { immediate: true },
);

watch(efficacySelectionKey, () => {
  efficacyRandomBenchmark.value = null;
  efficacyRetestCount.value = 0;
  isEfficacyRetesting.value = false;
});

watch(selectedPrediction, (prediction) => {
  const available = new Set<string>(
    prediction?.strategies.map((strategy) => strategy.id) ?? [],
  );
  if (
    selectedCoverageThreshold.value !== null &&
    selectedCoverageThreshold.value > available.size
  ) {
    selectedCoverageThreshold.value = null;
  }
  if (
    prediction &&
    selectedStrategyId.value !== "all" &&
    !available.has(selectedStrategyId.value)
  ) {
    selectedStrategyId.value = prediction.strategies[0]?.id ?? "all";
  }
  mutedStrategyIds.value = new Set(
    [...mutedStrategyIds.value].filter((strategyId) => available.has(strategyId)),
  );
});

function scoreLabel(entry: StrategyNumberPrediction): string {
  return `${(entry.score * 100).toFixed(2)}%`;
}

function efficacyFor(strategy: StrategyPrediction): StrategyEfficacy | null {
  return efficacyByStrategy.value.get(strategy.id) ?? null;
}

function efficacyDifferenceFor(strategy: StrategyPrediction): string {
  const efficacy = efficacyFor(strategy);
  return efficacy ? hitDifferenceLabel(efficacy) : "Unavailable";
}

function efficacyVerdictFor(strategy: StrategyPrediction): string {
  const efficacy = efficacyFor(strategy);
  return efficacy ? efficacyVerdict(efficacy) : "Unavailable";
}

function normalizeEfficacyDrawCount(): void {
  efficacyDrawCount.value = appliedEfficacyDrawCount.value;
}

function formatHitTotal(value: number): string {
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

function randomTailLabel(strategy: StrategyPrediction): string {
  const benchmark = efficacyRandomBenchmark.value;
  const efficacy = efficacyFor(strategy);
  if (!benchmark || !efficacy) return "Run Retest";
  const probability = randomTailProbability(benchmark, efficacy.strategyHits);
  if (probability < 0.001) return "<0.1%";
  if (probability < 0.1) return `${(probability * 100).toFixed(1)}%`;
  return `${Math.round(probability * 100)}%`;
}

async function retestRandomDataset(): Promise<void> {
  const drawCount = selectedEfficacyHistory.value.length;
  if (drawCount === 0 || isEfficacyRetesting.value) return;
  isEfficacyRetesting.value = true;
  await nextTick();
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  try {
    efficacyRandomBenchmark.value = runRandomBenchmark(drawCount);
    efficacyRetestCount.value += 1;
  } finally {
    isEfficacyRetesting.value = false;
  }
}

function hitDifferenceLabel(efficacy: StrategyEfficacy): string {
  if (efficacy.evaluatedDraws === 0) return "Awaiting results";
  if (efficacy.hitDifference > 0) {
    return `+${formatHitTotal(efficacy.hitDifference)} hits`;
  }
  if (efficacy.hitDifference < 0) {
    return `${formatHitTotal(efficacy.hitDifference)} hits`;
  }
  return "Tied";
}

function efficacyVerdict(efficacy: StrategyEfficacy): string {
  if (efficacy.evaluatedDraws === 0) return "No completed predictions yet";
  if (efficacy.hitDifference > 0) return "Beats random mean";
  if (efficacy.hitDifference < 0) return "Trails random mean";
  return "Tied with random mean";
}

function efficacyClass(efficacy: StrategyEfficacy | null): string {
  if (!efficacy || efficacy.evaluatedDraws === 0) return "is-pending";
  if (efficacy.hitDifference > 0) return "is-ahead";
  if (efficacy.hitDifference < 0) return "is-behind";
  return "is-tied";
}

function strategyFullName(strategy: StrategyPrediction): string {
  return {
    proximity: "Proximity",
    freshness: "Freshness",
    emd: "Earth Mover Distance",
    randomness: "Random baseline",
    fresh_random: "Fresh Random",
    chi_square: "Chi-square Frequency",
    entropy: "Entropy",
    markov100: "Markov 100",
    mkfr: "Markov Freshness",
    bayesian: "Bayesian",
    predictive_grid: "Predictive Score Grid",
    mixed: "Mixed Prediction",
    svc: "Support Vector Classifier",
    tbl: "Temporal Behavior Learning",
    cis: "Collective Intelligence Strategy",
  }[strategy.id] ?? strategy.name;
}

function strategyColor(strategyId: string): string {
  return {
    proximity: "#efb23e",
    freshness: "#f47f5b",
    emd: "#55a873",
    randomness: "#3264ad",
    fresh_random: "#7b56c2",
    chi_square: "#6256c7",
    entropy: "#c95d42",
    markov100: "#e8c238",
    mkfr: "#1f8f75",
    bayesian: "#d477b8",
    predictive_grid: "#008ca8",
    mixed: "#dd6b20",
    svc: "#9567e8",
    tbl: "#1695a8",
    cis: "#b12f67",
  }[strategyId] ?? "#6e8195";
}

function strategyDisplayColor(strategyId: string): string {
  return mutedStrategyIds.value.has(strategyId)
    ? "#ffffff"
    : strategyColor(strategyId);
}

function toggleStrategyColor(strategyId: string): void {
  const next = new Set(mutedStrategyIds.value);
  if (next.has(strategyId)) next.delete(strategyId);
  else next.add(strategyId);
  mutedStrategyIds.value = next;
}

function toggleAllStrategyColors(): void {
  mutedStrategyIds.value = allStrategiesVisible.value
    ? new Set(
        selectedPrediction.value?.strategies.map((strategy) => strategy.id) ?? [],
      )
    : new Set();
}

function cellTitle(
  entry: StrategyNumberPrediction,
  strategy: StrategyPrediction,
): string {
  const outcome = actualNumbers.value.has(entry.number) ? " — drawn next" : "";
  const details = entry.details.length > 0 ? ` — ${entry.details.join(" · ")}` : "";
  return `Number ${entry.number}: rank ${entry.rank}, ${strategy.name} score ${scoreLabel(entry)}${details}${outcome} — Ctrl+Click: Possible; Ctrl+Double-click: For Sure`;
}

function allCellTitle(cell: (typeof allReportCells.value)[number]): string {
  const reports = cell.reports.map((report) => report.name).join(", ");
  const outcome = actualNumbers.value.has(cell.number) ? " — drawn next" : "";
  const consensus =
    cell.reports.length >= 3
      ? ` — high agreement (${cell.reports.length} strategies)`
      : "";
  return `Number ${cell.number}${outcome}${consensus} — predicted by ${reports || "no report"} — Ctrl+Click: Possible; Ctrl+Double-click: For Sure`;
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

function clearNumberActionTimer(): void {
  if (numberActionTimer !== null) {
    clearTimeout(numberActionTimer);
    numberActionTimer = null;
  }
}

function sendNumberToPossibleDraw(
  number: number,
  state: PossibleDrawNumberState,
): void {
  void window.randAiDesktop?.sendPredictionNumberToPossibleDraw({ number, state });
}

function handlePredictionClick(event: MouseEvent, number: number): void {
  if (!event.ctrlKey) return;
  event.preventDefault();
  clearNumberActionTimer();
  numberActionTimer = setTimeout(() => {
    sendNumberToPossibleDraw(number, "possible");
    numberActionTimer = null;
  }, 350);
}

function handlePredictionDoubleClick(event: MouseEvent, number: number): void {
  if (!event.ctrlKey) return;
  event.preventDefault();
  clearNumberActionTimer();
  sendNumberToPossibleDraw(number, "for-sure");
}

onBeforeUnmount(clearNumberActionTimer);
</script>

<template>
  <PredictionWorkspaceNavigation active="predictions">
    <template #controls>
      <div
        v-if="selectedPrediction"
        class="reference-buttons"
        aria-label="Prediction history navigation"
      >
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
      <div v-if="selectedPrediction" class="prediction-reference-summary">
        <span>After draw</span>
        <strong>{{ selectedPrediction.referenceDrawNumber }}</strong>
        <span aria-hidden="true">→</span>
        <span>{{ selectedPrediction.targetDrawNumber }}</span>
      </div>
    </template>
  </PredictionWorkspaceNavigation>

  <section
    v-if="selectedPrediction && selectedPrediction.strategies.length > 0"
    class="combined-prediction-view"
  >
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
          <span>{{ strategy.name }}</span>
          <small>{{ efficacyDifferenceFor(strategy) }}</small>
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

        <section
          v-if="selectedStrategy && selectedStrategyEfficacy"
          class="strategy-efficacy-card"
          :class="efficacyClass(selectedStrategyEfficacy)"
          aria-label="Historical efficacy compared with random selections"
        >
          <header>
            <div>
              <span>Historical Top-6 efficacy</span>
              <strong>{{ efficacyVerdict(selectedStrategyEfficacy) }}</strong>
            </div>
            <b>{{ hitDifferenceLabel(selectedStrategyEfficacy) }}</b>
          </header>
          <dl>
            <div>
              <dt>Completed predictions</dt>
              <dd>{{ selectedStrategyEfficacy.evaluatedDraws }}</dd>
            </div>
            <div>
              <dt>{{ selectedStrategy.name }} hits</dt>
              <dd>
                {{ formatHitTotal(selectedStrategyEfficacy.strategyHits) }}
                <small>
                  {{ selectedStrategyEfficacy.averageHitsPerDraw.toFixed(3) }}/draw
                </small>
              </dd>
            </div>
            <div>
              <dt>Random mean</dt>
              <dd>
                {{ formatHitTotal(selectedStrategyEfficacy.randomHits) }}
                <small>
                  {{ selectedStrategyEfficacy.randomAverageHitsPerDraw.toFixed(3) }}/draw
                </small>
              </dd>
            </div>
            <div>
              <dt>Random 95% range</dt>
              <dd>
                {{ efficacyRandomRangeLabel }}
                <small>
                  Random ≥ result: {{ randomTailLabel(selectedStrategy) }}
                </small>
              </dd>
            </div>
          </dl>
          <p>
            Every completed record in the imported database is evaluated
            walk-forward: six predicted numbers are compared with the actual next
            draw. Positive lift means more matches than the random mean. Retest
            uses 10,000 random datasets to estimate the empirical random range and
            the chance of reaching the strategy result; it is evidence for
            comparison, not a guarantee of future results.
          </p>
        </section>

        <div
          v-if="selectedStrategyId === 'all'"
          class="all-predictions-wrapper"
        >
          <section class="prediction-coverage-report" aria-label="Strategy coverage report">
            <header>
              <div>
                <strong>Strategy coverage report</strong>
                <span>Cumulative coverage across all 49 numbers</span>
              </div>
              <button
                type="button"
                :disabled="selectedCoverageThreshold === null"
                @click="selectedCoverageThreshold = null"
              >
                No circles
              </button>
            </header>
            <ul role="radiogroup" aria-label="Highlight a strategy coverage zone">
              <li
                v-for="entry in coverageReport"
                :key="entry.threshold"
                :class="{ selected: selectedCoverageThreshold === entry.threshold }"
              >
                <label>
                  <input
                    v-model="selectedCoverageThreshold"
                    type="radio"
                    name="coverage-zone"
                    :value="entry.threshold"
                  />
                  <span>
                    At least {{ entry.threshold }}
                    {{ entry.threshold === 1 ? "strategy" : "strategies" }}
                  </span>
                  <strong>{{ entry.numberCount }} numbers</strong>
                </label>
              </li>
            </ul>
          </section>
          <div class="prediction-color-legend" role="group" aria-label="Prediction report colors">
            <label>
              <input
                type="checkbox"
                :checked="allStrategiesVisible"
                @change="toggleAllStrategyColors"
              />
              <span>All colors</span>
            </label>
            <label
              v-for="strategy in selectedPrediction.strategies"
              :key="strategy.id"
            >
              <input
                type="checkbox"
                :checked="!mutedStrategyIds.has(strategy.id)"
                :style="{ '--legend-color': strategyColor(strategy.id) }"
                @change="toggleStrategyColor(strategy.id)"
              />
              <span>{{ strategy.name }}</span>
            </label>
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
                'is-in-selected-coverage-zone':
                  selectedCoverageThreshold !== null &&
                  cell.reports.length >= selectedCoverageThreshold,
              }"
              :style="sectorStyle(cell)"
              :title="allCellTitle(cell)"
              role="gridcell"
              @click="handlePredictionClick($event, cell.number)"
              @dblclick="handlePredictionDoubleClick($event, cell.number)"
            >
              <strong>{{ cell.number }}</strong>
            </article>
          </div>
          <section
            class="prediction-efficacy-overview"
            aria-label="All strategies compared with random selections"
          >
            <header>
              <div>
                <strong>Historical efficacy versus random</strong>
                <span>
                  Walk-forward Top-6 matches · Retest compares each strategy
                  with 10,000 random datasets
                </span>
              </div>
              <small>Uniform expectation: 0.735 hits per draw</small>
            </header>
            <div class="efficacy-range-controls">
              <label class="efficacy-count-control">
                <span>Completed draws to compare</span>
                <input
                  v-model.number="efficacyDrawCount"
                  type="number"
                  min="1"
                  :max="maximumEfficacyDraws"
                  :disabled="maximumEfficacyDraws === 0"
                  @change="normalizeEfficacyDrawCount"
                />
                <small>Maximum {{ maximumEfficacyDraws }}</small>
              </label>
              <fieldset>
                <legend>Take records from</legend>
                <label>
                  <input
                    v-model="efficacyRangeAnchor"
                    type="radio"
                    value="first"
                  />
                  <span>First N — top to bottom</span>
                </label>
                <label>
                  <input
                    v-model="efficacyRangeAnchor"
                    type="radio"
                    value="latest"
                  />
                  <span>Latest N records</span>
                </label>
              </fieldset>
              <output>
                <strong>{{ appliedEfficacyDrawCount }} tests</strong>
                <span>{{ efficacyRangeLabel }}</span>
              </output>
              <div class="efficacy-retest-control">
                <button
                  type="button"
                  :disabled="
                    selectedEfficacyHistory.length === 0 || isEfficacyRetesting
                  "
                  title="Run 10,000 random Top-6 datasets for the selected historical draws"
                  @click="retestRandomDataset"
                >
                  {{ isEfficacyRetesting ? "Retesting…" : "Retest" }}
                </button>
                <small>{{ efficacyRandomDatasetLabel }}</small>
              </div>
            </div>
            <div class="efficacy-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Strategy name</th>
                    <th>Tests</th>
                    <th>Strategy hits</th>
                    <th>Random mean</th>
                    <th>Random 95%</th>
                    <th>Hits/draw</th>
                    <th>Lift vs mean</th>
                    <th>Random ≥ result</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(strategy, index) in efficacyRanking"
                    :key="strategy.id"
                    :class="efficacyClass(efficacyFor(strategy))"
                  >
                    <th scope="row">
                      <span class="efficacy-rank">#{{ index + 1 }}</span>
                      {{ strategyFullName(strategy) }}
                    </th>
                    <td>{{ efficacyFor(strategy)?.evaluatedDraws ?? 0 }}</td>
                    <td>
                      {{ formatHitTotal(efficacyFor(strategy)?.strategyHits ?? 0) }}
                    </td>
                    <td>
                      {{ formatHitTotal(efficacyFor(strategy)?.randomHits ?? 0) }}
                    </td>
                    <td>{{ efficacyRandomRangeLabel }}</td>
                    <td>
                      {{ (efficacyFor(strategy)?.averageHitsPerDraw ?? 0).toFixed(3) }}
                    </td>
                    <td>{{ efficacyDifferenceFor(strategy) }}</td>
                    <td>{{ randomTailLabel(strategy) }}</td>
                    <td>{{ efficacyVerdictFor(strategy) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
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
                @click="handlePredictionClick($event, entry.number)"
                @dblclick="handlePredictionDoubleClick($event, entry.number)"
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
