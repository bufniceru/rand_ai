<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import EfficacyComparisonChart from "../components/EfficacyComparisonChart.vue";
import StrategySelectionPanel from "../components/StrategySelectionPanel.vue";
import {
  RANDOM_BENCHMARK_SIMULATIONS,
  randomTailProbability,
  runPooledRandomBenchmarks,
  type RandomBenchmarkSummary,
} from "../lib/randomBenchmark";
import {
  RANDOM_HITS_PER_EVALUATION,
  buildFamilyEfficacy,
  type FamilyEfficacy,
} from "../lib/familyEfficacy";
import { groupStrategiesByFamily } from "../lib/strategyFamilies";
import { strategyColor } from "../lib/strategyColors";
import { enabledStrategyPlugins } from "../lib/strategySelection";
import type { EfficacyChartRow } from "../lib/efficacyChart";
import type {
  PossibleDrawNumberState,
  PredictionSuite,
  StrategyEfficacy,
  StrategyEfficacyRecord,
  StrategyId,
  StrategyNumberPrediction,
  StrategyPlugin,
  StrategyPrediction,
} from "../types";

const props = defineProps<{
  predictionSuites: PredictionSuite[];
  efficacyHistory: StrategyEfficacyRecord[];
  strategyPlugins: StrategyPlugin[];
  enabledStrategyIds: StrategyId[];
  strategySelectionBusy: boolean;
  embedded?: boolean;
}>();
const emit = defineEmits<{
  sendNumber: [request: { number: number; state: PossibleDrawNumberState }];
  applyStrategies: [strategyIds: StrategyId[]];
}>();

const predictionControlTabs = [
  "strategies",
  "selection",
  "colors",
  "coverage",
] as const;
type PredictionControlTab = (typeof predictionControlTabs)[number];

const referenceOffset = ref(0);
const selectedStrategyId = ref("all");
const selectedControlTab = ref<PredictionControlTab>("strategies");
const selectedCoverageThreshold = ref<number | null>(null);
const mutedStrategyIds = ref<Set<string>>(new Set());
const isStrategySidebarExpanded = ref(true);
const isNumbersGridExpanded = ref(true);
const isEfficacyExpanded = ref(true);
const isFamilyEfficacyExpanded = ref(true);
const efficacyDrawCount = ref(0);
const efficacyRangeAnchor = ref<"first" | "latest">("first");
const efficacyRandomBenchmarks = shallowRef<
  ReadonlyMap<number, RandomBenchmarkSummary>
>(new Map());
const efficacyRetestCount = ref(0);
const isEfficacyRetesting = ref(false);
let numberActionTimer: ReturnType<typeof setTimeout> | null = null;

const maximumOffset = computed(() => Math.max(0, props.predictionSuites.length - 1));
const selectedIndex = computed(() =>
  Math.max(0, props.predictionSuites.length - 1 - referenceOffset.value),
);
const selectedPrediction = computed(() => props.predictionSuites[selectedIndex.value] ?? null);
const selectableStrategyPlugins = computed(() =>
  enabledStrategyPlugins(props.strategyPlugins, props.enabledStrategyIds),
);
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
const selectedStrategyNumbersByScore = computed(() =>
  [...(selectedStrategy.value?.numbers ?? [])].sort(
    (left, right) =>
      right.score - left.score ||
      left.rank - right.rank ||
      left.number - right.number,
  ),
);
const selectedStrategyNumberByNumber = computed(
  () =>
    new Map(
      selectedStrategyNumbersByScore.value.map((entry) => [entry.number, entry]),
    ),
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
      strategyEntry: selectedStrategyNumberByNumber.value.get(number) ?? null,
      reports: (selectedPrediction.value?.strategies ?? [])
        .filter((strategy) => strategy.topNumbers.includes(number))
        .map((strategy) => ({
          id: strategy.id,
          name: strategyFullName(strategy),
          color: strategyDisplayColor(strategy.id),
        })),
    };
  }),
);
const orderedReportCells = computed(() => {
  if (!selectedStrategy.value) return allReportCells.value;
  return selectedStrategyNumbersByScore.value
    .map((entry) => allReportCells.value[entry.number - 1])
    .filter((cell) => cell !== undefined);
});
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
const efficacyRandomBenchmark = computed(
  () => efficacyRandomBenchmarks.value.get(1) ?? null,
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
  efficacyRandomBenchmarks.value.size > 0
    ? `${RANDOM_BENCHMARK_SIMULATIONS.toLocaleString()} pooled datasets · run #${efficacyRetestCount.value}`
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
      (rightEfficacy?.averageHitsPerDraw ?? -Number.MAX_SAFE_INTEGER) -
      (leftEfficacy?.averageHitsPerDraw ?? -Number.MAX_SAFE_INTEGER);
    if (difference !== 0) return difference;
    return left.name.localeCompare(right.name);
  }),
);
const familyEfficacyRanking = computed(() =>
  buildFamilyEfficacy(
    selectedPrediction.value?.strategies ?? [],
    efficacyByStrategy.value,
    selectedEfficacyHistory.value.length,
    efficacyRandomBenchmarks.value,
  ),
);
const groupedFamilyEfficacyRanking = computed(() =>
  groupStrategiesByFamily(
    efficacyRanking.value,
    familyEfficacyRanking.value.map((family) => family.id),
  ),
);
const familyStrategyCounts = computed(() => [
  ...new Set(familyEfficacyRanking.value.map((family) => family.strategyCount)),
]);
const strategyEfficacyChartRows = computed<EfficacyChartRow[]>(() =>
  selectedEfficacyHistory.value.length === 0
    ? []
    : efficacyRanking.value.map((strategy) => {
        const efficacy = efficacyFor(strategy);
        const rate = efficacy?.averageHitsPerDraw ?? 0;
        return {
          id: strategy.id,
          label: strategyFullName(strategy),
          rate,
          normalizedLift: rate - RANDOM_HITS_PER_EVALUATION,
          detail: `${efficacy?.evaluatedDraws ?? 0} selected draws`,
        };
      }),
);
const familyEfficacyChartRows = computed<EfficacyChartRow[]>(() =>
  selectedEfficacyHistory.value.length === 0
    ? []
    : familyEfficacyRanking.value.map((family) => ({
        id: family.id,
        label: family.label,
        rate: family.hitsPerEvaluation,
        normalizedLift: family.normalizedLift,
        detail: `${family.strategyCount} enabled strategies · ${family.evaluations} strategy-draw evaluations`,
      })),
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
  efficacyRandomBenchmarks.value = new Map();
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
    efficacyRandomBenchmarks.value = runPooledRandomBenchmarks(
      drawCount,
      [1, ...familyStrategyCounts.value],
    );
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

function familyRandomRangeLabel(family: FamilyEfficacy): string {
  const benchmark = family.randomBenchmark;
  return benchmark
    ? `${benchmark.lower95Hits}–${benchmark.upper95Hits}`
    : "Run Retest";
}

function familyRandomTailLabel(family: FamilyEfficacy): string {
  const benchmark = family.randomBenchmark;
  if (!benchmark) return "Run Retest";
  const probability = randomTailProbability(benchmark, family.familyHits);
  if (probability < 0.001) return "<0.1%";
  if (probability < 0.1) return `${(probability * 100).toFixed(1)}%`;
  return `${Math.round(probability * 100)}%`;
}

function familyLiftLabel(family: FamilyEfficacy): string {
  if (family.evaluations === 0) return "Awaiting results";
  const lift = family.normalizedLift.toFixed(3);
  return family.normalizedLift > 0 ? `+${lift}` : lift;
}

function familyVerdict(family: FamilyEfficacy): string {
  if (family.evaluations === 0) return "No completed predictions yet";
  if (family.normalizedLift > 0) return "Beats random mean";
  if (family.normalizedLift < 0) return "Trails random mean";
  return "Tied with random mean";
}

function familyEfficacyClass(family: FamilyEfficacy): string {
  if (family.evaluations === 0) return "is-pending";
  if (family.normalizedLift > 0) return "is-ahead";
  if (family.normalizedLift < 0) return "is-behind";
  return "is-tied";
}

function familyStrategyNames(family: FamilyEfficacy): string {
  const names = new Map(
    (selectedPrediction.value?.strategies ?? []).map((strategy) => [
      strategy.id,
      strategyFullName(strategy),
    ]),
  );
  return family.strategyIds
    .map((strategyId) => names.get(strategyId) ?? strategyId)
    .join(", ");
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
    mksp: "Markov Spaces",
    mknp: "Markov Normalized Positions",
    mkrd: "Markov Relative Dispersion",
    bayesian: "Bayesian",
    predictive_grid: "Predictive Score Grid",
    co_occurrence: "Next Draw Co-occurrence",
    doublet_triplet_markov: "Doublet & Triplet Markov",
    mixed: "Mixed Prediction",
    svc: "Support Vector Classifier",
    tbl: "Temporal Behavior Learning",
    sklearn_svm: "Scikit Online SVM",
    lag_logistic: "Lagged Logistic",
    sparse_neural_ticket: "Sparse Neural Ticket (Experimental)",
    cis: "Collective Intelligence Strategy",
    residual_coverage: "Residual Coverage",
    chained: "Chained Strategy",
  }[strategy.id] ?? strategy.name;
}

function strategyDisplayColor(strategyId: StrategyId): string {
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

function selectControlTabAndFocus(tab: PredictionControlTab): void {
  selectedControlTab.value = tab;
  nextTick(() => {
    document.getElementById(`prediction-control-tab-${tab}`)?.focus();
  });
}

function selectAdjacentControlTab(offset: number): void {
  const currentIndex = predictionControlTabs.indexOf(selectedControlTab.value);
  const nextIndex =
    (currentIndex + offset + predictionControlTabs.length) %
    predictionControlTabs.length;
  selectControlTabAndFocus(predictionControlTabs[nextIndex]);
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
  emit("sendNumber", { number, state });
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
  <section
    v-if="selectedPrediction && selectedPrediction.strategies.length > 0"
    class="combined-prediction-view"
  >
    <Teleport to="#prediction-toolbar-navigation">
      <div class="prediction-reference-toolbar">
        <div
          class="reference-buttons"
          aria-label="Prediction history navigation"
        >
          <button
            type="button"
            :disabled="referenceOffset >= maximumOffset"
            aria-label="First prediction"
            title="First"
            @click="referenceOffset = maximumOffset"
          >
            ⏮
          </button>
          <button
            type="button"
            :disabled="referenceOffset >= maximumOffset"
            aria-label="Previous prediction"
            title="Previous"
            @click="referenceOffset += 1"
          >
            ◀
          </button>
          <output
            class="prediction-reference-summary"
            :aria-label="`Reference draw ${selectedPrediction.referenceDrawNumber}`"
          >
            <strong>{{ selectedPrediction.referenceDrawNumber }}</strong>
          </output>
          <button
            type="button"
            :disabled="referenceOffset === 0"
            aria-label="Next prediction"
            title="Next"
            @click="referenceOffset -= 1"
          >
            ▶
          </button>
          <button
            type="button"
            :disabled="referenceOffset === 0"
            aria-label="Last prediction"
            title="Last"
            @click="referenceOffset = 0"
          >
            ⏭
          </button>
        </div>
      </div>
    </Teleport>

    <div
      class="prediction-workspace-layout"
      :class="{ 'is-strategy-sidebar-collapsed': !isStrategySidebarExpanded }"
    >
      <aside
        v-show="isStrategySidebarExpanded"
        id="prediction-strategy-sidebar"
        class="prediction-strategy-sidebar"
        aria-label="Prediction strategy selectors"
      >
        <section
          class="prediction-control-cassette"
          aria-labelledby="prediction-controls-title"
        >
          <div class="prediction-controls-heading">
            <h2 id="prediction-controls-title" class="prediction-controls-title">
              Strategies
            </h2>
            <button
              type="button"
              class="prediction-sidebar-toggle"
              aria-controls="prediction-strategy-sidebar"
              :aria-expanded="isStrategySidebarExpanded"
              aria-label="Collapse strategies pane"
              title="Collapse strategies pane"
              @click="isStrategySidebarExpanded = false"
            >
              «
            </button>
          </div>
          <div
            class="prediction-control-tablist"
            role="tablist"
            aria-label="Strategy control groups"
            @keydown.left.prevent="selectAdjacentControlTab(-1)"
            @keydown.right.prevent="selectAdjacentControlTab(1)"
            @keydown.home.prevent="selectControlTabAndFocus('strategies')"
            @keydown.end.prevent="selectControlTabAndFocus('coverage')"
          >
            <button
              id="prediction-control-tab-strategies"
              type="button"
              role="tab"
              :aria-selected="selectedControlTab === 'strategies'"
              aria-controls="prediction-control-panel-strategies"
              :tabindex="selectedControlTab === 'strategies' ? 0 : -1"
              :class="{ active: selectedControlTab === 'strategies' }"
              @click="selectedControlTab = 'strategies'"
            >
              List
            </button>
            <button
              id="prediction-control-tab-selection"
              type="button"
              role="tab"
              :aria-selected="selectedControlTab === 'selection'"
              aria-controls="prediction-control-panel-selection"
              :tabindex="selectedControlTab === 'selection' ? 0 : -1"
              :class="{ active: selectedControlTab === 'selection' }"
              @click="selectedControlTab = 'selection'"
            >
              Selection
            </button>
            <button
              id="prediction-control-tab-colors"
              type="button"
              role="tab"
              :aria-selected="selectedControlTab === 'colors'"
              aria-controls="prediction-control-panel-colors"
              :tabindex="selectedControlTab === 'colors' ? 0 : -1"
              :class="{ active: selectedControlTab === 'colors' }"
              @click="selectedControlTab = 'colors'"
            >
              Colors
            </button>
            <button
              id="prediction-control-tab-coverage"
              type="button"
              role="tab"
              :aria-selected="selectedControlTab === 'coverage'"
              aria-controls="prediction-control-panel-coverage"
              :tabindex="selectedControlTab === 'coverage' ? 0 : -1"
              :class="{ active: selectedControlTab === 'coverage' }"
              @click="selectedControlTab = 'coverage'"
            >
              Report
            </button>
          </div>

        <div
          v-show="selectedControlTab === 'strategies'"
          id="prediction-control-panel-strategies"
          class="prediction-control-panel"
          role="tabpanel"
          aria-labelledby="prediction-control-tab-strategies"
        >
          <div class="prediction-strategy-tabs" aria-label="Prediction strategy">
            <button
              type="button"
              :aria-pressed="selectedStrategyId === 'all'"
              :class="{ active: selectedStrategyId === 'all' }"
              @click="selectedStrategyId = 'all'"
            >
              All
            </button>
            <button
              v-for="(strategy, index) in efficacyRanking"
              :key="strategy.id"
              type="button"
              :aria-pressed="strategy.id === selectedStrategyId"
              :class="{ active: strategy.id === selectedStrategyId }"
              :title="`#${index + 1} by effectiveness · ${strategyFullName(strategy)}`"
              @click="selectedStrategyId = strategy.id"
            >
              <span>
                <b>#{{ index + 1 }}</b>
                {{ strategyFullName(strategy) }}
              </span>
            </button>
          </div>
        </div>

        <div
          v-show="selectedControlTab === 'selection'"
          id="prediction-control-panel-selection"
          class="prediction-control-panel"
          role="tabpanel"
          aria-labelledby="prediction-control-tab-selection"
        >
          <StrategySelectionPanel
            :plugins="selectableStrategyPlugins"
            :enabled-strategy-ids="enabledStrategyIds"
            :busy="strategySelectionBusy"
            @apply="emit('applyStrategies', $event)"
          />
        </div>

        <div
          v-show="selectedControlTab === 'colors'"
          id="prediction-control-panel-colors"
          class="prediction-control-panel"
          role="tabpanel"
          aria-labelledby="prediction-control-tab-colors"
        >
          <div
            class="prediction-color-legend"
            role="group"
            aria-label="Prediction report colors"
          >
            <label>
              <input
                type="checkbox"
                :checked="allStrategiesVisible"
                @change="toggleAllStrategyColors"
              />
              <span>All colors</span>
            </label>
            <section
              v-for="group in groupedFamilyEfficacyRanking"
              :key="group.id"
              class="prediction-strategy-family"
              :style="{ '--family-color': group.color }"
              :aria-labelledby="`prediction-color-family-${group.id}`"
            >
              <h3 :id="`prediction-color-family-${group.id}`">
                {{ group.label }}
              </h3>
              <label
                v-for="strategy in group.strategies"
                :key="strategy.id"
                class="prediction-color-row"
                :style="{ '--legend-color': strategyColor(strategy.id) }"
              >
                <input
                  type="checkbox"
                  :checked="!mutedStrategyIds.has(strategy.id)"
                  @change="toggleStrategyColor(strategy.id)"
                />
                <span>{{ strategyFullName(strategy) }}</span>
                <i class="prediction-color-swatch" aria-hidden="true"></i>
              </label>
            </section>
          </div>
        </div>

        <div
          v-show="selectedControlTab === 'coverage'"
          id="prediction-control-panel-coverage"
          class="prediction-control-panel"
          role="tabpanel"
          aria-labelledby="prediction-control-tab-coverage"
        >
          <section
            class="prediction-coverage-report"
            aria-label="Strategy coverage report"
          >
            <ul role="radiogroup" aria-label="Highlight a strategy coverage zone">
              <li :class="{ selected: selectedCoverageThreshold === null }">
                <label>
                  <input
                    v-model="selectedCoverageThreshold"
                    type="radio"
                    name="coverage-zone"
                    :value="null"
                  />
                  <span>No circles</span>
                  <strong>Off</strong>
                </label>
              </li>
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
          </div>
        </section>
      </aside>

      <button
        v-if="!isStrategySidebarExpanded"
        type="button"
        class="prediction-sidebar-toggle prediction-sidebar-expand-toggle"
        aria-controls="prediction-strategy-sidebar"
        :aria-expanded="isStrategySidebarExpanded"
        aria-label="Expand strategies pane"
        title="Expand strategies pane"
        @click="isStrategySidebarExpanded = true"
      >
        »
      </button>

      <div class="prediction-workspace-content">
        <section
          class="prediction-grid-section"
          :class="{ 'is-collapsed': !isNumbersGridExpanded }"
        >
          <header class="prediction-collapsible-header">
            <div>
              <strong>
                Numbers grid ·
                {{ selectedStrategy ? strategyFullName(selectedStrategy) : "All strategies" }}
              </strong>
            </div>
            <button
              type="button"
              class="prediction-collapsible-toggle"
              :aria-expanded="isNumbersGridExpanded"
              aria-controls="prediction-numbers-grid-panel"
              :aria-label="isNumbersGridExpanded ? 'Collapse numbers grid' : 'Expand numbers grid'"
              :title="isNumbersGridExpanded ? 'Collapse numbers grid' : 'Expand numbers grid'"
              @click="isNumbersGridExpanded = !isNumbersGridExpanded"
            >
              {{ isNumbersGridExpanded ? "«" : "»" }}
            </button>
          </header>
          <div
            v-show="isNumbersGridExpanded"
            id="prediction-numbers-grid-panel"
            class="prediction-grid-panel"
          >
            <div
              class="all-predictions-grid"
              :class="{ 'is-historical-prediction': actualNumbers.size > 0 }"
              role="grid"
              :aria-label="
                selectedStrategy
                  ? `${strategyFullName(selectedStrategy)} score order with all prediction report colors`
                  : 'All prediction reports by number from 1 through 49'
              "
            >
              <article
                v-for="cell in orderedReportCells"
                :key="cell.number"
                class="all-predictions-cell"
                :class="{
                  'has-prediction': cell.reports.length > 0,
                  'is-drawn': actualNumbers.has(cell.number),
                  'is-in-selected-coverage-zone':
                    selectedCoverageThreshold !== null &&
                    cell.reports.length >= selectedCoverageThreshold,
                }"
                role="gridcell"
                tabindex="0"
                :aria-label="`Prediction number ${cell.number}`"
                :aria-describedby="`prediction-number-tooltip-${cell.number}`"
                @click="handlePredictionClick($event, cell.number)"
                @dblclick="handlePredictionDoubleClick($event, cell.number)"
              >
                <span
                  class="prediction-cell-color"
                  :style="sectorStyle(cell)"
                  aria-hidden="true"
                ></span>
                <strong>{{ cell.number }}</strong>
                <aside
                  :id="`prediction-number-tooltip-${cell.number}`"
                  class="prediction-number-tooltip"
                  role="tooltip"
                >
                  <header>
                    <div>
                      <span>Prediction number</span>
                      <strong>{{ cell.number }}</strong>
                    </div>
                    <span class="prediction-tooltip-count">
                      {{ cell.reports.length }}
                      {{ cell.reports.length === 1 ? "strategy" : "strategies" }}
                    </span>
                  </header>

                  <div v-if="cell.strategyEntry" class="prediction-tooltip-metrics">
                    <div>
                      <span>Selected rank</span>
                      <strong>#{{ cell.strategyEntry.rank }}</strong>
                    </div>
                    <div>
                      <span>Score</span>
                      <strong>{{ scoreLabel(cell.strategyEntry) }}</strong>
                    </div>
                  </div>

                  <p v-if="actualNumbers.has(cell.number)" class="prediction-tooltip-outcome">
                    Drawn in target draw {{ selectedPrediction?.targetDrawNumber }}
                  </p>

                  <section class="prediction-tooltip-strategies">
                    <h3>Strategies implying this number</h3>
                    <ul
                      v-if="cell.reports.length > 0"
                      :class="{ 'is-long': cell.reports.length > 5 }"
                    >
                      <li
                        v-for="report in cell.reports"
                        :key="report.id"
                        :style="{ '--strategy-color': strategyColor(report.id) }"
                      >
                        {{ report.name }}
                      </li>
                    </ul>
                    <p v-else>No active strategy implies this number.</p>
                  </section>

                  <footer>
                    <span><kbd>Ctrl</kbd> + click: Possible</span>
                    <span><kbd>Ctrl</kbd> + double-click: For sure</span>
                  </footer>
                </aside>
              </article>
            </div>
          </div>
        </section>

        <div v-if="selectedStrategy" class="prediction-strategy-heading">
          <div>
            <span>Prediction</span>
            <h1>{{ strategyFullName(selectedStrategy) }}</h1>
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
              <dt>{{ strategyFullName(selectedStrategy) }} hits</dt>
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

        <template v-if="selectedStrategyId === 'all'">
          <section
            class="prediction-efficacy-controls"
            aria-label="Historical efficacy comparison controls"
          >
            <header>
              <strong>Historical efficacy comparison</strong>
              <small>Uniform expectation: 0.735 hits per strategy-draw</small>
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
                <strong>{{ appliedEfficacyDrawCount }} draws</strong>
                <span>{{ efficacyRangeLabel }}</span>
              </output>
              <div class="efficacy-retest-control">
                <button
                  type="button"
                  :disabled="
                    selectedEfficacyHistory.length === 0 || isEfficacyRetesting
                  "
                  title="Run 10,000 random Top-6 datasets for the strategy and family reports"
                  @click="retestRandomDataset"
                >
                  {{ isEfficacyRetesting ? "Retesting…" : "Retest" }}
                </button>
                <small>{{ efficacyRandomDatasetLabel }}</small>
              </div>
            </div>
          </section>

          <section
            class="prediction-efficacy-overview"
            :class="{ 'is-collapsed': !isEfficacyExpanded }"
            aria-label="All strategies compared with random selections"
          >
            <header>
              <div>
                <strong>Historical strategies efficacy versus random</strong>
              </div>
              <div class="prediction-collapsible-actions">
                <small>{{ efficacyRanking.length }} enabled strategies</small>
                <button
                  type="button"
                  class="prediction-collapsible-toggle"
                  :aria-expanded="isEfficacyExpanded"
                  aria-controls="prediction-strategy-efficacy-panel"
                  :aria-label="isEfficacyExpanded ? 'Collapse historical strategies efficacy report' : 'Expand historical strategies efficacy report'"
                  :title="isEfficacyExpanded ? 'Collapse historical strategies efficacy report' : 'Expand historical strategies efficacy report'"
                  @click="isEfficacyExpanded = !isEfficacyExpanded"
                >
                  {{ isEfficacyExpanded ? "«" : "»" }}
                </button>
              </div>
            </header>
            <div
              v-show="isEfficacyExpanded"
              id="prediction-strategy-efficacy-panel"
            >
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
              <EfficacyComparisonChart
                id="strategy-efficacy-rate-chart"
                title="Strategy hits per draw"
                :rows="strategyEfficacyChartRows"
                mode="rate"
                rate-unit="draw"
              />
              <EfficacyComparisonChart
                id="strategy-efficacy-lift-chart"
                title="Strategy normalized lift from random"
                :rows="strategyEfficacyChartRows"
                mode="lift"
                rate-unit="draw"
              />
            </div>
          </section>

          <section
            class="prediction-efficacy-overview"
            :class="{ 'is-collapsed': !isFamilyEfficacyExpanded }"
            aria-label="Strategy families compared with random selections"
          >
            <header>
              <div>
                <strong>Historical families efficacy versus random</strong>
              </div>
              <div class="prediction-collapsible-actions">
                <small>{{ familyEfficacyRanking.length }} active families</small>
                <button
                  type="button"
                  class="prediction-collapsible-toggle"
                  :aria-expanded="isFamilyEfficacyExpanded"
                  aria-controls="prediction-family-efficacy-panel"
                  :aria-label="isFamilyEfficacyExpanded ? 'Collapse historical families efficacy report' : 'Expand historical families efficacy report'"
                  :title="isFamilyEfficacyExpanded ? 'Collapse historical families efficacy report' : 'Expand historical families efficacy report'"
                  @click="isFamilyEfficacyExpanded = !isFamilyEfficacyExpanded"
                >
                  {{ isFamilyEfficacyExpanded ? "«" : "»" }}
                </button>
              </div>
            </header>
            <div
              v-show="isFamilyEfficacyExpanded"
              id="prediction-family-efficacy-panel"
            >
              <p class="family-efficacy-note">
                Family results pool every enabled member strategy across the
                selected draws. Each member strategy on each draw is one evaluation;
                this is not a new family consensus ticket.
              </p>
              <div class="efficacy-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Family</th>
                      <th>Strategies</th>
                      <th>Draws</th>
                      <th>Evaluations</th>
                      <th>Pooled hits</th>
                      <th>Random mean</th>
                      <th>Random 95%</th>
                      <th>Hits/evaluation</th>
                      <th>Lift/evaluation</th>
                      <th>Random ≥ result</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(family, index) in familyEfficacyRanking"
                      :key="family.id"
                      :class="familyEfficacyClass(family)"
                    >
                      <th scope="row" :title="familyStrategyNames(family)">
                        <span class="efficacy-rank">#{{ index + 1 }}</span>
                        {{ family.label }}
                      </th>
                      <td :title="familyStrategyNames(family)">
                        {{ family.strategyCount }}
                      </td>
                      <td>{{ family.evaluatedDraws }}</td>
                      <td>{{ family.evaluations }}</td>
                      <td>{{ formatHitTotal(family.familyHits) }}</td>
                      <td>{{ formatHitTotal(family.randomHits) }}</td>
                      <td>{{ familyRandomRangeLabel(family) }}</td>
                      <td>{{ family.hitsPerEvaluation.toFixed(3) }}</td>
                      <td>{{ familyLiftLabel(family) }}</td>
                      <td>{{ familyRandomTailLabel(family) }}</td>
                      <td>{{ familyVerdict(family) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <EfficacyComparisonChart
                id="family-efficacy-rate-chart"
                title="Family hits per evaluation"
                :rows="familyEfficacyChartRows"
                mode="rate"
                rate-unit="evaluation"
              />
              <EfficacyComparisonChart
                id="family-efficacy-lift-chart"
                title="Family normalized lift from random"
                :rows="familyEfficacyChartRows"
                mode="lift"
                rate-unit="evaluation"
              />
            </div>
          </section>
        </template>

      </div>
    </div>
  </section>

  <section v-else class="dialog-empty-state">
    <strong>
      {{
        enabledStrategyIds.length === 0 || selectedPrediction
          ? "No prediction strategies are active"
          : "No prediction draws are available"
      }}
    </strong>
    <p>
      {{
        enabledStrategyIds.length === 0
          ? "Enable at least one strategy in Settings and run the analysis again."
          : selectedPrediction
            ? "The enabled selection produced no strategy results."
          : "The imported dataset does not contain any draws."
      }}
    </p>
    <StrategySelectionPanel
      v-if="selectableStrategyPlugins.length > 0"
      class="prediction-empty-strategy-selection"
      :plugins="selectableStrategyPlugins"
      :enabled-strategy-ids="enabledStrategyIds"
      :busy="strategySelectionBusy"
      @apply="emit('applyStrategies', $event)"
    />
  </section>
</template>
