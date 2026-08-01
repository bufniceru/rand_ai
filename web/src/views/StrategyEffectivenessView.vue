<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type {
  AnalysisPayload,
  StrategyEfficacyRecord,
  StrategyId,
} from "../types";

const props = defineProps<{ analysis: AnalysisPayload }>();

type HistoryScope = "all" | 100 | 250 | 500;
type EffectivenessMode = "cumulative" | "rolling";

const randomExpectedHits = 36 / 49;
const exactHitLevels = [6, 5, 4, 3, 2, 1] as const;
const chartWidth = 1100;
const chartHeight = 500;
const chartLeft = 64;
const chartRight = 24;
const chartTop = 28;
const chartBottom = 58;
const historyScope = ref<HistoryScope>("all");
const effectivenessMode = ref<EffectivenessMode>("cumulative");
const rollingWindow = ref(25);
const visibleStrategyIds = ref<Set<StrategyId>>(new Set());

const strategyNames: Record<StrategyId, string> = {
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
  cis: "Collective Intelligence Strategy",
  residual_coverage: "Residual Coverage",
  chained: "Chained Strategy",
};

const strategyColors: Record<StrategyId, string> = {
  proximity: "#ffd866",
  freshness: "#fc9867",
  emd: "#a9dc76",
  randomness: "#78dce8",
  fresh_random: "#ab9df2",
  chi_square: "#ab9df2",
  entropy: "#ff6188",
  markov100: "#ffd866",
  mkfr: "#a9dc76",
  mksp: "#78dce8",
  mknp: "#78dce8",
  mkrd: "#ab9df2",
  bayesian: "#ff6188",
  predictive_grid: "#78dce8",
  co_occurrence: "#a9dc76",
  doublet_triplet_markov: "#ab9df2",
  mixed: "#fc9867",
  svc: "#ab9df2",
  tbl: "#78dce8",
  sklearn_svm: "#2fb7a8",
  lag_logistic: "#e8793e",
  cis: "#ff6188",
  residual_coverage: "#a9dc76",
  chained: "#fc9867",
};

const records = computed(() => props.analysis.strategyEfficacyHistory);
const scopedRecords = computed(() => {
  if (historyScope.value === "all") return records.value;
  return records.value.slice(-historyScope.value);
});
const strategyIds = computed(() => {
  const ids = new Set<StrategyId>();
  for (const record of records.value) {
    for (const strategyId of Object.keys(record.strategyHits) as StrategyId[]) {
      ids.add(strategyId);
    }
  }
  return [...ids];
});
const strategyRows = computed(() =>
  strategyIds.value
    .map((id) => {
      const hits = scopedRecords.value.reduce(
        (total, record) => total + (record.strategyHits[id] ?? 0),
        0,
      );
      const average =
        scopedRecords.value.length > 0 ? hits / scopedRecords.value.length : 0;
      const exactHitDraws = exactHitLevels.map((hitCount) => ({
        hitCount,
        draws: scopedRecords.value.filter(
          (record) => (record.strategyHits[id] ?? 0) === hitCount,
        ).length,
      }));
      return {
        id,
        name: strategyNames[id] ?? id,
        color: strategyColors[id] ?? "#727072",
        hits,
        average,
        lift: average - randomExpectedHits,
        exactHitDraws,
      };
    })
    .sort(
      (left, right) =>
        right.average - left.average || left.name.localeCompare(right.name),
    ),
);
const allStrategiesExactHitDraws = computed(() =>
  exactHitLevels.map((hitCount, index) => ({
    hitCount,
    draws: strategyRows.value.reduce(
      (total, row) => total + (row.exactHitDraws[index]?.draws ?? 0),
      0,
    ),
  })),
);

watch(
  strategyRows,
  (rows) => {
    if (visibleStrategyIds.value.size > 0 || rows.length === 0) return;
    visibleStrategyIds.value = new Set(rows.slice(0, 6).map((row) => row.id));
  },
  { immediate: true },
);

const lineSeries = computed(() =>
  strategyRows.value
    .filter((row) => visibleStrategyIds.value.has(row.id))
    .map((row) => ({
      ...row,
      values: effectivenessValues(scopedRecords.value, row.id),
    })),
);
const chartMaximum = computed(() => {
  const values = lineSeries.value.flatMap((series) => series.values);
  const maximum = Math.max(randomExpectedHits, ...values, 1);
  return Math.ceil(maximum * 4) / 4;
});
const yTicks = computed(() =>
  Array.from({ length: 6 }, (_value, index) => {
    const value = (chartMaximum.value * index) / 5;
    return { value, y: yForValue(value) };
  }),
);
const xTicks = computed(() => {
  const count = scopedRecords.value.length;
  if (count === 0) return [];
  const tickCount = Math.min(8, count);
  const indexes = new Set<number>();
  for (let tick = 0; tick < tickCount; tick += 1) {
    indexes.add(Math.round((tick * (count - 1)) / Math.max(tickCount - 1, 1)));
  }
  return [...indexes].map((index) => ({
    index,
    draw: scopedRecords.value[index].targetDrawNumber,
  }));
});
const bestStrategy = computed(() => strategyRows.value[0] ?? null);
const totalStrategyHits = computed(() =>
  strategyRows.value.reduce((total, row) => total + row.hits, 0),
);

function effectivenessValues(
  source: StrategyEfficacyRecord[],
  strategyId: StrategyId,
): number[] {
  const hits = source.map((record) => record.strategyHits[strategyId] ?? 0);
  if (effectivenessMode.value === "cumulative") {
    let total = 0;
    return hits.map((value, index) => {
      total += value;
      return total / (index + 1);
    });
  }
  return hits.map((_value, index) => {
    const start = Math.max(0, index - rollingWindow.value + 1);
    const window = hits.slice(start, index + 1);
    return window.reduce((total, value) => total + value, 0) / window.length;
  });
}

function xForIndex(index: number): number {
  const count = Math.max(scopedRecords.value.length - 1, 1);
  return chartLeft + (index / count) * (chartWidth - chartLeft - chartRight);
}

function yForValue(value: number): number {
  const plotHeight = chartHeight - chartTop - chartBottom;
  return chartTop + plotHeight - (value / chartMaximum.value) * plotHeight;
}

function seriesPath(values: number[]): string {
  return values
    .map(
      (value, index) =>
        `${index === 0 ? "M" : "L"} ${xForIndex(index).toFixed(2)} ${yForValue(value).toFixed(2)}`,
    )
    .join(" ");
}

function setStrategyVisible(strategyId: StrategyId, visible: boolean): void {
  const next = new Set(visibleStrategyIds.value);
  if (visible) next.add(strategyId);
  else next.delete(strategyId);
  visibleStrategyIds.value = next;
}

function showTopStrategies(): void {
  visibleStrategyIds.value = new Set(
    strategyRows.value.slice(0, 6).map((row) => row.id),
  );
}

function showAllStrategies(): void {
  visibleStrategyIds.value = new Set(strategyIds.value);
}

function clearStrategies(): void {
  visibleStrategyIds.value = new Set();
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}
</script>

<template>
  <section class="workspace-view strategy-effectiveness-view">
    <header class="view-heading prediction-analysis-heading">
      <div>
        <p class="eyebrow">Walk-forward performance</p>
        <h1>Strategy effectiveness timeline</h1>
        <p>
          Follow each strategy’s average correct Top‑6 hits per draw through
          time, using only predictions made before the target draw.
        </p>
      </div>
      <div class="prediction-analysis-controls">
        <label>
          <span>Metric</span>
          <select v-model="effectivenessMode">
            <option value="cumulative">Cumulative average</option>
            <option value="rolling">Rolling average</option>
          </select>
        </label>
        <label v-if="effectivenessMode === 'rolling'">
          <span>Rolling draws</span>
          <input v-model.number="rollingWindow" min="2" max="250" type="number">
        </label>
        <label>
          <span>History</span>
          <select v-model="historyScope">
            <option value="all">All evaluated draws</option>
            <option :value="100">Latest 100</option>
            <option :value="250">Latest 250</option>
            <option :value="500">Latest 500</option>
          </select>
        </label>
      </div>
    </header>

    <div class="prediction-analysis-facts">
      <article>
        <span>Evaluated draws</span>
        <strong>{{ scopedRecords.length.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Strategies measured</span>
        <strong>{{ strategyRows.length }}</strong>
      </article>
      <article>
        <span>Leading strategy</span>
        <strong>{{ bestStrategy?.name ?? "—" }}</strong>
        <small>{{ bestStrategy?.average.toFixed(3) ?? "0.000" }} hits/draw</small>
      </article>
      <article>
        <span>All correct implications</span>
        <strong>{{ totalStrategyHits.toLocaleString() }}</strong>
        <small>Random expectation {{ randomExpectedHits.toFixed(3) }}/draw</small>
      </article>
    </div>

    <section class="strategy-ranking-panel hit-distribution-panel">
      <header>
        <div>
          <h2>Draws by exact match count</h2>
          <p>
            Number of evaluated draws where each strategy’s prior Top‑6
            prediction matched exactly 6, 5, 4, 3, 2, or 1 winning numbers.
            The first row is the sum of every strategy row below it.
          </p>
        </div>
      </header>
      <div class="prediction-summary-table-wrap">
        <table class="hit-distribution-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th v-for="hitCount in exactHitLevels" :key="hitCount">
                Exactly {{ hitCount }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr class="all-strategies-summary">
              <th scope="row">
                <div class="hit-distribution-total-label">
                  <strong>All strategies</strong>
                  <small>Sum of {{ strategyRows.length }} strategy rows</small>
                </div>
              </th>
              <td
                v-for="bucket in allStrategiesExactHitDraws"
                :key="bucket.hitCount"
                :class="{ 'has-exact-hits': bucket.draws > 0 }"
                :title="`All strategies: ${bucket.draws} total strategy-draw results with exactly ${bucket.hitCount} hits`"
              >
                {{ bucket.draws.toLocaleString() }}
              </td>
            </tr>
            <tr v-for="row in strategyRows" :key="row.id">
              <th scope="row">
                <i
                  class="strategy-rank-color"
                  :style="{ '--strategy-line-color': row.color }"
                />
                {{ row.name }}
              </th>
              <td
                v-for="bucket in row.exactHitDraws"
                :key="bucket.hitCount"
                :class="{ 'has-exact-hits': bucket.draws > 0 }"
                :title="`${row.name}: ${bucket.draws} draws with exactly ${bucket.hitCount} hits`"
              >
                {{ bucket.draws.toLocaleString() }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="strategy-timeline-panel">
      <header>
        <div>
          <h2>
            {{
              effectivenessMode === "cumulative"
                ? "Cumulative average hits per draw"
                : `${rollingWindow}-draw rolling average hits`
            }}
          </h2>
          <p>Dashed reference line: uniform random Top‑6 expectation.</p>
        </div>
        <div class="strategy-visibility-actions">
          <button type="button" @click="showTopStrategies">Top 6</button>
          <button type="button" @click="showAllStrategies">Show all</button>
          <button type="button" @click="clearStrategies">Clear</button>
        </div>
      </header>

      <div class="strategy-timeline-layout">
        <div class="strategy-timeline-chart">
          <svg
            v-if="scopedRecords.length > 0"
            :viewBox="`0 0 ${chartWidth} ${chartHeight}`"
            role="img"
            aria-label="Strategy effectiveness line chart over draw history"
          >
            <g v-for="tick in yTicks" :key="tick.value">
              <line
                :x1="chartLeft"
                :x2="chartWidth - chartRight"
                :y1="tick.y"
                :y2="tick.y"
                class="effectiveness-grid-line"
              />
              <text
                :x="chartLeft - 12"
                :y="tick.y + 4"
                class="effectiveness-y-label"
              >{{ tick.value.toFixed(2) }}</text>
            </g>
            <g v-for="tick in xTicks" :key="tick.draw">
              <line
                :x1="xForIndex(tick.index)"
                :x2="xForIndex(tick.index)"
                :y1="chartTop"
                :y2="chartHeight - chartBottom"
                class="effectiveness-x-line"
              />
              <text
                :x="xForIndex(tick.index)"
                :y="chartHeight - 26"
                class="effectiveness-x-label"
              >{{ tick.draw }}</text>
            </g>
            <line
              :x1="chartLeft"
              :x2="chartWidth - chartRight"
              :y1="yForValue(randomExpectedHits)"
              :y2="yForValue(randomExpectedHits)"
              class="random-expectation-line"
            />
            <text
              :x="chartWidth - chartRight - 4"
              :y="yForValue(randomExpectedHits) - 7"
              class="random-expectation-label"
            >Random {{ randomExpectedHits.toFixed(3) }}</text>
            <path
              v-for="series in lineSeries"
              :key="series.id"
              :d="seriesPath(series.values)"
              :stroke="series.color"
              class="effectiveness-series"
            >
              <title>
                {{ series.name }} · {{ series.average.toFixed(3) }} hits/draw
              </title>
            </path>
            <text
              :x="chartWidth / 2"
              :y="chartHeight - 5"
              class="effectiveness-axis-title"
            >Target draw number →</text>
          </svg>
          <p v-else class="prediction-analysis-empty">
            No completed walk-forward predictions are available.
          </p>
        </div>

        <div class="strategy-timeline-legend" role="group" aria-label="Visible strategies">
          <label
            v-for="row in strategyRows"
            :key="row.id"
            :class="{ selected: visibleStrategyIds.has(row.id) }"
          >
            <input
              type="checkbox"
              :checked="visibleStrategyIds.has(row.id)"
              @change="
                setStrategyVisible(
                  row.id,
                  ($event.target as HTMLInputElement).checked,
                )
              "
            >
            <i :style="{ '--strategy-line-color': row.color }" />
            <span>
              <strong>{{ row.name }}</strong>
              <small>{{ row.average.toFixed(3) }} · {{ signed(row.lift) }} lift</small>
            </span>
          </label>
        </div>
      </div>
    </section>

    <section class="strategy-ranking-panel">
      <header>
        <div>
          <h2>Effectiveness ranking for selected history</h2>
          <p>Total correct numbers and average lift over random expectation.</p>
        </div>
      </header>
      <div class="prediction-summary-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Strategy</th>
              <th>Correct numbers</th>
              <th>Hits/draw</th>
              <th>Lift vs random</th>
              <th>Relative performance</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, index) in strategyRows" :key="row.id">
              <td>#{{ index + 1 }}</td>
              <th scope="row">
                <i
                  class="strategy-rank-color"
                  :style="{ '--strategy-line-color': row.color }"
                />
                {{ row.name }}
              </th>
              <td>{{ row.hits }}</td>
              <td>{{ row.average.toFixed(3) }}</td>
              <td :class="row.lift >= 0 ? 'positive-lift' : 'negative-lift'">
                {{ signed(row.lift) }}
              </td>
              <td>
                <div class="effectiveness-meter">
                  <i
                    :style="{
                      width: `${Math.min(100, (row.average / Math.max(bestStrategy?.average ?? 1, 0.001)) * 100)}%`,
                      background: row.color,
                    }"
                  />
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </section>
</template>
