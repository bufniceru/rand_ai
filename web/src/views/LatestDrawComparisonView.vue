<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { strategyColor } from "../lib/strategyColors";
import type {
  AnalysisPayload,
  DrawComparisonStrategy,
} from "../types";

const props = defineProps<{ analysis: AnalysisPayload }>();

const selectedTargetDrawNumber = ref<number | null>(null);
const pdfSaving = ref(false);
const printing = ref(false);
const reportActionMessage = ref("");
const comparisons = computed(() =>
  props.analysis.drawComparisonHistory.length
    ? props.analysis.drawComparisonHistory
    : props.analysis.latestDrawComparison
      ? [props.analysis.latestDrawComparison]
      : [],
);
const comparisonOptions = computed(() => [...comparisons.value].reverse());
const comparison = computed(
  () =>
    comparisons.value.find(
      (item) => item.targetDrawNumber === selectedTargetDrawNumber.value,
    ) ??
    comparisons.value.at(-1) ??
    null,
);
const actualSet = computed(
  () => new Set(comparison.value?.actualNumbers ?? []),
);
const sortedStrategies = computed(() =>
  [...(comparison.value?.strategies ?? [])].sort(
    (left, right) =>
      right.hitCount - left.hitCount ||
      (right.efficacy?.averageHitsPerDraw ?? 0) -
        (left.efficacy?.averageHitsPerDraw ?? 0) ||
      strategyFullName(left).localeCompare(strategyFullName(right)),
  ),
);
const bestHitCount = computed(
  () => sortedStrategies.value[0]?.hitCount ?? 0,
);
const strategiesWithHits = computed(
  () => sortedStrategies.value.filter((strategy) => strategy.hitCount > 0).length,
);
const totalStrategyHits = computed(() =>
  sortedStrategies.value.reduce(
    (total, strategy) => total + strategy.hitCount,
    0,
  ),
);
const averageHits = computed(() =>
  sortedStrategies.value.length
    ? totalStrategyHits.value / sortedStrategies.value.length
    : 0,
);
const actualCoverage = computed(() =>
  (comparison.value?.actualNumbers ?? []).map((number) => {
    const strategies = sortedStrategies.value.filter((strategy) =>
      strategy.predictedNumbers.includes(number),
    );
    return { number, strategies };
  }),
);
const coveredActualCount = computed(
  () => actualCoverage.value.filter((item) => item.strategies.length > 0).length,
);
const matchDistribution = computed(() =>
  Array.from({ length: 7 }, (_value, index) => 6 - index).map((hitCount) => ({
    hitCount,
    strategyCount: sortedStrategies.value.filter(
      (strategy) => strategy.hitCount === hitCount,
    ).length,
  })),
);

watch(
  comparisons,
  (items) => {
    if (
      items.some(
        (item) => item.targetDrawNumber === selectedTargetDrawNumber.value,
      )
    ) {
      return;
    }
    selectedTargetDrawNumber.value = items.at(-1)?.targetDrawNumber ?? null;
  },
  { immediate: true },
);

watch(selectedTargetDrawNumber, () => {
  reportActionMessage.value = "";
});

function strategyFullName(strategy: DrawComparisonStrategy): string {
  return {
    proximity: "Proximity",
    freshness: "Freshness",
    emd: "Earth Mover Distance",
    randomness: "Random Baseline",
    fresh_random: "Fresh Random",
    chi_square: "Chi-square Frequency",
    categorical_chi_square: "Categorical Chi-square",
    entropy: "Entropy",
    markov100: "Markov 100",
    mkgsv: "Markov Gap-Space Vector (Experimental)",
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
    decision_tree_selector: "Decision Tree Selector",
    border_group_statistical: "Border Group Statistical",
    border_group_markov: "Border Group Markov",
    border_group_bayesian: "Border Group Bayesian",
    border_group_ml: "Border Group ML",
    border_group_hybrid: "Border Group Hybrid",
    residual_coverage: "Residual Coverage",
    chained: "Chained Strategy",
  }[strategy.id] ?? strategy.name;
}

function formattedDate(date: string | null): string {
  if (!date) return "Date unavailable";
  const parsed = new Date(`${date}T00:00:00`);
  return Number.isNaN(parsed.getTime())
    ? date
    : new Intl.DateTimeFormat(undefined, {
        year: "numeric",
        month: "long",
        day: "numeric",
      }).format(parsed);
}

function efficacyLabel(strategy: DrawComparisonStrategy): string {
  const efficacy = strategy.efficacy;
  if (!efficacy || efficacy.evaluatedDraws === 0) return "Pending";
  return `${efficacy.averageHitsPerDraw.toFixed(2)} hits/draw`;
}

function reportOptionLabel(
  item: NonNullable<AnalysisPayload["latestDrawComparison"]>,
): string {
  return item.date
    ? `${formattedDate(item.date)} - Draw ${item.targetDrawNumber}`
    : `Draw ${item.targetDrawNumber} - Date unavailable`;
}

function suggestedPdfName(): string {
  const report = comparison.value;
  if (!report) return "rand-ai-draw-comparison.pdf";
  const identifier = report.date ?? `draw-${report.targetDrawNumber}`;
  return `rand-ai-draw-comparison-${identifier}.pdf`;
}

async function savePdf(): Promise<void> {
  if (!comparison.value || !window.randAiDesktop) {
    reportActionMessage.value = "PDF saving is available inside the desktop app.";
    return;
  }
  pdfSaving.value = true;
  reportActionMessage.value = "";
  try {
    await nextTick();
    const result = await window.randAiDesktop.saveDrawComparisonPdf({
      suggestedName: suggestedPdfName(),
    });
    reportActionMessage.value = result.canceled
      ? "PDF saving canceled."
      : `PDF saved to ${result.path ?? "the selected location"}.`;
  } catch (error) {
    reportActionMessage.value =
      error instanceof Error ? error.message : String(error);
  } finally {
    pdfSaving.value = false;
  }
}

async function printReport(): Promise<void> {
  if (!comparison.value) return;
  if (!window.randAiDesktop) {
    window.print();
    return;
  }
  printing.value = true;
  reportActionMessage.value = "";
  try {
    await nextTick();
    await window.randAiDesktop.printDrawComparison();
    reportActionMessage.value = "The report was sent to the print dialog.";
  } catch (error) {
    reportActionMessage.value =
      error instanceof Error ? error.message : String(error);
  } finally {
    printing.value = false;
  }
}
</script>

<template>
  <section class="workspace-view latest-draw-comparison-view">
    <template v-if="comparison">
      <section class="latest-draw-report-controls">
        <label>
          <span>Report date</span>
          <select v-model.number="selectedTargetDrawNumber">
            <option
              v-for="item in comparisonOptions"
              :key="item.targetDrawNumber"
              :value="item.targetDrawNumber"
            >
              {{ reportOptionLabel(item) }}
            </option>
          </select>
        </label>
        <div>
          <button
            type="button"
            :disabled="pdfSaving || printing"
            @click="savePdf"
          >
            {{ pdfSaving ? "Generating PDF…" : "Save PDF" }}
          </button>
          <button
            type="button"
            :disabled="pdfSaving || printing"
            @click="printReport"
          >
            {{ printing ? "Opening print dialog…" : "Print report" }}
          </button>
        </div>
        <p v-if="reportActionMessage" role="status">
          {{ reportActionMessage }}
        </p>
      </section>

      <header class="view-heading latest-draw-heading">
        <div>
          <p class="eyebrow">Prior forecast versus completed result</p>
          <h1>Latest draw vs predictions</h1>
          <p>
            Target draw {{ comparison.targetDrawNumber }} was predicted from draw
            {{ comparison.referenceDrawNumber }}. Results below use only the
            prediction saved before the target draw was known.
          </p>
        </div>
        <div class="latest-draw-date">
          <span>Draw date</span>
          <strong>{{ formattedDate(comparison.date) }}</strong>
        </div>
      </header>

      <section class="latest-actual-draw" aria-label="Actual latest draw">
        <div>
          <span>Actual numbers</span>
          <small>Draw {{ comparison.targetDrawNumber }}</small>
        </div>
        <ol>
          <li v-for="number in comparison.actualNumbers" :key="number">
            {{ number }}
          </li>
        </ol>
      </section>

      <div class="latest-comparison-facts">
        <article>
          <span>Best strategy result</span>
          <strong>{{ bestHitCount }}/6</strong>
          <small>exact matches</small>
        </article>
        <article>
          <span>Strategies with a hit</span>
          <strong>{{ strategiesWithHits }}/{{ sortedStrategies.length }}</strong>
          <small>active strategies</small>
        </article>
        <article>
          <span>All-strategy coverage</span>
          <strong>{{ coveredActualCount }}/6</strong>
          <small>actual numbers covered</small>
        </article>
        <article>
          <span>Mean result</span>
          <strong>{{ averageHits.toFixed(2) }}</strong>
          <small>hits per strategy</small>
        </article>
      </div>

      <section class="latest-match-distribution">
        <header>
          <div>
            <h2>Strategies by exact match count</h2>
            <p>All {{ sortedStrategies.length }} active strategies are included.</p>
          </div>
        </header>
        <div>
          <article
            v-for="row in matchDistribution"
            :key="row.hitCount"
            :class="{ populated: row.strategyCount > 0 }"
          >
            <strong>{{ row.hitCount }}</strong>
            <span>{{ row.hitCount === 1 ? "match" : "matches" }}</span>
            <b>{{ row.strategyCount }}</b>
            <small>{{ row.strategyCount === 1 ? "strategy" : "strategies" }}</small>
          </article>
        </div>
      </section>

      <section class="latest-actual-coverage">
        <header>
          <div>
            <h2>Actual-number coverage</h2>
            <p>How many prior strategy Top 6 lists contained each drawn number.</p>
          </div>
        </header>
        <div>
          <article v-for="item in actualCoverage" :key="item.number">
            <strong>{{ item.number }}</strong>
            <span>{{ item.strategies.length }}/{{ sortedStrategies.length }}</span>
            <small
              :title="item.strategies.map(strategyFullName).join(', ') || 'No strategy'"
            >
              {{
                item.strategies.length
                  ? item.strategies.map(strategyFullName).join(" · ")
                  : "Not predicted"
              }}
            </small>
          </article>
        </div>
      </section>

      <section class="latest-strategy-results">
        <header>
          <div>
            <h2>Strategy-by-strategy result</h2>
            <p>Ordered by today’s exact matches, then walk-forward effectiveness.</p>
          </div>
        </header>
        <div class="latest-strategy-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Predicted six</th>
                <th>Result</th>
                <th>Matched</th>
                <th>Actual numbers missed</th>
                <th>Historical mean</th>
              </tr>
            </thead>
            <tbody>
              <tr class="all-strategies-row">
                <th scope="row">
                  <span>All strategies</span>
                  <small>{{ sortedStrategies.length }} taken into account</small>
                </th>
                <td>
                  <span class="all-strategy-summary">
                    {{ totalStrategyHits }} total correct implications
                  </span>
                </td>
                <td>
                  <strong>{{ coveredActualCount }}/6 covered</strong>
                </td>
                <td>
                  <div class="comparison-ball-list compact">
                    <span
                      v-for="item in actualCoverage.filter(row => row.strategies.length)"
                      :key="item.number"
                      class="hit"
                      :title="`${item.strategies.length} strategies`"
                    >
                      {{ item.number }}
                    </span>
                  </div>
                </td>
                <td>
                  <div class="comparison-ball-list compact">
                    <span
                      v-for="item in actualCoverage.filter(row => !row.strategies.length)"
                      :key="item.number"
                      class="miss"
                    >
                      {{ item.number }}
                    </span>
                    <small v-if="coveredActualCount === 6">None</small>
                  </div>
                </td>
                <td>{{ averageHits.toFixed(2) }} hits/draw</td>
              </tr>
              <tr
                v-for="strategy in sortedStrategies"
                :key="strategy.id"
                :style="{ '--strategy-color': strategyColor(strategy.id) }"
              >
                <th scope="row">
                  <span>{{ strategyFullName(strategy) }}</span>
                  <small>{{ strategy.description }}</small>
                </th>
                <td>
                  <div class="comparison-ball-list">
                    <span
                      v-for="number in strategy.predictedNumbers"
                      :key="number"
                      :class="{ hit: actualSet.has(number), miss: !actualSet.has(number) }"
                    >
                      {{ number }}
                    </span>
                  </div>
                </td>
                <td>
                  <strong
                    class="strategy-hit-count"
                    :class="{
                      strong: strategy.hitCount >= 3,
                      positive: strategy.hitCount > 0,
                    }"
                  >
                    {{ strategy.hitCount }}/6
                  </strong>
                </td>
                <td>
                  <div class="comparison-ball-list compact">
                    <span
                      v-for="number in strategy.matchedNumbers"
                      :key="number"
                      class="hit"
                    >
                      {{ number }}
                    </span>
                    <small v-if="strategy.matchedNumbers.length === 0">None</small>
                  </div>
                </td>
                <td>
                  <div class="comparison-ball-list compact">
                    <span
                      v-for="number in strategy.missedActualNumbers"
                      :key="number"
                      class="miss"
                    >
                      {{ number }}
                    </span>
                    <small v-if="strategy.missedActualNumbers.length === 0">None</small>
                  </div>
                </td>
                <td>{{ efficacyLabel(strategy) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </template>

    <section v-else class="prediction-analysis-empty latest-comparison-empty">
      <strong>No completed prediction is available yet.</strong>
      <p>
        Add today’s official draw to Draw History and analyze the dataset again.
        At least two draws and one enabled strategy are required.
      </p>
    </section>
  </section>
</template>
