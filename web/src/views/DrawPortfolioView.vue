<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";

import {
  generateDrawPortfolio,
  type DrawPortfolioResult,
} from "../lib/drawPortfolio";
import {
  portfolioBacktestCacheKey,
  portfolioHistoryStatistics,
} from "../lib/drawPortfolioBacktest";
import {
  activePossibleDrawState,
  getDrawPortfolioMode,
  possibleDrawPlanRevision,
  setDrawPortfolioMode,
} from "../lib/possibleDrawPlans";
import type {
  DrawPortfolioMode,
  PortfolioBacktestProgress,
  PortfolioBacktestResult,
  PredictionSuite,
  RelationshipEdge,
} from "../types";

const props = defineProps<{
  datasetId: string;
  predictionSuites: PredictionSuite[];
  relationshipEdges: RelationshipEdge[];
}>();

const requestedDrawCount = ref(10);
const result = ref<DrawPortfolioResult | null>(null);
const portfolioMode = ref<DrawPortfolioMode>(getDrawPortfolioMode(props.datasetId));
const resultStale = ref(false);
const staleReason = ref("");
const generating = ref(false);
const copyMessage = ref("");
const pdfSaving = ref(false);
const pdfMessage = ref("");
const generationMessage = ref("");
const simulationPortfolioSize = ref(10);
const simulationResult = ref<PortfolioBacktestResult | null>(null);
const simulationRunning = ref(false);
const simulationProgress = ref<PortfolioBacktestProgress>({
  percent: 0,
  message: "Ready to simulate the complete history",
});
const simulationMessage = ref("");
const simulationFromCache = ref(false);
const simulationElapsedMs = ref(0);
const hitFilter = ref("all");
const auditPage = ref(1);
const auditPageSize = 100;
let simulationWorker: Worker | null = null;
let simulationTimer: number | null = null;
let simulationStartedAt = 0;
let simulationToken = 0;
let unsubscribeSimulationProgress: (() => void) | null = null;

const latestSuite = computed(() => props.predictionSuites.at(-1) ?? null);
const predictiveStrategyCount = computed(
  () =>
    latestSuite.value?.strategies.filter(
      (strategy) => !["randomness", "fresh_random"].includes(strategy.id),
    ).length ?? 0,
);
const filteredAudit = computed(() => {
  const rows = simulationResult.value?.audit ?? [];
  if (hitFilter.value === "all") return rows;
  const hits = Number(hitFilter.value);
  return rows.filter((row) => row.bestHits === hits);
});
const auditPageCount = computed(() =>
  Math.max(1, Math.ceil(filteredAudit.value.length / auditPageSize)),
);
const visibleAudit = computed(() => {
  const start = (auditPage.value - 1) * auditPageSize;
  return filteredAudit.value.slice(start, start + auditPageSize);
});
const simulationStatistics = computed(() =>
  portfolioHistoryStatistics(simulationResult.value),
);
const planState = activePossibleDrawState;

function markResultStale(reason: string): void {
  if (!result.value) return;
  resultStale.value = true;
  staleReason.value = reason;
  copyMessage.value = "";
  pdfMessage.value = "";
}

function setMode(mode: DrawPortfolioMode): void {
  if (portfolioMode.value === mode) return;
  portfolioMode.value = mode;
  setDrawPortfolioMode(props.datasetId, mode);
  markResultStale(`Mode changed to ${mode === "guided" ? "Guided" : "Classic"}. Regenerate to apply it.`);
}

function normalizeDrawCount(): number {
  const value = Number.isFinite(requestedDrawCount.value)
    ? Math.trunc(requestedDrawCount.value)
    : 10;
  requestedDrawCount.value = Math.min(Math.max(value, 1), 100);
  return requestedDrawCount.value;
}

async function generate(): Promise<void> {
  const suite = latestSuite.value;
  if (!suite || predictiveStrategyCount.value === 0) return;
  const drawCount = normalizeDrawCount();
  generating.value = true;
  copyMessage.value = "";
  pdfMessage.value = "";
  generationMessage.value = "";
  await nextTick();
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  try {
    result.value = generateDrawPortfolio(
      suite,
      props.relationshipEdges,
      drawCount,
      {
        mode: portfolioMode.value,
        fixedNumbers: planState.value.fixedNumbers,
        candidateNumbers: planState.value.candidateNumbers,
        excludedNumbers: planState.value.excludedNumbers,
      },
    );
    resultStale.value = false;
    staleReason.value = "";
    if (!result.value) {
      generationMessage.value = "No predictive strategies are available.";
    }
  } catch (error) {
    result.value = null;
    generationMessage.value =
      error instanceof Error ? error.message : "Portfolio generation failed.";
  } finally {
    generating.value = false;
  }
}

async function copyAll(): Promise<void> {
  if (!result.value?.draws.length || resultStale.value) return;
  const text = result.value.draws
    .map((draw) => draw.numbers.join(","))
    .join("\n");
  try {
    await navigator.clipboard.writeText(text);
    copyMessage.value = `${result.value.draws.length} draws copied.`;
  } catch {
    copyMessage.value = "Clipboard access failed. Select and copy the draws manually.";
  }
}

function suggestedPdfName(): string {
  const suite = latestSuite.value;
  if (!suite) return "rand-ai-draw-portfolio.pdf";
  return `rand-ai-draw-portfolio-target-${suite.targetDrawNumber}.pdf`;
}

async function savePdf(): Promise<void> {
  if (!result.value || resultStale.value) return;
  const api = window.randAiDesktop;
  if (!api) {
    pdfMessage.value = "PDF saving is available inside the desktop app.";
    return;
  }

  pdfSaving.value = true;
  pdfMessage.value = "";
  document.documentElement.classList.add("draw-portfolio-pdf-export");
  try {
    await nextTick();
    const exportResult = await api.saveDrawPortfolioPdf({
      suggestedName: suggestedPdfName(),
    });
    pdfMessage.value = exportResult.canceled
      ? "PDF saving canceled."
      : `PDF saved to ${exportResult.path ?? "the selected location"}.`;
  } catch (error) {
    pdfMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    document.documentElement.classList.remove("draw-portfolio-pdf-export");
    pdfSaving.value = false;
  }
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function normalizeSimulationPortfolioSize(): number {
  const value = Number.isFinite(simulationPortfolioSize.value)
    ? Math.trunc(simulationPortfolioSize.value)
    : 10;
  simulationPortfolioSize.value = Math.min(Math.max(value, 1), 100);
  return simulationPortfolioSize.value;
}

function stopSimulationTimer(): void {
  if (simulationTimer !== null) {
    window.clearInterval(simulationTimer);
    simulationTimer = null;
  }
}

function finishSimulation(): void {
  simulationRunning.value = false;
  stopSimulationTimer();
  simulationWorker?.terminate();
  simulationWorker = null;
}

function cancelSimulation(): void {
  simulationToken += 1;
  finishSimulation();
  simulationProgress.value = {
    ...simulationProgress.value,
    message: "Simulation canceled",
  };
  simulationMessage.value = "The historical simulation was canceled.";
}

async function runHistoricalSimulation(): Promise<void> {
  const api = window.randAiDesktop;
  const suite = latestSuite.value;
  if (!api || !suite || simulationRunning.value) return;
  const portfolioSize = normalizeSimulationPortfolioSize();
  const token = ++simulationToken;
  simulationResult.value = null;
  simulationFromCache.value = false;
  simulationMessage.value = "";
  simulationRunning.value = true;
  simulationStartedAt = performance.now();
  simulationElapsedMs.value = 0;
  simulationProgress.value = {
    percent: 0,
    message: "Loading compact full-history prediction rankings",
  };
  stopSimulationTimer();
  simulationTimer = window.setInterval(() => {
    simulationElapsedMs.value = performance.now() - simulationStartedAt;
  }, 250);

  try {
    const data = await api.getPortfolioBacktestData({
      strategyIds: suite.strategies.map((strategy) => strategy.id),
    });
    if (token !== simulationToken || !simulationRunning.value) return;
    const cacheKey = portfolioBacktestCacheKey(data.cacheKey, portfolioSize);
    const cached = await api.loadPortfolioBacktest(cacheKey);
    if (token !== simulationToken || !simulationRunning.value) return;
    if (cached) {
      simulationResult.value = cached;
      simulationFromCache.value = true;
      simulationElapsedMs.value = cached.durationMs;
      simulationProgress.value = {
        percent: 100,
        processed: cached.evaluatedTargets,
        total: cached.evaluatedTargets,
        message: "Loaded completed simulation from cache",
      };
      finishSimulation();
      return;
    }

    simulationProgress.value = {
      percent: 0,
      processed: 0,
      total: data.records.length,
      message: "Starting leakage-free portfolio simulation",
    };
    simulationWorker = new Worker(
      new URL("../workers/drawPortfolioBacktest.worker.ts", import.meta.url),
      { type: "module" },
    );
    simulationWorker.onmessage = (event) => {
      if (token !== simulationToken) return;
      if (event.data?.type === "progress") {
        simulationProgress.value = event.data.progress as PortfolioBacktestProgress;
        return;
      }
      if (event.data?.type === "error") {
        simulationMessage.value = String(event.data.message ?? "Simulation failed.");
        finishSimulation();
        return;
      }
      if (event.data?.type === "result") {
        const completed = event.data.result as PortfolioBacktestResult;
        simulationResult.value = completed;
        simulationElapsedMs.value = completed.durationMs;
        simulationProgress.value = {
          percent: 100,
          processed: completed.evaluatedTargets,
          total: completed.evaluatedTargets,
          message: "Full-history simulation complete",
        };
        finishSimulation();
        void api.savePortfolioBacktest({ key: cacheKey, result: completed }).catch(
          (error) => {
            simulationMessage.value = `Simulation completed, but its cache could not be saved: ${error instanceof Error ? error.message : String(error)}`;
          },
        );
      }
    };
    simulationWorker.onerror = (event) => {
      if (token !== simulationToken) return;
      simulationMessage.value = event.message || "Historical simulation worker failed.";
      finishSimulation();
    };
    simulationWorker.postMessage({ data, portfolioSize });
  } catch (error) {
    if (token !== simulationToken) return;
    simulationMessage.value = error instanceof Error ? error.message : String(error);
    finishSimulation();
  }
}

function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.round(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function stepAuditPage(direction: -1 | 1): void {
  auditPage.value = Math.min(
    Math.max(auditPage.value + direction, 1),
    auditPageCount.value,
  );
}

function showMaximumHitTargets(): void {
  hitFilter.value = String(simulationStatistics.value.maximumHits);
}

function resultNumberState(number: number): "neutral" | "candidate" | "fixed" | "excluded" {
  if (result.value?.metadata.fixedNumbers.includes(number)) return "fixed";
  if (result.value?.metadata.candidateNumbers.includes(number)) return "candidate";
  if (result.value?.metadata.excludedNumbers.includes(number)) return "excluded";
  return "neutral";
}

watch(
  () => [props.datasetId, latestSuite.value?.referenceDrawNumber] as const,
  () => {
    result.value = null;
    resultStale.value = false;
    staleReason.value = "";
    portfolioMode.value = getDrawPortfolioMode(props.datasetId);
    copyMessage.value = "";
    pdfMessage.value = "";
    generationMessage.value = "";
    if (simulationRunning.value) cancelSimulation();
    simulationResult.value = null;
    simulationFromCache.value = false;
  },
);

watch(
  () => latestSuite.value?.strategies,
  () => markResultStale("Prediction strategies changed. Regenerate to refresh the portfolio."),
  { deep: true },
);

watch(
  () => props.relationshipEdges,
  () => markResultStale("Relationship inputs changed. Regenerate to refresh the portfolio."),
  { deep: true },
);

watch(requestedDrawCount, () => {
  markResultStale("The requested draw count changed. Regenerate to apply it.");
});

watch(possibleDrawPlanRevision, () => {
  if (result.value?.metadata.mode === "guided") {
    markResultStale("The active Possible Draw plan changed. Regenerate the Guided portfolio.");
  }
});

watch([simulationResult, hitFilter], () => {
  auditPage.value = 1;
});

onMounted(() => {
  unsubscribeSimulationProgress =
    window.randAiDesktop?.onPortfolioBacktestProgress((progress) => {
      if (simulationRunning.value && simulationWorker === null) {
        simulationProgress.value = progress;
      }
    }) ?? null;
});

onBeforeUnmount(() => {
  simulationToken += 1;
  finishSimulation();
  unsubscribeSimulationProgress?.();
});
</script>

<template>
  <section class="draw-portfolio-shell">
    <header class="draw-portfolio-header">
      <div>
        <p class="eyebrow">Prediction portfolio</p>
        <h1>Draw Portfolio</h1>
        <p>
          Generate deterministic six-number draws from efficacy-weighted model
          agreement, then diversify their numbers, pairs, and triples.
        </p>
      </div>
      <dl v-if="latestSuite" class="draw-portfolio-target">
        <div>
          <dt>Reference draw</dt>
          <dd>{{ latestSuite.referenceDrawNumber }}</dd>
        </div>
        <div>
          <dt>Target draw</dt>
          <dd>{{ latestSuite.targetDrawNumber }}</dd>
        </div>
        <div>
          <dt>Predictive strategies</dt>
          <dd>{{ predictiveStrategyCount }}</dd>
        </div>
      </dl>
    </header>

    <section v-if="latestSuite && predictiveStrategyCount > 0" class="draw-portfolio-workspace">
      <div class="draw-portfolio-controls">
        <div class="portfolio-mode-control">
          <div class="portfolio-mode-switch" role="group" aria-label="Draw Portfolio generation mode">
            <button
              type="button"
              :class="{ active: portfolioMode === 'classic' }"
              :aria-pressed="portfolioMode === 'classic'"
              @click="setMode('classic')"
            >Classic</button>
            <button
              type="button"
              :class="{ active: portfolioMode === 'guided' }"
              :aria-pressed="portfolioMode === 'guided'"
              @click="setMode('guided')"
            >Guided</button>
          </div>
        </div>
        <label>
          <input
            v-model.number="requestedDrawCount"
            type="number"
            min="1"
            max="100"
            aria-label="Number of draws"
            title="Number of draws"
            :disabled="generating"
            @change="normalizeDrawCount"
          >
        </label>
        <button type="button" :disabled="generating" @click="generate">
          {{ generating ? "Generating…" : "Generate portfolio" }}
        </button>
      </div>

      <p v-if="resultStale" class="draw-portfolio-message stale" role="status">
        <strong>Displayed portfolio is stale.</strong> {{ staleReason }}
        Copy and PDF export remain disabled until regeneration.
      </p>

      <p v-if="generationMessage" class="draw-portfolio-message error">
        {{ generationMessage }}
      </p>

      <template v-if="result">
        <div class="portfolio-result-mode-line">
          <span :class="['portfolio-mode-badge', result.metadata.mode]">
            {{ result.metadata.mode === "guided" ? "Guided" : "Classic" }} result
          </span>
          <span>
            {{ result.metadata.generatedDrawCount }}/{{ result.metadata.requestedDrawCount }} requested draws ·
            {{ result.metadata.availableUniqueCount.toLocaleString() }} eligible unique tickets
          </span>
        </div>
        <p
          v-if="result.metadata.constraintMessage"
          class="draw-portfolio-message constraint"
        >
          {{ result.metadata.constraintMessage }}
        </p>
        <section class="draw-portfolio-summary" aria-label="Portfolio summary">
          <article>
            <span>Generated draws</span>
            <strong>{{ result.draws.length }}</strong>
          </article>
          <article>
            <span>Average model score</span>
            <strong>{{ percent(result.metrics.averageModelScore) }}</strong>
          </article>
          <article>
            <span>Number coverage</span>
            <strong>{{ result.metrics.coveredNumbers }}/{{ result.pool.length }}</strong>
            <small>{{ percent(result.metrics.numberCoverage) }}</small>
          </article>
          <article>
            <span>Weighted pair coverage</span>
            <strong>{{ percent(result.metrics.pairCoverage) }}</strong>
          </article>
          <article>
            <span>Weighted triple coverage</span>
            <strong>{{ percent(result.metrics.tripleCoverage) }}</strong>
          </article>
          <article>
            <span>Maximum overlap</span>
            <strong>{{ result.metrics.maximumOverlap }}/6</strong>
          </article>
        </section>

        <section class="draw-portfolio-pool" aria-labelledby="portfolio-pool-title">
          <header>
            <div>
              <h2 id="portfolio-pool-title">Adaptive consensus pool</h2>
              <p>
                {{ result.pool.length }} candidates from
                {{ result.contributingStrategyCount }} predictive strategies
              </p>
            </div>
          </header>
          <ol>
            <li
              v-for="(entry, index) in result.pool"
              :key="entry.number"
              :class="entry.state ? `possible-${entry.state}` : ''"
              :title="`#${index + 1} · ${percent(entry.score)} ensemble strength · Top 6 in ${entry.topSixSupport} strategies${entry.state && entry.state !== 'neutral' ? ` · ${entry.state}` : ''}`"
            >
              <span>#{{ index + 1 }}</span>
              <strong>{{ entry.number }}</strong>
              <small>{{ percent(entry.score) }}</small>
              <small>{{ entry.topSixSupport }}× Top 6</small>
              <b v-if="entry.state === 'candidate'" class="portfolio-state-badge">C</b>
              <b v-if="entry.state === 'fixed'" class="portfolio-state-badge" aria-label="Fixed">🔒</b>
              <b v-if="entry.state === 'excluded'" class="portfolio-state-badge" aria-label="Excluded">×</b>
            </li>
          </ol>
        </section>

        <section class="draw-portfolio-results" aria-labelledby="portfolio-results-title">
          <header>
            <div>
              <h2 id="portfolio-results-title">Generated draws</h2>
              <p>Sorted numbers · model-relative score · overlap with another draw</p>
            </div>
            <div class="draw-portfolio-result-actions">
              <button type="button" :disabled="resultStale" @click="copyAll">Copy All</button>
              <button
                type="button"
                :disabled="pdfSaving || generating || resultStale"
                @click="savePdf"
              >
                {{ pdfSaving ? "Generating PDF…" : "Save PDF" }}
              </button>
              <span v-if="copyMessage" role="status">{{ copyMessage }}</span>
              <span v-if="pdfMessage" role="status">{{ pdfMessage }}</span>
            </div>
          </header>
          <ol>
            <li v-for="(draw, index) in result.draws" :key="draw.numbers.join('-')">
              <span class="draw-portfolio-index">Draw {{ index + 1 }}</span>
              <div class="draw-portfolio-balls">
                <strong
                  v-for="number in draw.numbers"
                  :key="number"
                  :class="`possible-${resultNumberState(number)}`"
                  :title="resultNumberState(number) === 'neutral' ? undefined : resultNumberState(number)"
                >
                  {{ number }}
                  <small v-if="resultNumberState(number) === 'candidate'">C</small>
                  <small v-if="resultNumberState(number) === 'fixed'" aria-label="Fixed">🔒</small>
                  <small v-if="resultNumberState(number) === 'excluded'" aria-label="Excluded">×</small>
                </strong>
              </div>
              <dl>
                <div>
                  <dt>Model score</dt>
                  <dd>{{ percent(draw.modelScore) }}</dd>
                </div>
                <div>
                  <dt>Max overlap</dt>
                  <dd>{{ draw.maximumOverlap }}/6</dd>
                </div>
              </dl>
            </li>
          </ol>
        </section>
      </template>

    </section>

    <section v-else class="draw-portfolio-empty unavailable">
      <strong>Draw Portfolio unavailable</strong>
      <p v-if="!latestSuite">
        Analyze a dataset with the Draw Portfolio report enabled.
      </p>
      <p v-else>
        Enable at least one predictive strategy. Randomness and Fresh Random are
        retained only as baselines.
      </p>
    </section>

    <section
      v-if="latestSuite && predictiveStrategyCount > 0"
      class="portfolio-backtest-panel"
      aria-labelledby="portfolio-backtest-title"
    >
      <header>
        <div>
          <p class="eyebrow">Leakage-free walk-forward audit</p>
          <h2 id="portfolio-backtest-title">Full-history simulation</h2>
          <p>
            For every known target, rebuild the requested number of portfolio draws,
            compare all of them with the actual result, and retain the highest number
            of correctly predicted values. Historical inputs stop at that reference.
            This audit always uses unconstrained Classic generation; today’s Possible
            Draw plan is never applied retroactively.
          </p>
        </div>
        <span v-if="simulationFromCache" class="portfolio-cache-badge">
          Cached result
        </span>
      </header>

      <div class="portfolio-backtest-controls">
        <label>
          <span>Number of draws per historical target</span>
          <input
            v-model.number="simulationPortfolioSize"
            type="number"
            min="1"
            max="100"
            :disabled="simulationRunning"
            @change="normalizeSimulationPortfolioSize"
          >
          <small>1–100 · default 10</small>
        </label>
        <button
          v-if="!simulationRunning"
          type="button"
          class="primary"
          @click="runHistoricalSimulation"
        >
          Run full-history simulation
        </button>
        <button v-else type="button" class="cancel" @click="cancelSimulation">
          Cancel
        </button>
        <div class="portfolio-backtest-status" aria-live="polite">
          <strong>{{ simulationProgress.message }}</strong>
          <span>
            <template v-if="simulationProgress.total !== undefined">
              {{ simulationProgress.processed ?? 0 }}/{{ simulationProgress.total }} targets ·
            </template>
            {{ simulationProgress.percent }}% · {{ formatDuration(simulationElapsedMs) }}
          </span>
        </div>
      </div>

      <div
        class="portfolio-backtest-progress"
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuenow="simulationProgress.percent"
      >
        <i :style="{ width: `${simulationProgress.percent}%` }" />
      </div>

      <p v-if="simulationMessage" class="draw-portfolio-message error">
        {{ simulationMessage }}
      </p>

      <template v-if="simulationResult">
        <section class="portfolio-backtest-facts">
          <article>
            <span>Evaluated targets</span>
            <strong>{{ simulationResult.evaluatedTargets.toLocaleString() }}</strong>
          </article>
          <article>
            <span>Tickets per target</span>
            <strong>{{ simulationResult.portfolioSize }}</strong>
          </article>
          <article>
            <span>Runtime</span>
            <strong>{{ formatDuration(simulationResult.durationMs) }}</strong>
          </article>
          <article>
            <span>At least one hit</span>
            <strong>{{ percent(simulationResult.buckets[1]?.atLeastRate ?? 0) }}</strong>
          </article>
        </section>

        <section
          class="portfolio-maximum-statistics"
          aria-labelledby="portfolio-maximum-title"
        >
          <header>
            <div>
              <p class="eyebrow">Historical efficacy</p>
              <h3 id="portfolio-maximum-title">Maximum predicted numbers</h3>
              <p>
                At each target, the best result among
                {{ simulationStatistics.portfolioSize }} generated draws is retained;
                the figures below summarize those maxima across the full history.
              </p>
            </div>
            <button type="button" @click="showMaximumHitTargets">
              Show maximum-hit targets
            </button>
          </header>
          <div>
            <article class="primary">
              <span>Historical maximum</span>
              <strong>{{ simulationStatistics.maximumHits }}/6</strong>
              <small>correctly predicted numbers in one generated draw</small>
            </article>
            <article>
              <span>Targets reaching maximum</span>
              <strong>{{ simulationStatistics.maximumHitTargets.toLocaleString() }}</strong>
              <small>of {{ simulationStatistics.evaluatedTargets.toLocaleString() }}</small>
            </article>
            <article>
              <span>Maximum-hit rate</span>
              <strong>{{ percent(simulationStatistics.maximumHitRate) }}</strong>
              <small>historical targets with the overall maximum</small>
            </article>
            <article>
              <span>Average best result</span>
              <strong>{{ simulationStatistics.averageBestHits.toFixed(3) }}/6</strong>
              <small>best generated draw per historical target</small>
            </article>
          </div>
        </section>

        <section class="portfolio-hit-distribution">
          <header>
            <div>
              <h3>Best-ticket hit distribution</h3>
              <p>Exact percentages total 100%; “at least” is cumulative.</p>
            </div>
          </header>
          <div class="portfolio-hit-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Hits</th>
                  <th>Exact targets</th>
                  <th>Exact percentage</th>
                  <th>At least</th>
                  <th>At-least percentage</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="bucket in simulationResult.buckets" :key="bucket.hits">
                  <th scope="row">{{ bucket.hits }}/6</th>
                  <td>{{ bucket.exactCount.toLocaleString() }}</td>
                  <td>
                    <div class="portfolio-hit-meter">
                      <i :style="{ width: percent(bucket.exactRate) }" />
                      <strong>{{ percent(bucket.exactRate) }}</strong>
                    </div>
                  </td>
                  <td>{{ bucket.atLeastCount.toLocaleString() }}</td>
                  <td>{{ percent(bucket.atLeastRate) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="portfolio-backtest-audit">
          <header>
            <div>
              <h3>Per-target audit</h3>
              <p>Actual draw versus the lexicographically first best ticket.</p>
            </div>
            <label>
              <span>Best hits</span>
              <select v-model="hitFilter">
                <option value="all">All results</option>
                <option v-for="hits in 7" :key="hits - 1" :value="String(hits - 1)">
                  {{ hits - 1 }}/6
                </option>
              </select>
            </label>
          </header>
          <div class="portfolio-audit-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Target</th>
                  <th>Date</th>
                  <th>Actual numbers</th>
                  <th>Best ticket</th>
                  <th>Hits</th>
                  <th>Tied tickets</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in visibleAudit" :key="row.targetDrawNumber">
                  <th scope="row">{{ row.targetDrawNumber }}</th>
                  <td>{{ row.date || "—" }}</td>
                  <td>
                    <span class="portfolio-audit-numbers actual">
                      <b v-for="number in row.actualNumbers" :key="number">{{ number }}</b>
                    </span>
                  </td>
                  <td>
                    <span class="portfolio-audit-numbers">
                      <b
                        v-for="number in row.bestTicket"
                        :key="number"
                        :class="{ matched: row.actualNumbers.includes(number) }"
                      >{{ number }}</b>
                    </span>
                  </td>
                  <td><strong>{{ row.bestHits }}/6</strong></td>
                  <td>{{ row.tiedBestCount }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <footer>
            <span>
              {{ filteredAudit.length.toLocaleString() }} results · page
              {{ auditPage }}/{{ auditPageCount }}
            </span>
            <div>
              <button type="button" :disabled="auditPage <= 1" @click="stepAuditPage(-1)">
                Previous
              </button>
              <button
                type="button"
                :disabled="auditPage >= auditPageCount"
                @click="stepAuditPage(1)"
              >
                Next
              </button>
            </div>
          </footer>
        </section>
      </template>
    </section>

    <footer class="draw-portfolio-disclaimer">
      Scores compare the enabled models and historical relationships; they are not
      calibrated lottery probabilities. Generated draws cannot guarantee an outcome
      or change the lottery’s physical odds.
    </footer>
  </section>
</template>
