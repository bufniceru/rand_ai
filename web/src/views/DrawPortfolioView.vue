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
} from "../lib/drawPortfolioBacktest";
import type {
  PortfolioBacktestProgress,
  PortfolioBacktestResult,
  PredictionSuite,
  RelationshipEdge,
} from "../types";

const props = defineProps<{
  predictionSuites: PredictionSuite[];
  relationshipEdges: RelationshipEdge[];
}>();

const requestedDrawCount = ref(10);
const result = ref<DrawPortfolioResult | null>(null);
const generating = ref(false);
const copyMessage = ref("");
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
  generating.value = true;
  copyMessage.value = "";
  generationMessage.value = "";
  await nextTick();
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  try {
    result.value = generateDrawPortfolio(
      suite,
      props.relationshipEdges,
      normalizeDrawCount(),
    );
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
  if (!result.value?.draws.length) return;
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

watch(
  () => [latestSuite.value?.referenceDrawNumber, latestSuite.value?.strategies] as const,
  () => {
    result.value = null;
    copyMessage.value = "";
    generationMessage.value = "";
    if (simulationRunning.value) cancelSimulation();
    simulationResult.value = null;
    simulationFromCache.value = false;
  },
  { deep: true },
);

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
        <label>
          <span>Number of draws</span>
          <input
            v-model.number="requestedDrawCount"
            type="number"
            min="1"
            max="100"
            :disabled="generating"
            @change="normalizeDrawCount"
          >
          <small>From 1 to 100 · default 10</small>
        </label>
        <button type="button" :disabled="generating" @click="generate">
          {{ generating ? "Generating…" : "Generate portfolio" }}
        </button>
        <p>
          Randomness and Fresh Random are baseline strategies and do not contribute
          to portfolio scoring.
        </p>
      </div>

      <p v-if="generationMessage" class="draw-portfolio-message error">
        {{ generationMessage }}
      </p>

      <template v-if="result">
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
              :title="`#${index + 1} · ${percent(entry.score)} ensemble strength · Top 6 in ${entry.topSixSupport} strategies`"
            >
              <span>#{{ index + 1 }}</span>
              <strong>{{ entry.number }}</strong>
              <small>{{ percent(entry.score) }}</small>
              <small>{{ entry.topSixSupport }}× Top 6</small>
            </li>
          </ol>
        </section>

        <section class="draw-portfolio-results" aria-labelledby="portfolio-results-title">
          <header>
            <div>
              <h2 id="portfolio-results-title">Generated draws</h2>
              <p>Sorted numbers · model-relative score · overlap with another draw</p>
            </div>
            <div>
              <button type="button" @click="copyAll">Copy All</button>
              <span role="status">{{ copyMessage }}</span>
            </div>
          </header>
          <ol>
            <li v-for="(draw, index) in result.draws" :key="draw.numbers.join('-')">
              <span class="draw-portfolio-index">Draw {{ index + 1 }}</span>
              <div class="draw-portfolio-balls">
                <strong v-for="number in draw.numbers" :key="number">{{ number }}</strong>
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

      <section v-else-if="!generating" class="draw-portfolio-empty">
        <strong>Choose how many draws to generate</strong>
        <p>The adaptive pool and portfolio metrics will appear here.</p>
      </section>
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
            Rebuild a portfolio before every known target draw and record the best
            single-ticket result. Historical weights and relationships use only
            information available at that reference.
          </p>
        </div>
        <span v-if="simulationFromCache" class="portfolio-cache-badge">
          Cached result
        </span>
      </header>

      <div class="portfolio-backtest-controls">
        <label>
          <span>Tickets per historical target</span>
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
