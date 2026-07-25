<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import SettingsDialog from "./components/SettingsDialog.vue";
import TrustDialog from "./components/TrustDialog.vue";
import { buildFigures } from "./lib/figureBuilders";
import ExportView from "./views/ExportView.vue";
import GapsView from "./views/GapsView.vue";
import NumbersView from "./views/NumbersView.vue";
import OverviewView from "./views/OverviewView.vue";
import RandomnessView from "./views/RandomnessView.vue";
import RelationshipsView from "./views/RelationshipsView.vue";
import SpacesView from "./views/SpacesView.vue";
import type {
  AnalysisOptions,
  AnalysisPayload,
  AnalysisProgress,
  DatasetSelection,
  MenuAction,
  ReportId,
  ReportPluginState,
  StrategyId,
  StrategyPlugin,
  StrategyPluginState,
  ViewId,
} from "./types";

const views: { id: ViewId; label: string; shortLabel: string }[] = [
  { id: "overview", label: "Overview", shortLabel: "Overview" },
  { id: "numbers", label: "Numbers", shortLabel: "Numbers" },
  { id: "spaces", label: "Spaces", shortLabel: "Spaces" },
  { id: "relationships", label: "Relationships", shortLabel: "Relationships" },
  { id: "randomness", label: "Randomness", shortLabel: "Randomness" },
  { id: "gaps", label: "Gaps", shortLabel: "Gaps" },
  { id: "export", label: "Export", shortLabel: "Export" },
];

const activeView = ref<ViewId>("overview");
const enabledReportIds = ref<ReportId[]>([]);
const enabledStrategyIds = ref<StrategyId[]>([]);
const strategyPlugins = ref<StrategyPlugin[]>([]);
const settingsOpen = ref(false);
const savingSettings = ref(false);
const pendingDataset = ref<DatasetSelection | null>(null);
const activeDataset = ref<DatasetSelection | null>(null);
const analysis = ref<AnalysisPayload | null>(null);
const loading = ref(false);
const loadingDataset = ref<DatasetSelection | null>(null);
const loadingProgress = ref(0);
const loadingExplanation = ref("Preparing the analysis request");
const exporting = ref(false);
const errorMessage = ref("");
const exportMessage = ref("");
const statusMessage = ref("Waiting for a trusted dataset");
const options = reactive<AnalysisOptions>({
  selectedNumbers: [1, 2, 3, 4, 5, 6],
  trendBins: 100,
  correlationMethod: "pearson",
  enabledReports: [],
  enabledStrategies: [],
});
let unsubscribeMenu: (() => void) | null = null;
let unsubscribeAnalysisProgress: (() => void) | null = null;
let progressTimer: ReturnType<typeof setInterval> | null = null;
let loadingProgressTarget = 1;
let progressCompletionResolver: (() => void) | null = null;
let reportRefreshPending = false;

const figures = computed(() =>
  analysis.value ? buildFigures(analysis.value) : {},
);
const enabledViews = computed(() =>
  views.filter(
    (view) =>
      view.id === "export" ||
      enabledReportIds.value.includes(view.id as ReportId),
  ),
);
const activeViewLabel = computed(
  () => views.find((view) => view.id === activeView.value)?.label ?? "Overview",
);

function isReportEnabled(reportId: ReportId): boolean {
  return enabledReportIds.value.includes(reportId);
}

function ensureActiveView(): void {
  if (enabledViews.value.some((view) => view.id === activeView.value)) return;
  activeView.value = enabledViews.value[0]?.id ?? "export";
}

function acceptReportPluginState(state: ReportPluginState): void {
  enabledReportIds.value = [...state.enabledReports];
  options.enabledReports = [...state.enabledReports];
  ensureActiveView();
}

function acceptStrategyPluginState(state: StrategyPluginState): void {
  strategyPlugins.value = state.plugins.map((plugin) => ({ ...plugin }));
  enabledStrategyIds.value = [...state.enabledStrategies];
  options.enabledStrategies = [...state.enabledStrategies];
}

async function openSettings(): Promise<void> {
  if (!window.randAiDesktop) return;
  errorMessage.value = "";
  try {
    acceptStrategyPluginState(
      await window.randAiDesktop.getStrategyPlugins(),
    );
    settingsOpen.value = true;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function saveStrategySettings(
  strategyIds: StrategyId[],
): Promise<void> {
  if (!window.randAiDesktop) return;
  savingSettings.value = true;
  errorMessage.value = "";
  const changed =
    strategyIds.length !== enabledStrategyIds.value.length ||
    strategyIds.some(
      (strategyId, index) => strategyId !== enabledStrategyIds.value[index],
    );
  try {
    const state = await window.randAiDesktop.setStrategyPlugins(strategyIds);
    acceptStrategyPluginState(state);
    settingsOpen.value = false;
    if (changed && activeDataset.value) {
      if (loading.value) reportRefreshPending = true;
      else await analyzeDataset(activeDataset.value, normalizeOptions());
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    savingSettings.value = false;
  }
}

function normalizeOptions(): AnalysisOptions {
  let selectedNumbers = [...new Set(options.selectedNumbers)]
    .filter((number) => Number.isInteger(number) && number >= 1 && number <= 49)
    .sort((left, right) => left - right);
  if (selectedNumbers.length === 0 && isReportEnabled("numbers")) {
    throw new Error("Select at least one number for the trend chart.");
  }
  if (selectedNumbers.length === 0) selectedNumbers = [1];
  const maximumBins = Math.min(500, analysis.value?.dataset.drawCount ?? 500);
  return {
    selectedNumbers,
    trendBins: Math.min(Math.max(Math.trunc(options.trendBins), 1), maximumBins),
    correlationMethod: options.correlationMethod,
    enabledReports: [...enabledReportIds.value],
    enabledStrategies: [...enabledStrategyIds.value],
  };
}

async function chooseDataset(): Promise<void> {
  errorMessage.value = "";
  if (!window.randAiDesktop) {
    errorMessage.value = "Dataset access is available in the Electron desktop application.";
    return;
  }
  try {
    const dataset = await window.randAiDesktop.openDataset();
    if (dataset) pendingDataset.value = dataset;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

function stageDataset(dataset: DatasetSelection): void {
  pendingDataset.value = dataset;
  errorMessage.value = "";
}

async function analyzeDataset(
  dataset: DatasetSelection,
  requestedOptions: AnalysisOptions,
): Promise<void> {
  if (!window.randAiDesktop) {
    errorMessage.value = "The Python bridge is only available inside Electron.";
    return;
  }
  loading.value = true;
  loadingDataset.value = dataset;
  loadingProgress.value = 1;
  loadingProgressTarget = 1;
  loadingExplanation.value = "Preparing the trusted dataset for the Python engine";
  errorMessage.value = "";
  exportMessage.value = "";
  statusMessage.value = `${loadingExplanation.value} · 1%`;
  if (progressTimer !== null) clearInterval(progressTimer);
  progressTimer = setInterval(() => {
    if (loadingProgress.value >= loadingProgressTarget) return;
    loadingProgress.value += 1;
    statusMessage.value = `${loadingExplanation.value} · ${loadingProgress.value}%`;
    if (loadingProgress.value === 100 && progressCompletionResolver) {
      progressCompletionResolver();
      progressCompletionResolver = null;
    }
  }, 35);
  try {
    const payload = await window.randAiDesktop.analyzeDataset({
      path: dataset.path,
      options: requestedOptions,
    });
    activeDataset.value = dataset;
    analysis.value = payload;
    options.selectedNumbers = [...payload.options.selectedNumbers];
    options.trendBins = payload.options.trendBins;
    options.correlationMethod = payload.options.correlationMethod;
    ensureActiveView();
    pendingDataset.value = null;
    loadingProgressTarget = 100;
    loadingExplanation.value = "Analysis ready";
    await new Promise<void>((resolve) => {
      if (loadingProgress.value === 100) {
        resolve();
      } else {
        progressCompletionResolver = resolve;
      }
    });
    statusMessage.value = "Ready";
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
    statusMessage.value = "Analysis failed";
  } finally {
    if (progressTimer !== null) clearInterval(progressTimer);
    progressTimer = null;
    progressCompletionResolver = null;
    loading.value = false;
    loadingDataset.value = null;
    if (reportRefreshPending && activeDataset.value) {
      reportRefreshPending = false;
      void analyzeDataset(activeDataset.value, normalizeOptions());
    }
  }
}

async function confirmTrust(): Promise<void> {
  if (!pendingDataset.value) return;
  const dataset = pendingDataset.value;
  pendingDataset.value = null;
  try {
    await analyzeDataset(dataset, normalizeOptions());
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
    statusMessage.value = "Analysis failed";
  }
}

function handleAnalysisProgress(progress: AnalysisProgress): void {
  if (!loading.value) return;
  const percent = Math.min(Math.max(Math.trunc(progress.percent), 0), 100);
  loadingProgressTarget = Math.max(loadingProgressTarget, percent);
  loadingExplanation.value = progress.message;
  statusMessage.value = `${progress.message} · ${loadingProgress.value}%`;
}

async function applyOptions(): Promise<void> {
  if (!activeDataset.value) return;
  try {
    await analyzeDataset(activeDataset.value, normalizeOptions());
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function exportAnalysis(): Promise<void> {
  if (!window.randAiDesktop || !analysis.value) return;
  exporting.value = true;
  exportMessage.value = "";
  statusMessage.value = "Exporting analysis…";
  try {
    const result = await window.randAiDesktop.exportAnalysis({
      options: normalizeOptions(),
    });
    exportMessage.value = result.canceled
      ? "Export canceled."
      : `Saved analysis to ${result.path}.`;
    statusMessage.value = result.canceled ? "Ready" : "Export complete";
  } catch (error) {
    exportMessage.value = error instanceof Error ? error.message : String(error);
    statusMessage.value = "Export failed";
  } finally {
    exporting.value = false;
  }
}

async function openLastSeenDialog(): Promise<void> {
  if (!window.randAiDesktop || !analysis.value) return;
  try {
    await window.randAiDesktop.openLastSeenDialog();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function openLastSeenGapDialog(): Promise<void> {
  if (!window.randAiDesktop || !analysis.value) return;
  try {
    await window.randAiDesktop.openLastSeenGapDialog();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function openCombinedPredictionDialog(): Promise<void> {
  if (!window.randAiDesktop || !analysis.value) return;
  try {
    await window.randAiDesktop.openCombinedPredictionDialog();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function openPossibleDrawDialog(): Promise<void> {
  if (!window.randAiDesktop || !analysis.value) return;
  try {
    await window.randAiDesktop.openPossibleDrawDialog();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function openDrawEditorDialog(): Promise<void> {
  if (!window.randAiDesktop || !analysis.value) return;
  try {
    await window.randAiDesktop.openDrawEditorDialog();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

function handleMenuAction(message: MenuAction): void {
  if (message.action === "datasetSelected") {
    stageDataset(message.dataset);
    return;
  }
  if (message.action === "openView") {
    activeView.value = message.view;
    return;
  }
  if (message.action === "openSettings") {
    void openSettings();
    return;
  }
  if (message.action === "reportPluginsChanged") {
    acceptReportPluginState(message);
    if (activeDataset.value) {
      if (loading.value) reportRefreshPending = true;
      else void analyzeDataset(activeDataset.value, normalizeOptions());
    }
    return;
  }
  if (message.action === "strategyPluginsChanged") {
    acceptStrategyPluginState(message);
    if (activeDataset.value) {
      if (loading.value) reportRefreshPending = true;
      else void analyzeDataset(activeDataset.value, normalizeOptions());
    }
    return;
  }
  if (message.action === "export") void exportAnalysis();
}

onMounted(async () => {
  unsubscribeMenu = window.randAiDesktop?.onMenuAction(handleMenuAction) ?? null;
  unsubscribeAnalysisProgress =
    window.randAiDesktop?.onAnalysisProgress(handleAnalysisProgress) ?? null;
  if (window.randAiDesktop) {
    try {
      const [reportState, strategyState] = await Promise.all([
        window.randAiDesktop.getReportPlugins(),
        window.randAiDesktop.getStrategyPlugins(),
      ]);
      acceptReportPluginState(reportState);
      acceptStrategyPluginState(strategyState);
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : String(error);
    }
  }
});

onBeforeUnmount(() => {
  if (progressTimer !== null) clearInterval(progressTimer);
  unsubscribeMenu?.();
  unsubscribeAnalysisProgress?.();
});
</script>

<template>
  <div class="app-shell">
    <header class="app-toolbar">
      <div class="brand-lockup">
        <div class="brand-mark">RA</div>
        <div>
          <strong>Rand AI</strong>
          <span>Draw statistics desktop</span>
        </div>
      </div>
      <div class="toolbar-actions">
        <button class="button secondary" type="button" @click="chooseDataset">
          Open dataset
        </button>
        <button
          v-if="isReportEnabled('last-seen')"
          class="button secondary"
          :disabled="!analysis || loading"
          type="button"
          @click="applyOptions"
        >
          Reanalyze
        </button>
        <button
          v-if="isReportEnabled('last-seen-gap')"
          class="button secondary"
          :disabled="!analysis || loading"
          type="button"
          @click="openLastSeenDialog"
        >
          Last seen
        </button>
        <button
          v-if="isReportEnabled('predictions')"
          class="button secondary"
          :disabled="!analysis || loading"
          type="button"
          @click="openLastSeenGapDialog"
        >
          Last seen gaps
        </button>
        <button
          v-if="isReportEnabled('possible-draw')"
          class="button secondary"
          :disabled="!analysis || loading"
          type="button"
          @click="openCombinedPredictionDialog"
        >
          Predictions
        </button>
        <button
          class="button secondary"
          :disabled="!analysis || loading"
          type="button"
          @click="openPossibleDrawDialog"
        >
          Possible draw
        </button>
        <button
          class="button secondary"
          :disabled="!analysis || loading"
          type="button"
          @click="openDrawEditorDialog"
        >
          Draw history
        </button>
        <button
          class="button primary"
          :disabled="!analysis || exporting"
          type="button"
          @click="exportAnalysis"
        >
          {{ exporting ? "Exporting…" : "Export" }}
        </button>
      </div>
    </header>

    <nav v-if="analysis" class="workspace-tabs" aria-label="Statistics views">
      <button
        v-for="view in enabledViews"
        :key="view.id"
        :class="{ active: activeView === view.id }"
        type="button"
        @click="activeView = view.id"
      >
        {{ view.shortLabel }}
      </button>
    </nav>

    <div v-if="analysis" class="dashboard-layout">
      <aside class="control-panel">
        <div>
          <p class="eyebrow">Analysis controls</p>
          <h2>Filters</h2>
        </div>
        <label v-if="isReportEnabled('numbers')" class="field-group">
          <span>Trend bins</span>
          <input
            v-model.number="options.trendBins"
            :max="Math.min(500, analysis.dataset.drawCount)"
            min="1"
            type="number"
          >
        </label>
        <label v-if="isReportEnabled('relationships')" class="field-group">
          <span>Correlation method</span>
          <select v-model="options.correlationMethod">
            <option value="pearson">Pearson</option>
            <option value="spearman">Spearman</option>
          </select>
        </label>
        <fieldset v-if="isReportEnabled('numbers')" class="number-filter">
          <legend>Trend numbers</legend>
          <div>
            <label v-for="number in 49" :key="number">
              <input v-model="options.selectedNumbers" :value="number" type="checkbox">
              <span>{{ number }}</span>
            </label>
          </div>
        </fieldset>
        <button class="button primary apply-button" :disabled="loading" type="button" @click="applyOptions">
          Apply controls
        </button>
        <dl class="dataset-facts">
          <div><dt>Dataset</dt><dd :title="analysis.dataset.path">{{ analysis.dataset.name }}</dd></div>
          <div><dt>Draws</dt><dd>{{ analysis.dataset.drawCount.toLocaleString() }}</dd></div>
          <div><dt>Observations</dt><dd>{{ analysis.dataset.numberObservations.toLocaleString() }}</dd></div>
          <div><dt>Sample</dt><dd>{{ analysis.dataset.sampleSize.toLocaleString() }}</dd></div>
        </dl>
      </aside>

      <main class="workspace">
        <OverviewView
          v-if="activeView === 'overview' && analysis.options.enabledReports.includes('overview')"
          :analysis="analysis"
          :figures="figures"
        />
        <NumbersView
          v-else-if="activeView === 'numbers' && analysis.options.enabledReports.includes('numbers')"
          :analysis="analysis"
          :figures="figures"
        />
        <SpacesView
          v-else-if="activeView === 'spaces' && analysis.options.enabledReports.includes('spaces')"
          :analysis="analysis"
          :figures="figures"
        />
        <RelationshipsView
          v-else-if="activeView === 'relationships' && analysis.options.enabledReports.includes('relationships')"
          :analysis="analysis"
          :figures="figures"
        />
        <RandomnessView
          v-else-if="activeView === 'randomness' && analysis.options.enabledReports.includes('randomness')"
          :analysis="analysis"
          :figures="figures"
        />
        <GapsView
          v-else-if="activeView === 'gaps' && analysis.options.enabledReports.includes('gaps')"
          :analysis="analysis"
          :figures="figures"
        />
        <ExportView
          v-else
          :analysis="analysis"
          :export-message="exportMessage"
          :exporting="exporting"
          @export="exportAnalysis"
        />
      </main>
    </div>

    <main v-else class="welcome-screen">
      <section class="welcome-card">
        <p class="eyebrow">Vue 3 + Electron</p>
        <h1>Explore draw history without a browser dashboard.</h1>
        <p>
          Open a trusted Draws pickle to calculate exact frequency, space,
          relationship, randomness, and last-seen views with the existing Python engine.
        </p>
        <button class="button primary large" type="button" @click="chooseDataset">
          Open trusted dataset
        </button>
        <small>Pickle files can execute code. A trust confirmation appears before analysis.</small>
      </section>
      <section class="welcome-features">
        <article><strong>21</strong><span>Interactive statistics charts</span></article>
        <article><strong>49</strong><span>Number-level frequency positions</span></article>
        <article><strong>250</strong><span>Draws in the last-seen workspace</span></article>
      </section>
    </main>

    <div v-if="errorMessage" class="error-banner">
      <strong>Something went wrong</strong>
      <span>{{ errorMessage }}</span>
      <button type="button" @click="errorMessage = ''">Dismiss</button>
    </div>

    <div v-if="loading" class="loading-backdrop">
      <div class="loading-card">
        <div class="loading-card-heading">
          <span class="spinner" />
          <div>
            <strong>Loading {{ loadingDataset?.name ?? "dataset" }}</strong>
            <span>{{ loadingProgress }}%</span>
          </div>
        </div>
        <div
          class="analysis-progress-track"
          role="progressbar"
          aria-label="Dataset analysis progress"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="loadingProgress"
        >
          <span :style="{ width: `${loadingProgress}%` }" />
        </div>
        <p>{{ loadingExplanation }}</p>
      </div>
    </div>

    <TrustDialog
      v-if="pendingDataset"
      :dataset="pendingDataset"
      @cancel="pendingDataset = null"
      @confirm="confirmTrust"
    />

    <SettingsDialog
      v-if="settingsOpen"
      :plugins="strategyPlugins"
      :enabled-strategy-ids="enabledStrategyIds"
      :saving="savingSettings"
      @cancel="settingsOpen = false"
      @save="saveStrategySettings"
    />

    <footer class="status-bar">
      <span :class="{ ready: Boolean(analysis) && !loading }">{{ statusMessage }}</span>
      <span>Dataset: {{ loadingDataset?.name ?? analysis?.dataset.name ?? activeDataset?.name ?? "none" }}</span>
      <span>Draws: {{ analysis?.dataset.drawCount.toLocaleString() ?? "—" }}</span>
      <span>View: {{ analysis ? activeViewLabel : "Welcome" }}</span>
      <span>Python engine · Vue renderer · Electron shell</span>
    </footer>
  </div>
</template>
