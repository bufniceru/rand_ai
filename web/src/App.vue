<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import SettingsDialog from "./components/SettingsDialog.vue";
import TrustDialog from "./components/TrustDialog.vue";
import DrawEditorDialogApp from "./DrawEditorDialogApp.vue";
import { buildFigures } from "./lib/figureBuilders";
import PossibleDrawDialogApp from "./PossibleDrawDialogApp.vue";
import AutocorrelationView from "./views/AutocorrelationView.vue";
import CoOccurrenceView from "./views/CoOccurrenceView.vue";
import CombinedPredictionGridView from "./views/CombinedPredictionGridView.vue";
import DrawPortfolioView from "./views/DrawPortfolioView.vue";
import ExportView from "./views/ExportView.vue";
import GapsView from "./views/GapsView.vue";
import LastSeenGapHighlightView from "./views/LastSeenGapHighlightView.vue";
import LastSeenHighlightView from "./views/LastSeenHighlightView.vue";
import LastSeenSpaceHighlightView from "./views/LastSeenSpaceHighlightView.vue";
import LatestDrawComparisonView from "./views/LatestDrawComparisonView.vue";
import MetaStrategyView from "./views/MetaStrategyView.vue";
import NumbersView from "./views/NumbersView.vue";
import OverviewView from "./views/OverviewView.vue";
import PredictionAuditView from "./views/PredictionAuditView.vue";
import RandomnessView from "./views/RandomnessView.vue";
import RelationshipsView from "./views/RelationshipsView.vue";
import SpacesView from "./views/SpacesView.vue";
import StrategyEffectivenessView from "./views/StrategyEffectivenessView.vue";
import type {
  AnalysisOptions,
  AnalysisPayload,
  AnalysisProgress,
  CombinedPredictionDialogData,
  DatasetSelection,
  MenuAction,
  PossibleDrawNumberRequest,
  RecentDataset,
  ReportId,
  ReportPluginState,
  StrategyId,
  StrategyPlugin,
  StrategyPluginState,
  ViewId,
  WorkspaceTabId,
} from "./types";

const views: { id: ViewId; label: string; shortLabel: string }[] = [
  { id: "overview", label: "Overview", shortLabel: "Overview" },
  { id: "numbers", label: "Numbers", shortLabel: "Numbers" },
  { id: "spaces", label: "Spaces", shortLabel: "Spaces" },
  { id: "relationships", label: "Relationships", shortLabel: "Relationships" },
  { id: "randomness", label: "Randomness", shortLabel: "Randomness" },
  {
    id: "autocorrelation",
    label: "Autocorrelation",
    shortLabel: "Autocorrelation",
  },
  {
    id: "co-occurrence",
    label: "Co-occurrence",
    shortLabel: "Co-occurrence",
  },
  {
    id: "prediction-audit",
    label: "Prediction Audit",
    shortLabel: "Prediction Audit",
  },
  {
    id: "draw-comparison",
    label: "Latest Draw vs Predictions",
    shortLabel: "Draw vs Predictions",
  },
  {
    id: "strategy-effectiveness",
    label: "Strategy Effectiveness",
    shortLabel: "Effectiveness",
  },
  { id: "gaps", label: "Gaps", shortLabel: "Gaps" },
  { id: "export", label: "Export", shortLabel: "Export" },
];

const workspaceTabs: {
  id: WorkspaceTabId;
  label: string;
  reportId?: ReportId;
}[] = [
  { id: "statistics", label: "Statistics" },
  { id: "last-seen", label: "Last seen", reportId: "last-seen" },
  {
    id: "last-seen-gap",
    label: "Last seen gaps",
    reportId: "last-seen-gap",
  },
  {
    id: "last-seen-space",
    label: "Last seen spaces",
    reportId: "last-seen-space",
  },
  { id: "predictions", label: "Predictions", reportId: "predictions" },
  {
    id: "meta-strategy",
    label: "Meta Strategy",
    reportId: "meta-strategy",
  },
  {
    id: "draw-portfolio",
    label: "Draw Portfolio",
    reportId: "draw-portfolio",
  },
  {
    id: "possible-draw",
    label: "Possible Draw",
    reportId: "possible-draw",
  },
  { id: "draw-history", label: "Draw History" },
];

const activeView = ref<ViewId>("overview");
const activeWorkspaceTab = ref<WorkspaceTabId>("statistics");
const visitedWorkspaceTabs = ref<Set<WorkspaceTabId>>(new Set(["statistics"]));
const enabledReportIds = ref<ReportId[]>([]);
const enabledStrategyIds = ref<StrategyId[]>([]);
const strategyPlugins = ref<StrategyPlugin[]>([]);
const recentDatasets = ref<RecentDataset[]>([]);
const settingsOpen = ref(false);
const savingSettings = ref(false);
const updatingStrategySelection = ref(false);
const lastSeenDrawCountStorageKey = "rand-ai-last-seen-draw-count";
const storedLastSeenDrawCount = Number(
  window.localStorage.getItem(lastSeenDrawCountStorageKey) ?? "50",
);
const lastSeenDrawCount = ref(
  Number.isFinite(storedLastSeenDrawCount)
    ? Math.max(1, Math.trunc(storedLastSeenDrawCount))
    : 50,
);
const lastSeenReferenceOffset = ref(0);
const pendingDataset = ref<DatasetSelection | null>(null);
const activeDataset = ref<DatasetSelection | null>(null);
const analysis = ref<AnalysisPayload | null>(null);
const analysisStale = ref(false);
const possibleDrawNumberRequest = ref<
  (PossibleDrawNumberRequest & { token: number }) | null
>(null);
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
let possibleDrawRequestToken = 0;

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
const enabledWorkspaceTabs = computed(() =>
  workspaceTabs.filter(
    (tab) => !tab.reportId || enabledReportIds.value.includes(tab.reportId),
  ),
);
const activeViewLabel = computed(
  () => views.find((view) => view.id === activeView.value)?.label ?? "Overview",
);
const activeWorkspaceLabel = computed(() => {
  if (activeWorkspaceTab.value === "statistics") {
    return `Statistics · ${activeViewLabel.value}`;
  }
  return (
    workspaceTabs.find((tab) => tab.id === activeWorkspaceTab.value)?.label ??
    "Statistics"
  );
});
const combinedPredictionData = computed<CombinedPredictionDialogData | null>(
  () =>
    analysis.value
      ? {
          dataset: analysis.value.dataset,
          predictions: analysis.value.combinedPredictions,
          predictionSuites: analysis.value.predictionSuites,
          strategyEfficacyHistory: analysis.value.strategyEfficacyHistory,
          history: analysis.value.history,
          possibleDraw: analysis.value.possibleDraw,
        }
      : null,
);
const maxLastSeenDrawCount = computed(() =>
  Math.max(1, analysis.value?.history.length ?? 250),
);
const visibleLastSeenDrawCount = computed(() =>
  Math.min(lastSeenDrawCount.value, maxLastSeenDrawCount.value),
);

function isReportEnabled(reportId: ReportId): boolean {
  return enabledReportIds.value.includes(reportId);
}

function ensureActiveView(): void {
  if (enabledViews.value.some((view) => view.id === activeView.value)) return;
  activeView.value = enabledViews.value[0]?.id ?? "export";
}

function ensureActiveWorkspaceTab(): void {
  if (
    enabledWorkspaceTabs.value.some(
      (tab) => tab.id === activeWorkspaceTab.value,
    )
  ) {
    return;
  }
  selectWorkspaceTab("statistics");
}

function selectWorkspaceTab(tabId: WorkspaceTabId): void {
  if (!enabledWorkspaceTabs.value.some((tab) => tab.id === tabId)) return;
  activeWorkspaceTab.value = tabId;
  if (!visitedWorkspaceTabs.value.has(tabId)) {
    visitedWorkspaceTabs.value = new Set([
      ...visitedWorkspaceTabs.value,
      tabId,
    ]);
  }
}

function acceptReportPluginState(state: ReportPluginState): void {
  enabledReportIds.value = [...state.enabledReports];
  options.enabledReports = [...state.enabledReports];
  ensureActiveView();
  ensureActiveWorkspaceTab();
}

function acceptStrategyPluginState(state: StrategyPluginState): void {
  strategyPlugins.value = state.plugins.map((plugin) => ({ ...plugin }));
  enabledStrategyIds.value = [...state.enabledStrategies];
  options.enabledStrategies = [...state.enabledStrategies];
}

function strategySelectionChanged(strategyIds: readonly StrategyId[]): boolean {
  return (
    strategyIds.length !== enabledStrategyIds.value.length ||
    strategyIds.some(
      (strategyId, index) => strategyId !== enabledStrategyIds.value[index],
    )
  );
}

async function updateStrategySelection(
  strategyIds: StrategyId[],
  afterPersist?: () => void,
): Promise<boolean> {
  if (!window.randAiDesktop || !strategySelectionChanged(strategyIds)) {
    return false;
  }

  updatingStrategySelection.value = true;
  try {
    const state = await window.randAiDesktop.setStrategyPlugins(strategyIds);
    acceptStrategyPluginState(state);
    afterPersist?.();
    if (activeDataset.value) {
      if (loading.value) reportRefreshPending = true;
      else await analyzeDataset(activeDataset.value, normalizeOptions());
    }
    return true;
  } finally {
    updatingStrategySelection.value = false;
  }
}

async function applyPredictionStrategySelection(
  strategyIds: StrategyId[],
): Promise<void> {
  errorMessage.value = "";
  try {
    await updateStrategySelection(strategyIds);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
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

async function saveSettings(
  strategyIds: StrategyId[],
  requestedLastSeenDrawCount: number,
): Promise<void> {
  if (!window.randAiDesktop) return;
  savingSettings.value = true;
  errorMessage.value = "";
  try {
    const nextLastSeenDrawCount = Math.min(
      Math.max(Math.trunc(requestedLastSeenDrawCount || 1), 1),
      maxLastSeenDrawCount.value,
    );
    lastSeenDrawCount.value = nextLastSeenDrawCount;
    lastSeenReferenceOffset.value = Math.min(
      lastSeenReferenceOffset.value,
      nextLastSeenDrawCount - 1,
    );
    window.localStorage.setItem(
      lastSeenDrawCountStorageKey,
      String(nextLastSeenDrawCount),
    );
    if (strategySelectionChanged(strategyIds)) {
      await updateStrategySelection(strategyIds, () => {
        settingsOpen.value = false;
      });
    } else {
      settingsOpen.value = false;
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
    if (dataset) stageDataset(dataset);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

function recentDatasetKind(dataset: RecentDataset): string {
  return /\.ya?ml$/i.test(dataset.path) ? "YAML" : "Pickle";
}

async function chooseRecentDataset(dataset: RecentDataset): Promise<void> {
  errorMessage.value = "";
  if (!window.randAiDesktop) {
    errorMessage.value = "Dataset access is available in the Electron desktop application.";
    return;
  }
  try {
    stageDataset(await window.randAiDesktop.openRecentDataset(dataset.path));
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
    try {
      recentDatasets.value = await window.randAiDesktop.getRecentDatasets();
    } catch {
      // Keep the existing list if refreshing it also fails.
    }
  }
}

function stageDataset(dataset: DatasetSelection): void {
  errorMessage.value = "";
  if (dataset.requiresTrust === false) {
    pendingDataset.value = null;
    statusMessage.value = `Opening ${dataset.name}; preparing managed pickle…`;
    void analyzeDataset(dataset, normalizeOptions());
    return;
  }
  pendingDataset.value = dataset;
}

async function analyzeDataset(
  dataset: DatasetSelection,
  requestedOptions: AnalysisOptions,
  forceReanalysis = false,
): Promise<void> {
  if (!window.randAiDesktop) {
    errorMessage.value = "The Python bridge is only available inside Electron.";
    return;
  }
  loading.value = true;
  loadingDataset.value = dataset;
  loadingProgress.value = 1;
  loadingProgressTarget = 1;
  loadingExplanation.value =
    dataset.requiresTrust === false
      ? "Preparing the YAML dataset and managed pickle"
      : "Preparing the trusted dataset for the Python engine";
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
      forceReanalysis,
    });
    const datasetChanged = activeDataset.value?.path !== dataset.path;
    activeDataset.value = dataset;
    analysis.value = payload;
    analysisStale.value = false;
    if (datasetChanged) {
      lastSeenReferenceOffset.value = 0;
      selectWorkspaceTab("statistics");
    }
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

async function applyOptions(forceReanalysis = false): Promise<void> {
  if (!activeDataset.value) return;
  try {
    await analyzeDataset(
      activeDataset.value,
      normalizeOptions(),
      forceReanalysis,
    );
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

function handlePossibleDrawNumberRequest(
  request: PossibleDrawNumberRequest,
): void {
  possibleDrawRequestToken += 1;
  possibleDrawNumberRequest.value = {
    ...request,
    token: possibleDrawRequestToken,
  };
  selectWorkspaceTab("possible-draw");
}

function handleDrawHistorySaved(): void {
  analysisStale.value = true;
  statusMessage.value =
    "Draw history saved · Statistics need Analyze → Reanalyze";
}

function handleWorkspaceShortcut(event: KeyboardEvent): void {
  if (
    !event.altKey &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.shiftKey &&
    !settingsOpen.value &&
    !loading.value &&
    ["last-seen", "last-seen-gap", "last-seen-space"].includes(
      activeWorkspaceTab.value,
    ) &&
    (event.key === "ArrowDown" || event.key === "ArrowUp")
  ) {
    const maximumOffset = Math.max(0, visibleLastSeenDrawCount.value - 1);
    const nextOffset =
      event.key === "ArrowDown"
        ? Math.min(lastSeenReferenceOffset.value + 1, maximumOffset)
        : Math.max(lastSeenReferenceOffset.value - 1, 0);
    if (nextOffset !== lastSeenReferenceOffset.value) {
      lastSeenReferenceOffset.value = nextOffset;
    }
    event.preventDefault();
    return;
  }
  if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  const tabId = {
    Digit1: "predictions",
    Digit2: "possible-draw",
    Digit3: "draw-history",
    Digit4: "draw-portfolio",
    Digit5: "meta-strategy",
  }[event.code] as WorkspaceTabId | undefined;
  if (!tabId) return;
  event.preventDefault();
  selectWorkspaceTab(tabId);
}

function handleMenuAction(message: MenuAction): void {
  if (message.action === "datasetSelected") {
    stageDataset(message.dataset);
    return;
  }
  if (message.action === "openView") {
    selectWorkspaceTab("statistics");
    activeView.value = message.view;
    return;
  }
  if (message.action === "openWorkspaceTab") {
    selectWorkspaceTab(message.tab);
    return;
  }
  if (message.action === "openSettings") {
    void openSettings();
    return;
  }
  if (message.action === "reanalyze") {
    if (activeDataset.value && !loading.value) void applyOptions(true);
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
  window.addEventListener("keydown", handleWorkspaceShortcut);
  unsubscribeMenu = window.randAiDesktop?.onMenuAction(handleMenuAction) ?? null;
  unsubscribeAnalysisProgress =
    window.randAiDesktop?.onAnalysisProgress(handleAnalysisProgress) ?? null;
  if (window.randAiDesktop) {
    try {
      const [reportState, strategyState, recentDatasetState] = await Promise.all([
        window.randAiDesktop.getReportPlugins(),
        window.randAiDesktop.getStrategyPlugins(),
        window.randAiDesktop.getRecentDatasets(),
      ]);
      acceptReportPluginState(reportState);
      acceptStrategyPluginState(strategyState);
      recentDatasets.value = recentDatasetState;
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : String(error);
    }
  }
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleWorkspaceShortcut);
  if (progressTimer !== null) clearInterval(progressTimer);
  unsubscribeMenu?.();
  unsubscribeAnalysisProgress?.();
});
</script>

<template>
  <div class="app-shell">
    <header class="app-toolbar">
      <div class="brand-lockup">
        <div>
          <strong>Rand AI</strong>
        </div>
      </div>
      <div
        v-show="activeWorkspaceTab === 'predictions'"
        id="prediction-toolbar-navigation"
        class="prediction-toolbar-navigation-slot"
      ></div>
      <div
        v-show="activeWorkspaceTab === 'meta-strategy'"
        id="meta-strategy-toolbar-navigation"
        class="prediction-toolbar-navigation-slot"
      ></div>
      <nav
        v-if="analysis"
        class="toolbar-workspace-tabs"
        role="tablist"
        aria-label="Main workspaces"
      >
        <button
          v-for="tab in enabledWorkspaceTabs"
          :key="tab.id"
          :class="{
            active: activeWorkspaceTab === tab.id,
            stale: tab.id === 'statistics' && analysisStale,
          }"
          :disabled="loading"
          type="button"
          role="tab"
          :aria-selected="activeWorkspaceTab === tab.id"
          :tabindex="activeWorkspaceTab === tab.id ? 0 : -1"
          @click="selectWorkspaceTab(tab.id)"
        >
          <span>{{ tab.label }}</span>
        </button>
      </nav>
    </header>

    <nav
      v-if="analysis"
      v-show="activeWorkspaceTab === 'statistics'"
      class="workspace-tabs"
      aria-label="Statistics views"
    >
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

    <div
      v-if="analysis"
      v-show="activeWorkspaceTab === 'statistics'"
      class="dashboard-layout"
    >
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
        <button class="button primary apply-button" :disabled="loading" type="button" @click="applyOptions()">
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
        <AutocorrelationView
          v-else-if="activeView === 'autocorrelation' && analysis.options.enabledReports.includes('autocorrelation')"
          :analysis="analysis"
        />
        <CoOccurrenceView
          v-else-if="activeView === 'co-occurrence' && analysis.options.enabledReports.includes('co-occurrence')"
          :analysis="analysis"
        />
        <PredictionAuditView
          v-else-if="activeView === 'prediction-audit' && analysis.options.enabledReports.includes('prediction-audit')"
          :analysis="analysis"
        />
        <LatestDrawComparisonView
          v-else-if="activeView === 'draw-comparison' && analysis.options.enabledReports.includes('draw-comparison')"
          :analysis="analysis"
        />
        <StrategyEffectivenessView
          v-else-if="activeView === 'strategy-effectiveness' && analysis.options.enabledReports.includes('strategy-effectiveness')"
          :analysis="analysis"
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

    <main
      v-if="analysis && visitedWorkspaceTabs.has('last-seen')"
      v-show="activeWorkspaceTab === 'last-seen'"
      class="workspace-tab-panel last-seen-dialog-shell embedded-workspace-panel"
      role="tabpanel"
      aria-label="Last seen"
    >
      <LastSeenHighlightView
        :history="analysis.history"
        :draw-count="lastSeenDrawCount"
        :reference-draw-offset="lastSeenReferenceOffset"
      />
    </main>

    <main
      v-if="analysis && visitedWorkspaceTabs.has('last-seen-gap')"
      v-show="activeWorkspaceTab === 'last-seen-gap'"
      class="workspace-tab-panel last-seen-dialog-shell embedded-workspace-panel"
      role="tabpanel"
      aria-label="Last seen gaps"
    >
      <LastSeenGapHighlightView
        :history="analysis.history"
        :draw-count="lastSeenDrawCount"
        :reference-draw-offset="lastSeenReferenceOffset"
      />
    </main>

    <main
      v-if="analysis && visitedWorkspaceTabs.has('last-seen-space')"
      v-show="activeWorkspaceTab === 'last-seen-space'"
      class="workspace-tab-panel last-seen-dialog-shell embedded-workspace-panel"
      role="tabpanel"
      aria-label="Last seen spaces"
    >
      <LastSeenSpaceHighlightView
        :history="analysis.history"
        :draw-count="lastSeenDrawCount"
        :reference-draw-offset="lastSeenReferenceOffset"
      />
    </main>

    <main
      v-if="analysis && visitedWorkspaceTabs.has('predictions')"
      v-show="activeWorkspaceTab === 'predictions'"
      class="workspace-tab-panel combined-prediction-dialog-shell embedded-workspace-panel"
      role="tabpanel"
      aria-label="Predictions"
    >
      <CombinedPredictionGridView
        :prediction-suites="analysis.predictionSuites"
        :efficacy-history="analysis.strategyEfficacyHistory"
        :strategy-plugins="strategyPlugins"
        :enabled-strategy-ids="enabledStrategyIds"
        :strategy-selection-busy="updatingStrategySelection || loading"
        embedded
        @apply-strategies="applyPredictionStrategySelection"
        @send-number="handlePossibleDrawNumberRequest"
      />
    </main>

    <main
      v-if="analysis && visitedWorkspaceTabs.has('meta-strategy')"
      v-show="activeWorkspaceTab === 'meta-strategy'"
      class="workspace-tab-panel embedded-workspace-panel"
      role="tabpanel"
      aria-label="Meta Strategy"
    >
      <MetaStrategyView
        :meta-history="analysis.metaDrawHistory"
        :strategy-plugins="strategyPlugins"
      />
    </main>

    <section
      v-if="analysis && visitedWorkspaceTabs.has('draw-portfolio')"
      v-show="activeWorkspaceTab === 'draw-portfolio'"
      class="workspace-tab-panel embedded-workspace-panel"
      role="tabpanel"
      aria-label="Draw Portfolio"
    >
      <DrawPortfolioView
        :prediction-suites="analysis.predictionSuites"
        :relationship-edges="analysis.possibleDraw.relationshipEdges"
      />
    </section>

    <section
      v-if="combinedPredictionData && visitedWorkspaceTabs.has('possible-draw')"
      v-show="activeWorkspaceTab === 'possible-draw'"
      class="workspace-tab-panel embedded-workspace-panel"
      role="tabpanel"
      aria-label="Possible Draw"
    >
      <PossibleDrawDialogApp
        :dialog-data="combinedPredictionData"
        :number-request="possibleDrawNumberRequest"
        embedded
      />
    </section>

    <section
      v-if="analysis && visitedWorkspaceTabs.has('draw-history')"
      v-show="activeWorkspaceTab === 'draw-history'"
      class="workspace-tab-panel embedded-workspace-panel"
      role="tabpanel"
      aria-label="Draw History"
    >
      <DrawEditorDialogApp
        :key="analysis.dataset.path"
        embedded
        @saved="handleDrawHistorySaved"
      />
    </section>

    <main v-if="!analysis" class="welcome-screen">
      <section class="welcome-card">
        <p class="eyebrow">Vue 3 + Electron</p>
        <h1>Explore draw history without a browser dashboard.</h1>
        <p>
          Open a Draws YAML file to generate its managed pickle and start the
          analysis automatically. You can also open an existing trusted pickle.
        </p>
        <button class="button primary large" type="button" @click="chooseDataset">
          Open YAML or trusted pickle
        </button>
        <small>YAML imports start immediately. Existing pickle files require a trust confirmation.</small>
      </section>
      <div class="welcome-sidebar">
        <section class="welcome-recents" aria-label="Recent dataset files">
          <header>
            <div>
              <p class="eyebrow">Open again</p>
              <h2>Recent files</h2>
            </div>
            <span>{{ recentDatasets.length }}/10</span>
          </header>
          <ul v-if="recentDatasets.length > 0">
            <li v-for="dataset in recentDatasets" :key="dataset.path">
              <button
                type="button"
                :disabled="loading"
                :title="dataset.path"
                @click="chooseRecentDataset(dataset)"
              >
                <span>
                  <strong>{{ dataset.name }}</strong>
                  <small>{{ dataset.path }}</small>
                </span>
                <b>{{ recentDatasetKind(dataset) }}</b>
              </button>
            </li>
          </ul>
          <p v-else>
            YAML and trusted pickle files you open will appear here.
          </p>
        </section>
        <section class="welcome-features">
          <article><strong>21</strong><span>Interactive statistics charts</span></article>
          <article><strong>49</strong><span>Number-level frequency positions</span></article>
          <article><strong>250</strong><span>Draws in the last-seen workspace</span></article>
        </section>
      </div>
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
      :last-seen-draw-count="lastSeenDrawCount"
      :max-last-seen-draw-count="maxLastSeenDrawCount"
      :saving="savingSettings"
      @cancel="settingsOpen = false"
      @save="saveSettings"
    />

    <footer class="status-bar">
      <span :class="{ ready: Boolean(analysis) && !loading }">{{ statusMessage }}</span>
      <span>Dataset: {{ loadingDataset?.name ?? analysis?.dataset.name ?? activeDataset?.name ?? "none" }}</span>
      <span>Draws: {{ analysis?.dataset.drawCount.toLocaleString() ?? "—" }}</span>
      <span>View: {{ analysis ? activeWorkspaceLabel : "Welcome" }}</span>
      <span>Python engine · Vue renderer · Electron shell</span>
    </footer>
  </div>
</template>
