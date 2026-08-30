export type CorrelationMethod = "pearson" | "spearman";
export type ReportId =
  | "overview"
  | "numbers"
  | "spaces"
  | "space-groups"
  | "relationships"
  | "randomness"
  | "nonlinear-dynamics"
  | "autocorrelation"
  | "co-occurrence"
  | "prediction-audit"
  | "draw-comparison"
  | "strategy-effectiveness"
  | "gaps"
  | "last-seen"
  | "last-seen-gap"
  | "last-seen-space"
  | "predictions"
  | "draw-portfolio"
  | "possible-draw";
export type StrategyId =
  | "proximity"
  | "freshness"
  | "emd"
  | "recurrence_dynamics"
  | "randomness"
  | "fresh_random"
  | "chi_square"
  | "categorical_chi_square"
  | "entropy"
  | "markov100"
  | "mkgsv"
  | "mkfr"
  | "mksp"
  | "mknp"
  | "mkrd"
  | "bayesian"
  | "predictive_grid"
  | "co_occurrence"
  | "doublet_triplet_markov"
  | "mixed"
  | "svc"
  | "svc_recurrence_hybrid"
  | "svc_recurrence_proximity_hybrid"
  | "srph_residual_diversity_hybrid"
  | "srph_minimax_regret_hybrid"
  | "tbl"
  | "sklearn_svm"
  | "lag_logistic"
  | "sparse_neural_ticket"
  | "cis"
  | "decision_tree_selector"
  | "border_group_statistical"
  | "border_group_markov"
  | "border_group_bayesian"
  | "border_group_ml"
  | "border_group_hybrid"
  | "residual_coverage"
  | "chained";
export type ViewId =
  | "overview"
  | "numbers"
  | "spaces"
  | "space-groups"
  | "relationships"
  | "randomness"
  | "nonlinear-dynamics"
  | "autocorrelation"
  | "co-occurrence"
  | "prediction-audit"
  | "draw-comparison"
  | "strategy-effectiveness"
  | "gaps"
  | "export";
export type WorkspaceTabId =
  | "statistics"
  | "last-seen"
  | "last-seen-gap"
  | "last-seen-space"
  | "predictions"
  | "draw-portfolio"
  | "possible-draw"
  | "draw-history";

export interface AnalysisOptions {
  selectedNumbers: number[];
  trendBins: number;
  correlationMethod: CorrelationMethod;
  borderSpace: number;
  targetGroupCount: number | null;
  enabledReports: ReportId[];
  enabledStrategies: StrategyId[];
}

export interface ReportPlugin {
  id: ReportId;
  label: string;
  enabled: boolean;
}

export interface ReportPluginState {
  plugins: ReportPlugin[];
  enabledReports: ReportId[];
}

export interface StrategyPlugin {
  id: StrategyId;
  label: string;
  enabled: boolean;
}

export interface StrategyPluginState {
  plugins: StrategyPlugin[];
  enabledStrategies: StrategyId[];
}

export interface DatasetSelection {
  path: string;
  name: string;
  sizeBytes: number;
  requiresTrust?: boolean;
}

export interface RecentDataset extends DatasetSelection {
  lastOpenedAt?: string;
}

export interface DatasetSummary extends DatasetSelection {
  drawCount: number;
  numberObservations: number;
  sampleSize: number;
  historyWindowStart: number;
}

export type TableValue = string | number | boolean | null;
export type TableRow = Record<string, TableValue>;

export interface TablePayload {
  columns: string[];
  rows: TableRow[];
}

export type StatisticsCommandId = "statistics.number-frequency";

export interface StatisticsCommandPayload {
  id: StatisticsCommandId;
  datasetName: string;
  drawCount: number;
  table: TablePayload;
}

export interface HistoryNumber {
  value: number;
  gap: number;
  leftSpace: number;
  rightSpace: number;
}

export interface HistoryDraw {
  drawNumber: number;
  date: string | null;
  numbers: HistoryNumber[];
}

export interface AnalysisHistoryDraw {
  date: string | null;
  numbers: number[];
}

export interface AnalysisPayload {
  dataset: DatasetSummary;
  options: AnalysisOptions;
  tables: Record<string, TablePayload>;
  history: HistoryDraw[];
  analysisHistory: AnalysisHistoryDraw[];
  combinedPredictions: CombinedPredictionHistory[];
  predictionSuites: PredictionSuite[];
  strategyEfficacyHistory: StrategyEfficacyRecord[];
  predictionAuditHistory: PredictionAuditRecord[];
  drawComparisonHistory: LatestDrawComparison[];
  latestDrawComparison: LatestDrawComparison | null;
  possibleDraw: PossibleDrawAnalysis;
  spaceGroups: SpaceGroupAnalysis | null;
  nonlinearDynamics: NonlinearDynamicsAnalysis | null;
}

export type NonlinearEvidenceStatus =
  | "insufficient"
  | "weak"
  | "suggestive"
  | "supported";

export interface NonlinearDynamicsMetrics {
  recurrenceRate: number;
  determinism: number;
  meanDiagonalLength: number;
  maximumDiagonalLength: number;
  laminarity: number;
  trappingTime: number;
}

export interface NonlinearDynamicsAnalysis {
  status: NonlinearEvidenceStatus;
  summary: string;
  caveat: string;
  drawCount: number;
  embeddingCount: number;
  embeddingDimension: number;
  recurrenceThreshold: number;
  metrics: NonlinearDynamicsMetrics;
  surrogate: {
    count: number;
    meanDeterminism: number;
    standardDeviation: number;
    pValue: number;
  };
  forecast: {
    evaluatedDraws: number;
    averageHitsPerDraw: number;
    lowerConfidenceBound: number;
    expectedRandomHitsPerDraw: number;
  };
  latest: {
    analogueCount: number;
    effectiveNeighbors: number;
    distancePercentile: number;
    evidenceScore: number;
    topNumbers: number[];
  };
  plot: {
    size: number;
    points: { x: number; y: number }[];
  };
}

export interface SpaceGroupProbability {
  signature: string;
  groupCount: number;
  probability: number;
}

export interface SpaceGroupForecast {
  modelId: StrategyId;
  name: string;
  topSignature: string;
  topGroupCount: number;
  topProbability: number;
  probabilities: SpaceGroupProbability[];
}

export interface SpaceGroupAnalysis {
  borderSpace: number;
  targetGroupCount: number | null;
  smallSpaceDefinition: string;
  largeSpaceDefinition: string;
  bestModelId: StrategyId | null;
  provisional: boolean;
  forecasts: SpaceGroupForecast[];
  hybridWeights: Partial<Record<StrategyId, number>>;
  signatureChiSquare: number;
  signatureChiSquarePValue: number;
  transitionMutualInformation: number;
  transitionPermutationPValue: number;
}

export interface CoOccurrenceBand {
  id: string;
  label: string;
  description: string;
  color: string;
}

export interface CoOccurrenceEdge {
  pair: string;
  numbers: [number, number];
  count: number;
  expected: number;
  lift: number;
  residual: number;
  share: number;
  rank: number;
  bandId: string;
  label: string;
}

export interface CoOccurrenceNode {
  number: number;
  appearances: number;
  weightedDegree: number;
  averagePartnerCount: number;
  strongestPartner: number | null;
  strongestPartnerCount: number;
  strongestPartnerLift: number;
  rank: number;
}

export interface CoOccurrencePrediction {
  number: number;
  rank: number;
  score: number;
  averageLift: number;
  totalCount: number;
  strongestPartner: number | null;
  strongestPartnerCount: number;
  strongestPartnerLift: number;
  bandId: string;
  label: string;
}

export interface CoOccurrenceModel {
  bands: CoOccurrenceBand[];
  drawCount: number;
  totalPairEvents: number;
  expectedPairCount: number;
  pairUniverseSize: number;
  maxEdgeCount: number;
  maxWeightedDegree: number;
  edges: CoOccurrenceEdge[];
  nodes: CoOccurrenceNode[];
  predictions: CoOccurrencePrediction[];
  networkEdges: CoOccurrenceEdge[];
  latestProfile: {
    date: string | null;
    signature: string;
    edges: CoOccurrenceEdge[];
  };
  interpretation: string;
}

export interface AutocorrelationBand {
  id: string;
  label: string;
  description: string;
  color: string;
}

export interface AutocorrelationLagSummary {
  lag: number;
  pairCount: number;
  averageOverlap: number;
  expectedOverlap: number;
  overlapDelta: number;
  overlapRate: number;
  averageDoublets: number;
  expectedDoublets: number;
  doubletDelta: number;
  averageTriplets: number;
  expectedTriplets: number;
  tripletDelta: number;
  numberPresenceCorrelation: number;
  sumCorrelation: number;
  oddCountCorrelation: number;
  lowCountCorrelation: number;
  score: number;
  bandId: string;
  label: string;
}

export interface AutocorrelationNumberSummary {
  number: number;
  appearances: number;
  strongestLag: number;
  strongestCorrelation: number;
  score: number;
  bandId: string;
  label: string;
  rank: number;
}

export interface AutocorrelationModel {
  bands: AutocorrelationBand[];
  drawCount: number;
  maxLag: number;
  expectedOverlap: number;
  expectedDoublets: number;
  expectedTriplets: number;
  lagSummaries: AutocorrelationLagSummary[];
  numberSummaries: AutocorrelationNumberSummary[];
  strongestLag: AutocorrelationLagSummary | null;
  strongestPositiveLag: AutocorrelationLagSummary | null;
  strongestNegativeLag: AutocorrelationLagSummary | null;
  latestProfile: {
    date: string | null;
    signature: string;
    numbers: AutocorrelationNumberSummary[];
  };
  interpretation: string;
}

export interface CombinedPredictionNumber {
  number: number;
  rank: number;
  score: number;
  gap: number;
  leftSpace: number | null;
  rightSpace: number | null;
}

export interface CombinedPredictionHistory {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  actualNumbers: number[];
  topNumbers: number[];
  numbers: CombinedPredictionNumber[];
}

export interface CombinedPredictionDialogData {
  dataset: DatasetSummary;
  predictions: CombinedPredictionHistory[];
  predictionSuites: PredictionSuite[];
  strategyEfficacyHistory: StrategyEfficacyRecord[];
  history: HistoryDraw[];
  possibleDraw: PossibleDrawAnalysis;
}

export interface LastSeenRow {
  number: number;
  gap: number;
}

export interface RelationshipEdge {
  left: number;
  right: number;
  count: number;
  expected: number;
  lift: number;
  residual: number;
}

export interface PossibleDrawAnalysis {
  lastDrawNumbers: number[];
  lastSeenRows: LastSeenRow[];
  relationshipEdges: RelationshipEdge[];
}

export interface StrategyNumberPrediction {
  number: number;
  rank: number;
  score: number;
  gap: number;
  details: string[];
}

export interface StrategyEfficacy {
  evaluatedDraws: number;
  strategyHits: number;
  randomHits: number;
  expectedRandomHits: number;
  averageHitsPerDraw: number;
  randomAverageHitsPerDraw: number;
  hitDifference: number;
}

export interface StrategyEvidence {
  status: NonlinearEvidenceStatus;
  score: number;
  summary: string;
  evaluatedForecasts: number;
  analogueCount: number;
  effectiveNeighbors: number;
  distancePercentile: number;
  averageHitsPerDraw: number;
}

export interface StrategyEfficacyRecord {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  actualNumbers: number[];
  randomHits: number;
  strategyHits: Partial<Record<StrategyId, number>>;
}

export interface PredictionAuditStrategy {
  id: StrategyId;
  name: string;
}

export interface PredictionAuditNumber {
  number: number;
  strategies: PredictionAuditStrategy[];
}

export interface PredictionAuditRecord {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  date: string | null;
  numbers: PredictionAuditNumber[];
}

export interface DrawComparisonStrategy {
  id: StrategyId;
  name: string;
  description: string;
  predictedNumbers: number[];
  matchedNumbers: number[];
  missedPredictions: number[];
  missedActualNumbers: number[];
  hitCount: number;
  efficacy: StrategyEfficacy | null;
}

export interface LatestDrawComparison {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  date: string | null;
  actualNumbers: number[];
  strategies: DrawComparisonStrategy[];
}

export interface StrategyPrediction {
  id: StrategyId;
  name: string;
  description: string;
  topNumbers: number[];
  numbers: StrategyNumberPrediction[];
  efficacy: StrategyEfficacy | null;
  evidence?: StrategyEvidence | null;
}

export interface PredictionSuite {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  actualNumbers: number[];
  strategies: StrategyPrediction[];
}

export interface PortfolioBacktestStrategy {
  id: StrategyId;
  ranking: number[];
}

export interface PortfolioBacktestRecord {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  date: string | null;
  actualNumbers: number[];
  strategies: PortfolioBacktestStrategy[];
}

export interface PortfolioBacktestData {
  cacheKey: string;
  strategyIds: StrategyId[];
  draws: AnalysisHistoryDraw[];
  records: PortfolioBacktestRecord[];
}

export interface PortfolioBacktestAuditRow {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  date: string | null;
  actualNumbers: number[];
  bestTicket: number[];
  bestHits: number;
  tiedBestCount: number;
}

export interface PortfolioBacktestHitBucket {
  hits: number;
  exactCount: number;
  exactRate: number;
  atLeastCount: number;
  atLeastRate: number;
}

export interface PortfolioBacktestResult {
  algorithmVersion: number;
  sourceCacheKey: string;
  portfolioSize: number;
  evaluatedTargets: number;
  durationMs: number;
  buckets: PortfolioBacktestHitBucket[];
  audit: PortfolioBacktestAuditRow[];
}

export interface PortfolioBacktestProgress {
  percent: number;
  message: string;
  processed?: number;
  total?: number;
}

export interface AnalysisProgress {
  percent: number;
  message: string;
}

export interface DrawEditorEntry {
  index: number;
  date: string;
  numbers: number[];
}

export interface DrawEditorData {
  picklePath: string;
  yamlPath: string;
  draws: DrawEditorEntry[];
}

export interface DrawSaveRequest {
  date: string;
  numbers: number[];
  originalDate?: string;
}

export interface FigureSpec {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
  config?: Record<string, unknown>;
}

export type MenuAction =
  | { action: "datasetSelected"; dataset: DatasetSelection }
  | { action: "openView"; view: ViewId }
  | { action: "openWorkspaceTab"; tab: WorkspaceTabId }
  | { action: "openSettings" }
  | { action: "openCommandPalette" }
  | { action: "reanalyze" }
  | ({ action: "reportPluginsChanged" } & ReportPluginState)
  | ({ action: "strategyPluginsChanged" } & StrategyPluginState)
  | { action: "export" };

export interface ExportResult {
  canceled: boolean;
  path?: string;
}

export interface DrawComparisonPdfRequest {
  suggestedName: string;
}

export interface DrawPortfolioPdfRequest {
  suggestedName: string;
}

export type HexColor = `#${string}`;
export type ColorTokenId = string;

export interface ColorTemplate {
  kind: "rand-ai-color-template";
  schemaVersion: 1;
  name: string;
  description?: string;
  createdWith?: string;
  exportedAt?: string;
  colors: Record<ColorTokenId, HexColor>;
}

export type ColorTemplateDraft = ColorTemplate;

export interface ColorTemplateLoadResult {
  canceled: boolean;
  path?: string;
  template?: unknown;
}

export interface ColorTemplateSaveResult {
  canceled: boolean;
  path?: string;
}

export type PossibleDrawNumberState =
  | "neutral"
  | "candidate"
  | "fixed"
  | "excluded";

export type DrawPortfolioMode = "classic" | "guided";

export interface PossibleDrawPlan {
  id: string;
  name: string;
  fixedNumbers: number[];
  candidateNumbers: number[];
  excludedNumbers: number[];
}

export interface PossibleDrawPlanContext {
  datasetId: string;
  targetDrawId: string;
}

export interface PossibleDrawConstraints {
  mode: DrawPortfolioMode;
  fixedNumbers: number[];
  candidateNumbers: number[];
  excludedNumbers: number[];
}

export interface DrawPortfolioResultMetadata {
  mode: DrawPortfolioMode;
  requestedDrawCount: number;
  generatedDrawCount: number;
  availableUniqueCount: number;
  fixedNumbers: number[];
  candidateNumbers: number[];
  excludedNumbers: number[];
  omittedCandidates: number[];
  constraintLimited: boolean;
  constraintMessage?: string;
  borderSpace?: number;
  groupModelId?: StrategyId;
  provisionalGroupModel?: boolean;
}

export interface PossibleDrawNumberRequest {
  number: number;
  state: PossibleDrawNumberState;
}

export interface DesktopApi {
  openDataset(): Promise<DatasetSelection | null>;
  getRecentDatasets(): Promise<RecentDataset[]>;
  openRecentDataset(path: string): Promise<DatasetSelection>;
  getReportPlugins(): Promise<ReportPluginState>;
  getStrategyPlugins(): Promise<StrategyPluginState>;
  setStrategyPlugins(strategyIds: StrategyId[]): Promise<StrategyPluginState>;
  analyzeDataset(request: {
    path: string;
    options: AnalysisOptions;
    forceReanalysis?: boolean;
  }): Promise<AnalysisPayload>;
  runStatisticsCommand(
    commandId: StatisticsCommandId,
  ): Promise<StatisticsCommandPayload>;
  onAnalysisProgress(
    callback: (progress: AnalysisProgress) => void,
  ): () => void;
  getPortfolioBacktestData(request: {
    strategyIds: StrategyId[];
    borderSpace: number;
    targetGroupCount: number | null;
  }): Promise<PortfolioBacktestData>;
  loadPortfolioBacktest(key: string): Promise<PortfolioBacktestResult | null>;
  savePortfolioBacktest(request: {
    key: string;
    result: PortfolioBacktestResult;
  }): Promise<void>;
  onPortfolioBacktestProgress(
    callback: (progress: PortfolioBacktestProgress) => void,
  ): () => void;
  exportAnalysis(request: {
    options: AnalysisOptions;
  }): Promise<ExportResult>;
  saveDrawComparisonPdf(
    request: DrawComparisonPdfRequest,
  ): Promise<ExportResult>;
  saveDrawPortfolioPdf(request: DrawPortfolioPdfRequest): Promise<ExportResult>;
  getColorTemplate(): Promise<unknown | null>;
  applyColorTemplate(template: ColorTemplate): Promise<ColorTemplate>;
  loadColorTemplate(): Promise<ColorTemplateLoadResult>;
  saveColorTemplate(
    template: ColorTemplate,
  ): Promise<ColorTemplateSaveResult>;
  printDrawComparison(): Promise<void>;
  showForSureLimitError(number: number): Promise<void>;
  getDrawEditorData(): Promise<DrawEditorData>;
  saveDraw(request: DrawSaveRequest): Promise<DrawEditorData>;
  onMenuAction(callback: (message: MenuAction) => void): () => void;
}

declare global {
  interface Window {
    randAiDesktop?: DesktopApi;
  }
}
