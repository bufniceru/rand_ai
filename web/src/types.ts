export type CorrelationMethod = "pearson" | "spearman";
export type ReportId =
  | "overview"
  | "numbers"
  | "spaces"
  | "relationships"
  | "randomness"
  | "last-seen"
  | "last-seen-gap"
  | "predictions"
  | "possible-draw";
export type StrategyId =
  | "proximity"
  | "freshness"
  | "emd"
  | "randomness"
  | "entropy"
  | "markov100"
  | "bayesian"
  | "svc"
  | "tbl";
export type ViewId =
  | "overview"
  | "numbers"
  | "spaces"
  | "relationships"
  | "randomness"
  | "export";

export interface AnalysisOptions {
  selectedNumbers: number[];
  trendBins: number;
  correlationMethod: CorrelationMethod;
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

export interface AnalysisPayload {
  dataset: DatasetSummary;
  options: AnalysisOptions;
  tables: Record<string, TablePayload>;
  history: HistoryDraw[];
  combinedPredictions: CombinedPredictionHistory[];
  predictionSuites: PredictionSuite[];
  possibleDraw: PossibleDrawAnalysis;
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

export interface StrategyPrediction {
  id: StrategyId;
  name: string;
  description: string;
  topNumbers: number[];
  numbers: StrategyNumberPrediction[];
}

export interface PredictionSuite {
  referenceDrawNumber: number;
  targetDrawNumber: number;
  actualNumbers: number[];
  strategies: StrategyPrediction[];
}

export interface AnalysisProgress {
  percent: number;
  message: string;
}

export interface LastSeenDialogData {
  dataset: DatasetSummary;
  history: HistoryDraw[];
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
  | ({ action: "reportPluginsChanged" } & ReportPluginState)
  | ({ action: "strategyPluginsChanged" } & StrategyPluginState)
  | { action: "export" };

export interface ExportResult {
  canceled: boolean;
  path?: string;
}

export interface DesktopApi {
  openDataset(): Promise<DatasetSelection | null>;
  getReportPlugins(): Promise<ReportPluginState>;
  getStrategyPlugins(): Promise<StrategyPluginState>;
  analyzeDataset(request: {
    path: string;
    options: AnalysisOptions;
  }): Promise<AnalysisPayload>;
  onAnalysisProgress(
    callback: (progress: AnalysisProgress) => void,
  ): () => void;
  exportAnalysis(request: {
    options: AnalysisOptions;
  }): Promise<ExportResult>;
  openLastSeenDialog(): Promise<void>;
  openLastSeenGapDialog(): Promise<void>;
  openCombinedPredictionDialog(): Promise<void>;
  openPossibleDrawDialog(): Promise<void>;
  openDrawEditorDialog(): Promise<void>;
  getDrawEditorData(): Promise<DrawEditorData>;
  saveDraw(request: DrawSaveRequest): Promise<DrawEditorData>;
  getCombinedPredictionData(): Promise<CombinedPredictionDialogData | null>;
  onCombinedPredictionData(
    callback: (data: CombinedPredictionDialogData) => void,
  ): () => void;
  getLastSeenData(): Promise<LastSeenDialogData | null>;
  onLastSeenData(
    callback: (data: LastSeenDialogData) => void,
  ): () => void;
  onMenuAction(callback: (message: MenuAction) => void): () => void;
}

declare global {
  interface Window {
    randAiDesktop?: DesktopApi;
  }
}
