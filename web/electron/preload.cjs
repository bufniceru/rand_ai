const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("randAiDesktop", {
  openDataset: () => ipcRenderer.invoke("dataset:open"),
  getRecentDatasets: () => ipcRenderer.invoke("dataset:recent-list"),
  openRecentDataset: (path) => ipcRenderer.invoke("dataset:recent-open", path),
  getReportPlugins: () => ipcRenderer.invoke("report-plugins:get"),
  getStrategyPlugins: () => ipcRenderer.invoke("strategy-plugins:get"),
  setStrategyPlugins: (strategyIds) =>
    ipcRenderer.invoke("strategy-plugins:set", strategyIds),
  analyzeDataset: (request) => ipcRenderer.invoke("dataset:analyze", request),
  onAnalysisProgress: (callback) => {
    const listener = (_event, progress) => callback(progress);
    ipcRenderer.on("dataset:analysis-progress", listener);
    return () => ipcRenderer.removeListener("dataset:analysis-progress", listener);
  },
  getPortfolioBacktestData: (request) =>
    ipcRenderer.invoke("portfolio-backtest:data", request),
  loadPortfolioBacktest: (key) =>
    ipcRenderer.invoke("portfolio-backtest:cache-load", key),
  savePortfolioBacktest: (request) =>
    ipcRenderer.invoke("portfolio-backtest:cache-save", request),
  onPortfolioBacktestProgress: (callback) => {
    const listener = (_event, progress) => callback(progress);
    ipcRenderer.on("portfolio-backtest:progress", listener);
    return () => ipcRenderer.removeListener("portfolio-backtest:progress", listener);
  },
  exportAnalysis: (request) => ipcRenderer.invoke("analysis:export", request),
  saveDrawComparisonPdf: (request) =>
    ipcRenderer.invoke("draw-comparison:save-pdf", request),
  saveDrawPortfolioPdf: (request) =>
    ipcRenderer.invoke("draw-portfolio:save-pdf", request),
  printDrawComparison: () => ipcRenderer.invoke("draw-comparison:print"),
  showForSureLimitError: (number) =>
    ipcRenderer.invoke("possible-draw:for-sure-limit-error", number),
  getDrawEditorData: () => ipcRenderer.invoke("draw-editor:data"),
  saveDraw: (request) => ipcRenderer.invoke("draw-editor:save", request),
  onMenuAction: (callback) => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on("menu:action", listener);
    return () => ipcRenderer.removeListener("menu:action", listener);
  },
});
