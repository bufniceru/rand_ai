const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("randAiDesktop", {
  openDataset: () => ipcRenderer.invoke("dataset:open"),
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
  exportAnalysis: (request) => ipcRenderer.invoke("analysis:export", request),
  openLastSeenDialog: () => ipcRenderer.invoke("last-seen:open"),
  openLastSeenGapDialog: () => ipcRenderer.invoke("last-seen-gap:open"),
  openCombinedPredictionDialog: () => ipcRenderer.invoke("combined-prediction:open"),
  openPossibleDrawDialog: () => ipcRenderer.invoke("possible-draw:open"),
  openDrawEditorDialog: () => ipcRenderer.invoke("draw-editor:open"),
  getDrawEditorData: () => ipcRenderer.invoke("draw-editor:data"),
  saveDraw: (request) => ipcRenderer.invoke("draw-editor:save", request),
  getCombinedPredictionData: () => ipcRenderer.invoke("combined-prediction:data"),
  onCombinedPredictionData: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on("combined-prediction:updated", listener);
    return () => ipcRenderer.removeListener("combined-prediction:updated", listener);
  },
  getLastSeenData: () => ipcRenderer.invoke("last-seen:data"),
  onLastSeenData: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on("last-seen:updated", listener);
    return () => ipcRenderer.removeListener("last-seen:updated", listener);
  },
  onMenuAction: (callback) => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on("menu:action", listener);
    return () => ipcRenderer.removeListener("menu:action", listener);
  },
});
