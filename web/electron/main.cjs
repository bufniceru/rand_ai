const { app, BrowserWindow, Menu, dialog, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const devServerUrl = process.env.VITE_DEV_SERVER_URL || "";
const maximumUploadBytes = 100 * 1024 * 1024;
const recentDatasetLimit = 10;
const bridgeProgressPrefix = "RAND_AI_PROGRESS ";
let mainWindow = null;
let lastSeenWindow = null;
let lastSeenGapWindow = null;
let combinedPredictionWindow = null;
let possibleDrawWindow = null;
let drawEditorWindow = null;
let activeDatasetPath = null;
let activeLastSeenData = null;
let activePredictionData = null;
let recentDatasets = [];
const reportPlugins = [
  { id: "overview", label: "Overview" },
  { id: "numbers", label: "Numbers" },
  { id: "spaces", label: "Spaces" },
  { id: "relationships", label: "Relationships" },
  { id: "randomness", label: "Randomness" },
  { id: "gaps", label: "Gaps" },
  { id: "last-seen", label: "Last Seen Highlight" },
  { id: "last-seen-gap", label: "Last Seen Gap Highlight" },
  { id: "predictions", label: "Predictions" },
  { id: "possible-draw", label: "Possible Draw" },
];
let enabledReportIds = new Set(reportPlugins.map((plugin) => plugin.id));
const strategyPlugins = [
  { id: "proximity", label: "Prox" },
  { id: "freshness", label: "Fresh" },
  { id: "emd", label: "EMD" },
  { id: "randomness", label: "Rand" },
  { id: "entropy", label: "Entr" },
  { id: "markov100", label: "Mark" },
  { id: "mkfr", label: "MKFR" },
  { id: "bayesian", label: "Baye" },
  { id: "svc", label: "SVC" },
  { id: "tbl", label: "TBL" },
];
let enabledStrategyIds = new Set(strategyPlugins.map((plugin) => plugin.id));

const dashboardViews = [
  ["overview", "Overview"],
  ["numbers", "Numbers"],
  ["spaces", "Spaces"],
  ["relationships", "Relationships"],
  ["randomness", "Randomness"],
  ["gaps", "Gaps"],
  ["export", "Export"],
];

function projectRoot() {
  return path.resolve(__dirname, "..", "..");
}

function recentDatasetsPath() {
  return path.join(app.getPath("userData"), "recent-datasets.json");
}

function reportPreferencesPath() {
  return path.join(app.getPath("userData"), "report-plugins.json");
}

function strategyPreferencesPath() {
  return path.join(app.getPath("userData"), "strategy-plugins.json");
}

function enabledReportsList() {
  return reportPlugins
    .map((plugin) => plugin.id)
    .filter((reportId) => enabledReportIds.has(reportId));
}

function reportPluginState() {
  return {
    plugins: reportPlugins.map((plugin) => ({
      ...plugin,
      enabled: enabledReportIds.has(plugin.id),
    })),
    enabledReports: enabledReportsList(),
  };
}

function loadReportPreferences() {
  try {
    const parsed = JSON.parse(fs.readFileSync(reportPreferencesPath(), "utf8"));
    if (!Array.isArray(parsed?.enabledReports)) {
      return new Set(reportPlugins.map((plugin) => plugin.id));
    }
    const knownIds = new Set(reportPlugins.map((plugin) => plugin.id));
    return new Set(
      parsed.enabledReports.filter(
        (reportId) => typeof reportId === "string" && knownIds.has(reportId),
      ),
    );
  } catch (error) {
    if (error?.code !== "ENOENT") {
      console.warn("Could not load report plugin preferences:", error);
    }
    return new Set(reportPlugins.map((plugin) => plugin.id));
  }
}

function saveReportPreferences() {
  try {
    fs.mkdirSync(path.dirname(reportPreferencesPath()), { recursive: true });
    fs.writeFileSync(
      reportPreferencesPath(),
      `${JSON.stringify({ enabledReports: enabledReportsList() }, null, 2)}\n`,
      "utf8",
    );
  } catch (error) {
    console.warn("Could not save report plugin preferences:", error);
  }
}

function enabledStrategiesList() {
  return strategyPlugins
    .map((plugin) => plugin.id)
    .filter((strategyId) => enabledStrategyIds.has(strategyId));
}

function strategyPluginState() {
  return {
    plugins: strategyPlugins.map((plugin) => ({
      ...plugin,
      enabled: enabledStrategyIds.has(plugin.id),
    })),
    enabledStrategies: enabledStrategiesList(),
  };
}

function loadStrategyPreferences() {
  try {
    const parsed = JSON.parse(fs.readFileSync(strategyPreferencesPath(), "utf8"));
    if (!Array.isArray(parsed?.enabledStrategies)) {
      return new Set(strategyPlugins.map((plugin) => plugin.id));
    }
    const knownIds = new Set(strategyPlugins.map((plugin) => plugin.id));
    return new Set(
      parsed.enabledStrategies.filter(
        (strategyId) =>
          typeof strategyId === "string" && knownIds.has(strategyId),
      ),
    );
  } catch (error) {
    if (error?.code !== "ENOENT") {
      console.warn("Could not load strategy plugin preferences:", error);
    }
    return new Set(strategyPlugins.map((plugin) => plugin.id));
  }
}

function saveStrategyPreferences() {
  try {
    fs.mkdirSync(path.dirname(strategyPreferencesPath()), { recursive: true });
    fs.writeFileSync(
      strategyPreferencesPath(),
      `${JSON.stringify(
        { enabledStrategies: enabledStrategiesList() },
        null,
        2,
      )}\n`,
      "utf8",
    );
  } catch (error) {
    console.warn("Could not save strategy plugin preferences:", error);
  }
}

function loadRecentDatasets() {
  try {
    const parsed = JSON.parse(fs.readFileSync(recentDatasetsPath(), "utf8"));
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (dataset) =>
          typeof dataset?.path === "string" &&
          typeof dataset?.name === "string" &&
          Number.isFinite(dataset?.sizeBytes),
      )
      .slice(0, recentDatasetLimit);
  } catch (error) {
    if (error?.code !== "ENOENT") {
      console.warn("Could not load recent datasets:", error);
    }
    return [];
  }
}

function saveRecentDatasets() {
  try {
    fs.mkdirSync(path.dirname(recentDatasetsPath()), { recursive: true });
    fs.writeFileSync(
      recentDatasetsPath(),
      `${JSON.stringify(recentDatasets, null, 2)}\n`,
      "utf8",
    );
  } catch (error) {
    console.warn("Could not save recent datasets:", error);
  }
}

function addRecentDataset(filePath, sizeBytes) {
  const resolvedPath = path.resolve(filePath);
  const comparisonPath =
    process.platform === "win32" ? resolvedPath.toLowerCase() : resolvedPath;
  recentDatasets = [
    {
      path: resolvedPath,
      name: path.basename(resolvedPath),
      sizeBytes,
      lastOpenedAt: new Date().toISOString(),
    },
    ...recentDatasets.filter((dataset) => {
      const existingPath =
        process.platform === "win32"
          ? dataset.path.toLowerCase()
          : dataset.path;
      return existingPath !== comparisonPath;
    }),
  ].slice(0, recentDatasetLimit);
  saveRecentDatasets();
}

function recentDatasetLabel(dataset) {
  const label = `${dataset.name} — ${path.dirname(dataset.path)}`;
  return label.length <= 96 ? label : `${label.slice(0, 93)}...`;
}

async function selectRecentDataset(dataset) {
  try {
    const stats = await fs.promises.stat(dataset.path);
    if (!stats.isFile()) {
      throw new Error("The recent dataset is no longer a file.");
    }
    if (stats.size > maximumUploadBytes) {
      throw new Error("Pickle file must not exceed 100 MiB.");
    }
    sendMenuAction("datasetSelected", {
      dataset: {
        path: dataset.path,
        name: path.basename(dataset.path),
        sizeBytes: stats.size,
      },
    });
  } catch (error) {
    recentDatasets = recentDatasets.filter(
      (recentDataset) => recentDataset.path !== dataset.path,
    );
    saveRecentDatasets();
    buildApplicationMenu();
    dialog.showErrorBox(
      "Could not open recent dataset",
      `${dataset.path}\n\n${error.message}`,
    );
  }
}

function bridgeInvocation(argumentsList) {
  if (app.isPackaged) {
    return {
      executable: path.join(process.resourcesPath, "bridge", "rand-ai-bridge.exe"),
      arguments: argumentsList,
      cwd: path.dirname(app.getPath("exe")),
    };
  }

  const configuredPython = process.env.RAND_AI_PYTHON;
  const localPython = path.join(projectRoot(), ".venv", "Scripts", "python.exe");
  const executable = configuredPython || (fs.existsSync(localPython) ? localPython : "python");
  return {
    executable,
    arguments: ["-m", "rand_ai.gui_bridge", ...argumentsList],
    cwd: projectRoot(),
  };
}

function forwardBridgeProgress(line, onProgress) {
  if (!onProgress || !line.startsWith(bridgeProgressPrefix)) return;
  try {
    const progress = JSON.parse(line.slice(bridgeProgressPrefix.length));
    if (Number.isFinite(progress?.percent) && typeof progress?.message === "string") {
      onProgress(progress);
    }
  } catch (error) {
    console.warn("Could not parse bridge progress:", error);
  }
}

function runBridge(argumentsList, expectJson, onProgress = null) {
  return new Promise((resolve, reject) => {
    const invocation = bridgeInvocation(argumentsList);
    const child = spawn(invocation.executable, invocation.arguments, {
      cwd: invocation.cwd,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    let progressBuffer = "";

    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => {
      stderr.push(chunk);
      progressBuffer += chunk.toString("utf8");
      let newlineIndex = progressBuffer.indexOf("\n");
      while (newlineIndex >= 0) {
        forwardBridgeProgress(progressBuffer.slice(0, newlineIndex).trim(), onProgress);
        progressBuffer = progressBuffer.slice(newlineIndex + 1);
        newlineIndex = progressBuffer.indexOf("\n");
      }
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => {
      if (progressBuffer.trim()) {
        forwardBridgeProgress(progressBuffer.trim(), onProgress);
      }
      const output = Buffer.concat(stdout).toString("utf8");
      const errorOutput = Buffer.concat(stderr)
        .toString("utf8")
        .split(/\r?\n/)
        .filter((line) => !line.startsWith(bridgeProgressPrefix))
        .join("\n")
        .trim();
      if (code !== 0) {
        reject(new Error(errorOutput || `Analysis bridge exited with code ${code}.`));
        return;
      }
      if (!expectJson) {
        resolve(null);
        return;
      }
      try {
        resolve(JSON.parse(output));
      } catch {
        reject(new Error("The analysis bridge returned invalid JSON."));
      }
    });
  });
}

function sendMenuAction(action, payload = {}) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("menu:action", { action, ...payload });
  }
}

function closeDisabledReportWindows() {
  const windows = [
    ["last-seen", lastSeenWindow],
    ["last-seen-gap", lastSeenGapWindow],
    ["predictions", combinedPredictionWindow],
    ["possible-draw", possibleDrawWindow],
  ];
  for (const [reportId, reportWindow] of windows) {
    if (
      !enabledReportIds.has(reportId) &&
      reportWindow &&
      !reportWindow.isDestroyed()
    ) {
      reportWindow.close();
    }
  }
}

function updateReportPlugins(nextEnabledReportIds) {
  enabledReportIds = new Set(nextEnabledReportIds);
  saveReportPreferences();
  closeDisabledReportWindows();
  buildApplicationMenu();
  sendMenuAction("reportPluginsChanged", reportPluginState());
}

function updateReportPlugin(reportId, enabled) {
  const nextEnabledReportIds = new Set(enabledReportIds);
  if (enabled) nextEnabledReportIds.add(reportId);
  else nextEnabledReportIds.delete(reportId);
  updateReportPlugins(nextEnabledReportIds);
}

function updateStrategyPlugins(nextEnabledStrategyIds) {
  enabledStrategyIds = new Set(nextEnabledStrategyIds);
  saveStrategyPreferences();
  return strategyPluginState();
}

function lastSeenDialogData() {
  return activeLastSeenData;
}

function combinedPredictionDialogData() {
  return activePredictionData;
}

function loadRenderer(window, query = {}) {
  if (devServerUrl) {
    const url = new URL(devServerUrl);
    Object.entries(query).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
    window.loadURL(url.toString());
    return;
  }
  window.loadFile(path.join(__dirname, "..", "dist", "index.html"), { query });
}

function createLastSeenWindow() {
  if (!enabledReportIds.has("last-seen")) return;
  if (!activeLastSeenData) {
    dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Last Seen Highlight",
      message: "Analyze a dataset first.",
      detail: "The Last Seen Highlight dialog uses the active draw history.",
    });
    return;
  }
  if (lastSeenWindow && !lastSeenWindow.isDestroyed()) {
    if (lastSeenWindow.isMinimized()) lastSeenWindow.restore();
    lastSeenWindow.focus();
    return;
  }

  lastSeenWindow = new BrowserWindow({
    parent: mainWindow,
    title: "Last Seen Highlight — Rand AI",
    width: 1440,
    height: 920,
    minWidth: 980,
    minHeight: 680,
    backgroundColor: "#eef3f7",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  lastSeenWindow.once("ready-to-show", () => {
    lastSeenWindow.maximize();
    lastSeenWindow.show();
  });
  lastSeenWindow.on("closed", () => {
    lastSeenWindow = null;
  });
  loadRenderer(lastSeenWindow, { window: "last-seen" });
}

function createLastSeenGapWindow() {
  if (!enabledReportIds.has("last-seen-gap")) return;
  if (!activeLastSeenData) {
    dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Last Seen Gap Highlight",
      message: "Analyze a dataset first.",
      detail: "The Last Seen Gap Highlight dialog uses the active draw history.",
    });
    return;
  }
  if (lastSeenGapWindow && !lastSeenGapWindow.isDestroyed()) {
    if (lastSeenGapWindow.isMinimized()) lastSeenGapWindow.restore();
    lastSeenGapWindow.focus();
    return;
  }

  lastSeenGapWindow = new BrowserWindow({
    parent: mainWindow,
    title: "Last Seen Gap Highlight — Rand AI",
    width: 1440,
    height: 920,
    minWidth: 980,
    minHeight: 680,
    backgroundColor: "#eef3f7",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  lastSeenGapWindow.once("ready-to-show", () => {
    lastSeenGapWindow.maximize();
    lastSeenGapWindow.show();
  });
  lastSeenGapWindow.on("closed", () => {
    lastSeenGapWindow = null;
  });
  loadRenderer(lastSeenGapWindow, { window: "last-seen-gap" });
}

function createCombinedPredictionWindow() {
  if (!enabledReportIds.has("predictions")) return;
  if (!activePredictionData) {
    dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Predictions",
      message: "Analyze a dataset first.",
      detail: "The named predictions are calculated while the dataset is imported.",
    });
    return;
  }
  if (combinedPredictionWindow && !combinedPredictionWindow.isDestroyed()) {
    if (combinedPredictionWindow.isMinimized()) combinedPredictionWindow.restore();
    combinedPredictionWindow.focus();
    return;
  }

  combinedPredictionWindow = new BrowserWindow({
    parent: mainWindow,
    title: "Predictions — Rand AI",
    width: 1120,
    height: 920,
    minWidth: 760,
    minHeight: 680,
    backgroundColor: "#eef3f7",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  combinedPredictionWindow.once("ready-to-show", () => {
    combinedPredictionWindow.maximize();
    combinedPredictionWindow.show();
  });
  combinedPredictionWindow.on("closed", () => {
    combinedPredictionWindow = null;
  });
  loadRenderer(combinedPredictionWindow, { window: "combined-prediction" });
}

function createPossibleDrawWindow() {
  if (!enabledReportIds.has("possible-draw")) return;
  if (!activePredictionData) {
    dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Possible Draw",
      message: "Analyze a dataset first.",
      detail: "Possible Draw uses the Python-calculated prediction strategies.",
    });
    return;
  }
  if (possibleDrawWindow && !possibleDrawWindow.isDestroyed()) {
    if (possibleDrawWindow.isMinimized()) possibleDrawWindow.restore();
    possibleDrawWindow.focus();
    return;
  }

  possibleDrawWindow = new BrowserWindow({
    parent: mainWindow,
    title: "Possible Draw — Rand AI",
    width: 1480,
    height: 960,
    minWidth: 1060,
    minHeight: 720,
    backgroundColor: "#eef3f7",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  possibleDrawWindow.once("ready-to-show", () => {
    possibleDrawWindow.maximize();
    possibleDrawWindow.show();
  });
  possibleDrawWindow.on("closed", () => {
    possibleDrawWindow = null;
  });
  loadRenderer(possibleDrawWindow, { window: "possible-draw" });
}

function createDrawEditorWindow() {
  if (!activeDatasetPath) {
    dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Draw History",
      message: "Analyze a YAML-managed dataset first.",
      detail: "The editor updates the paired YAML first and then rebuilds its pickle.",
    });
    return;
  }
  if (drawEditorWindow && !drawEditorWindow.isDestroyed()) {
    if (drawEditorWindow.isMinimized()) drawEditorWindow.restore();
    drawEditorWindow.focus();
    return;
  }
  drawEditorWindow = new BrowserWindow({
    parent: mainWindow,
    title: "Draw History — Rand AI",
    width: 1040,
    height: 900,
    minWidth: 760,
    minHeight: 680,
    backgroundColor: "#eef3f7",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  drawEditorWindow.once("ready-to-show", () => {
    drawEditorWindow.show();
  });
  drawEditorWindow.on("closed", () => {
    drawEditorWindow = null;
  });
  loadRenderer(drawEditorWindow, { window: "draw-editor" });
}

async function chooseDataset() {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Open trusted Draws dataset",
    properties: ["openFile"],
    filters: [
      { name: "Draws pickle", extensions: ["pkl", "pickle"] },
      { name: "All files", extensions: ["*"] },
    ],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  const filePath = result.filePaths[0];
  const stats = await fs.promises.stat(filePath);
  if (stats.size > maximumUploadBytes) {
    throw new Error("Pickle file must not exceed 100 MiB.");
  }
  return {
    path: filePath,
    name: path.basename(filePath),
    sizeBytes: stats.size,
  };
}

function optionArguments(options = {}) {
  const selectedNumbers = Array.isArray(options.selectedNumbers)
    ? options.selectedNumbers.join(",")
    : "1,2,3,4,5,6";
  return [
    "--selected-numbers",
    selectedNumbers,
    "--trend-bins",
    String(options.trendBins ?? 100),
    "--correlation-method",
    String(options.correlationMethod ?? "pearson"),
    "--reports",
    enabledReportsList().join(","),
    "--strategies",
    enabledStrategiesList().join(","),
  ];
}

function buildApplicationMenu() {
  const recentDatasetItems =
    recentDatasets.length === 0
      ? [{ label: "No Recent Datasets", enabled: false }]
      : [
          ...recentDatasets.map((dataset) => ({
            label: recentDatasetLabel(dataset),
            click: () => selectRecentDataset(dataset),
          })),
          { type: "separator" },
          {
            label: "Clear Recent Datasets",
            click: () => {
              recentDatasets = [];
              saveRecentDatasets();
              buildApplicationMenu();
            },
          },
        ];
  const template = [
    {
      label: "File",
      submenu: [
        {
          label: "Open Dataset...",
          accelerator: "CmdOrCtrl+O",
          click: async () => {
            try {
              const dataset = await chooseDataset();
              if (dataset) sendMenuAction("datasetSelected", { dataset });
            } catch (error) {
              dialog.showErrorBox("Could not open dataset", error.message);
            }
          },
        },
        {
          label: "Recent Datasets",
          submenu: recentDatasetItems,
        },
        {
          label: "Export Analysis...",
          accelerator: "CmdOrCtrl+Shift+E",
          enabled: activeDatasetPath !== null,
          click: () => sendMenuAction("export"),
        },
        { type: "separator" },
        {
          label: "Settings...",
          accelerator: "CmdOrCtrl+,",
          click: () => sendMenuAction("openSettings"),
        },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "View",
      submenu: [
        ...dashboardViews.map(([id, label]) => ({
          label,
          enabled:
            activeDatasetPath !== null &&
            (id === "export" || enabledReportIds.has(id)),
          click: () => sendMenuAction("openView", { view: id }),
        })),
        {
          label: "Last Seen Highlight...",
          accelerator: "CmdOrCtrl+Shift+L",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("last-seen"),
          click: () => createLastSeenWindow(),
        },
        {
          label: "Last Seen Gap Highlight...",
          accelerator: "CmdOrCtrl+Shift+G",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("last-seen-gap"),
          click: () => createLastSeenGapWindow(),
        },
        {
          label: "Predictions...",
          accelerator: "CmdOrCtrl+Shift+P",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("predictions"),
          click: () => createCombinedPredictionWindow(),
        },
        {
          label: "Possible Draw...",
          accelerator: "CmdOrCtrl+Shift+D",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("possible-draw"),
          click: () => createPossibleDrawWindow(),
        },
        {
          label: "Draw History Editor...",
          accelerator: "CmdOrCtrl+Shift+H",
          enabled: activeDatasetPath !== null,
          click: () => createDrawEditorWindow(),
        },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Reports",
      submenu: [
        ...reportPlugins.flatMap((plugin) => {
          const item = {
            label: plugin.label,
            type: "checkbox",
            checked: enabledReportIds.has(plugin.id),
            click: (menuItem) =>
              updateReportPlugin(plugin.id, menuItem.checked),
          };
          return plugin.id === "gaps"
            ? [{ type: "separator" }, item, { type: "separator" }]
            : [item];
        }),
        { type: "separator" },
        {
          label: "Enable All Reports",
          enabled: enabledReportIds.size !== reportPlugins.length,
          click: () =>
            updateReportPlugins(reportPlugins.map((plugin) => plugin.id)),
        },
        {
          label: "Disable All Reports",
          enabled: enabledReportIds.size !== 0,
          click: () => updateReportPlugins([]),
        },
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "About Rand AI",
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: "info",
              title: "About Rand AI",
              message: "Rand AI",
              detail:
                "Vue 3 and Electron desktop statistics dashboard backed by the Python analysis engine.",
            });
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    title: "Rand AI",
    width: 1480,
    height: 980,
    minWidth: 1080,
    minHeight: 720,
    backgroundColor: "#eef3f7",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.once("ready-to-show", () => {
    mainWindow.maximize();
    mainWindow.show();
  });
  loadRenderer(mainWindow);
}

ipcMain.handle("dataset:open", async () => chooseDataset());

ipcMain.handle("report-plugins:get", () => reportPluginState());

ipcMain.handle("strategy-plugins:get", () => strategyPluginState());

ipcMain.handle("strategy-plugins:set", (_event, requestedStrategyIds) => {
  if (!Array.isArray(requestedStrategyIds)) {
    throw new TypeError("Strategy selection must be an array.");
  }
  const requested = new Set(requestedStrategyIds);
  return updateStrategyPlugins(
    strategyPlugins
      .map((plugin) => plugin.id)
      .filter((strategyId) => requested.has(strategyId)),
  );
});

ipcMain.handle("dataset:analyze", async (event, request) => {
  const sendProgress = (progress) => {
    if (!event.sender.isDestroyed()) {
      event.sender.send("dataset:analysis-progress", progress);
    }
  };
  sendProgress({ percent: 2, message: "Checking the selected file and its size" });
  const filePath = path.resolve(String(request?.path ?? ""));
  const extension = path.extname(filePath).toLowerCase();
  if (![".pkl", ".pickle"].includes(extension)) {
    throw new Error("Select a .pkl or .pickle Draws dataset.");
  }
  const stats = await fs.promises.stat(filePath);
  if (stats.size > maximumUploadBytes) {
    throw new Error("Pickle file must not exceed 100 MiB.");
  }
  sendProgress({ percent: 3, message: "Starting the Python analysis engine" });
  const payload = await runBridge(
    ["analyze", "--input", filePath, ...optionArguments(request?.options)],
    true,
    sendProgress,
  );
  sendProgress({ percent: 98, message: "Updating highlight data and application menus" });
  activeDatasetPath = filePath;
  addRecentDataset(filePath, stats.size);
  const activeReports = new Set(payload.options.enabledReports);
  activeLastSeenData =
    activeReports.has("last-seen") || activeReports.has("last-seen-gap")
      ? {
          dataset: payload.dataset,
          history: payload.history,
        }
      : null;
  activePredictionData =
    activeReports.has("predictions") || activeReports.has("possible-draw")
      ? {
          dataset: payload.dataset,
          predictions: payload.combinedPredictions,
          predictionSuites: payload.predictionSuites,
          history: payload.history,
          possibleDraw: payload.possibleDraw,
        }
      : null;
  if (lastSeenWindow && !lastSeenWindow.isDestroyed()) {
    lastSeenWindow.webContents.send("last-seen:updated", lastSeenDialogData());
  }
  if (lastSeenGapWindow && !lastSeenGapWindow.isDestroyed()) {
    lastSeenGapWindow.webContents.send("last-seen:updated", lastSeenDialogData());
  }
  if (combinedPredictionWindow && !combinedPredictionWindow.isDestroyed()) {
    combinedPredictionWindow.webContents.send(
      "combined-prediction:updated",
      combinedPredictionDialogData(),
    );
  }
  if (possibleDrawWindow && !possibleDrawWindow.isDestroyed()) {
    possibleDrawWindow.webContents.send(
      "combined-prediction:updated",
      combinedPredictionDialogData(),
    );
  }
  buildApplicationMenu();
  sendProgress({ percent: 100, message: "Analysis ready" });
  return payload;
});

ipcMain.handle("last-seen:open", () => {
  createLastSeenWindow();
});

ipcMain.handle("last-seen-gap:open", () => {
  createLastSeenGapWindow();
});

ipcMain.handle("last-seen:data", () => lastSeenDialogData());

ipcMain.handle("combined-prediction:open", () => {
  createCombinedPredictionWindow();
});

ipcMain.handle("possible-draw:open", () => {
  createPossibleDrawWindow();
});

ipcMain.handle("possible-draw:add-number", (_event, request) => {
  const number = Number(request?.number);
  const state = request?.state;
  if (!Number.isInteger(number) || number < 1 || number > 49) {
    throw new RangeError("Possible Draw numbers must be integers from 1 through 49.");
  }
  if (!["possible", "for-sure"].includes(state)) {
    throw new TypeError("Possible Draw state must be possible or for-sure.");
  }
  if (!enabledReportIds.has("possible-draw")) {
    dialog.showMessageBox(combinedPredictionWindow ?? mainWindow, {
      type: "error",
      title: "Possible Draw unavailable",
      message: "Possible Draw is disabled.",
      detail: "Enable Possible Draw in Settings before sending prediction numbers.",
    });
    return;
  }
  if (!activePredictionData) {
    dialog.showMessageBox(combinedPredictionWindow ?? mainWindow, {
      type: "error",
      title: "Possible Draw unavailable",
      message: "Analyze a dataset first.",
      detail: "Prediction numbers can be sent after the active dataset is analyzed.",
    });
    return;
  }

  createPossibleDrawWindow();
  const target = possibleDrawWindow;
  if (!target || target.isDestroyed()) return;
  const deliver = () => {
    if (!target.isDestroyed()) {
      target.webContents.send("possible-draw:number-requested", { number, state });
      if (target.isMinimized()) target.restore();
      target.focus();
    }
  };
  if (target.webContents.isLoading()) {
    target.webContents.once("did-finish-load", deliver);
  } else {
    deliver();
  }
});

ipcMain.handle("possible-draw:for-sure-limit-error", (event, requestedNumber) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? possibleDrawWindow ?? mainWindow;
  return dialog.showMessageBox(parent, {
    type: "error",
    title: "For Sure limit reached",
    message: "No more than six numbers can be marked For Sure.",
    detail: `Number ${requestedNumber} was not added. Remove a For Sure number before trying again.`,
  });
});

ipcMain.handle("draw-editor:open", () => {
  createDrawEditorWindow();
});

ipcMain.handle("draw-editor:data", async () => {
  if (!activeDatasetPath) {
    throw new Error("Analyze a YAML-managed dataset first.");
  }
  return runBridge(["draw-editor", "--input", activeDatasetPath], true);
});

ipcMain.handle("draw-editor:save", async (_event, request) => {
  if (!activeDatasetPath) {
    throw new Error("Analyze a YAML-managed dataset first.");
  }
  const numbers = Array.isArray(request?.numbers) ? request.numbers.join(",") : "";
  const argumentsList = [
    "draw-save",
    "--input",
    activeDatasetPath,
    "--date",
    String(request?.date ?? ""),
    "--numbers",
    numbers,
  ];
  if (request?.originalDate) {
    argumentsList.push("--original-date", String(request.originalDate));
  }
  const payload = await runBridge(argumentsList, true);
  const stats = await fs.promises.stat(activeDatasetPath);
  addRecentDataset(activeDatasetPath, stats.size);
  sendMenuAction("datasetSelected", {
    dataset: {
      path: activeDatasetPath,
      name: path.basename(activeDatasetPath),
      sizeBytes: stats.size,
    },
  });
  return payload;
});

ipcMain.handle("combined-prediction:data", () => combinedPredictionDialogData());

ipcMain.handle("analysis:export", async (_event, request) => {
  if (!activeDatasetPath) {
    throw new Error("Open and analyze a dataset before exporting.");
  }
  const result = await dialog.showSaveDialog(mainWindow, {
    title: "Export analysis",
    defaultPath: "draws-statistics.zip",
    filters: [{ name: "ZIP archive", extensions: ["zip"] }],
  });
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }
  await runBridge(
    [
      "export",
      "--input",
      activeDatasetPath,
      "--output",
      result.filePath,
      ...optionArguments(request?.options),
    ],
    false,
  );
  return { canceled: false, path: result.filePath };
});

app.whenReady().then(() => {
  recentDatasets = loadRecentDatasets();
  enabledReportIds = loadReportPreferences();
  enabledStrategyIds = loadStrategyPreferences();
  createWindow();
  buildApplicationMenu();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
