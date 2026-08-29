const { app, BrowserWindow, Menu, dialog, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { promisify } = require("node:util");
const zlib = require("node:zlib");

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

const devServerUrl = process.env.VITE_DEV_SERVER_URL || "";
const maximumUploadBytes = 100 * 1024 * 1024;
const maximumColorTemplateBytes = 1024 * 1024;
const recentDatasetLimit = 10;
const bridgeProgressPrefix = "RAND_AI_PROGRESS ";
let mainWindow = null;
let activeDatasetPath = null;
let recentDatasets = [];
const reportPlugins = [
  { id: "overview", label: "Overview" },
  { id: "numbers", label: "Numbers" },
  { id: "spaces", label: "Spaces" },
  { id: "space-groups", label: "Border Groups" },
  { id: "relationships", label: "Relationships" },
  { id: "randomness", label: "Randomness" },
  { id: "nonlinear-dynamics", label: "Nonlinear Dynamics" },
  { id: "autocorrelation", label: "Autocorrelation" },
  { id: "co-occurrence", label: "Co-occurrence" },
  { id: "prediction-audit", label: "Prediction Audit" },
  { id: "draw-comparison", label: "Latest Draw vs Predictions" },
  { id: "strategy-effectiveness", label: "Strategy Effectiveness" },
  { id: "gaps", label: "Gaps" },
  { id: "last-seen", label: "Last Seen Highlight" },
  { id: "last-seen-gap", label: "Last Seen Gap Highlight" },
  { id: "last-seen-space", label: "Last Seen Space Highlight" },
  { id: "predictions", label: "Predictions" },
  { id: "draw-portfolio", label: "Draw Portfolio" },
  { id: "possible-draw", label: "Possible Draw" },
];
const legacyReportPluginIds = reportPlugins
  .map((plugin) => plugin.id)
  .filter(
    (reportId) =>
      reportId !== "nonlinear-dynamics" &&
      reportId !== "autocorrelation" &&
      reportId !== "co-occurrence" &&
      reportId !== "prediction-audit" &&
      reportId !== "draw-comparison" &&
      reportId !== "strategy-effectiveness" &&
      reportId !== "last-seen-space" &&
      reportId !== "draw-portfolio",
  );
let enabledReportIds = new Set(reportPlugins.map((plugin) => plugin.id));
const legacyStrategyPluginIds = [
  "proximity",
  "freshness",
  "emd",
  "randomness",
  "entropy",
  "markov100",
  "mkfr",
  "bayesian",
  "svc",
  "tbl",
];
const strategyPlugins = [
  { id: "proximity", label: "Proximity" },
  { id: "freshness", label: "Freshness" },
  { id: "emd", label: "Earth Mover Distance" },
  {
    id: "recurrence_dynamics",
    label: "Recurrence Dynamics (Experimental)",
  },
  { id: "randomness", label: "Random baseline" },
  { id: "fresh_random", label: "Fresh Random" },
  { id: "chi_square", label: "Chi-square Frequency" },
  { id: "categorical_chi_square", label: "Categorical Chi-square" },
  { id: "entropy", label: "Entropy" },
  { id: "markov100", label: "Markov 100" },
  { id: "mkgsv", label: "Markov Gap-Space Vector (Experimental)" },
  { id: "mkfr", label: "Markov Freshness" },
  { id: "mksp", label: "Markov Spaces" },
  { id: "mknp", label: "Markov Normalized Positions" },
  { id: "mkrd", label: "Markov Relative Dispersion" },
  { id: "bayesian", label: "Bayesian" },
  { id: "predictive_grid", label: "Predictive Score Grid" },
  { id: "co_occurrence", label: "Next Draw Co-occurrence" },
  {
    id: "doublet_triplet_markov",
    label: "Doublet & Triplet Markov",
  },
  { id: "mixed", label: "Mixed Prediction" },
  { id: "svc", label: "Support Vector Classifier" },
  {
    id: "svc_recurrence_hybrid",
    label: "SVC–Recurrence Hybrid (Experimental)",
  },
  {
    id: "svc_recurrence_proximity_hybrid",
    label: "SVC–Recurrence–Proximity Hybrid (Experimental)",
  },
  {
    id: "srph_residual_diversity_hybrid",
    label: "SRPH Residual Diversity Hybrid (Experimental)",
  },
  { id: "tbl", label: "Temporal Behavior Learning" },
  { id: "sklearn_svm", label: "Scikit Online SVM" },
  { id: "lag_logistic", label: "Lagged Logistic" },
  {
    id: "sparse_neural_ticket",
    label: "Sparse Neural Ticket (Experimental)",
  },
  { id: "cis", label: "Collective Intelligence Strategy" },
  { id: "decision_tree_selector", label: "Decision Tree Selector" },
  { id: "border_group_statistical", label: "Border Group Statistical" },
  { id: "border_group_markov", label: "Border Group Markov" },
  { id: "border_group_bayesian", label: "Border Group Bayesian" },
  { id: "border_group_ml", label: "Border Group ML" },
  { id: "border_group_hybrid", label: "Border Group Hybrid" },
  { id: "residual_coverage", label: "Residual Coverage" },
  { id: "chained", label: "Chained Strategy" },
];
const defaultStrategyPluginIds = strategyPlugins
  .map((plugin) => plugin.id)
  .filter(
    (strategyId) =>
      strategyId !== "mkrd" &&
      strategyId !== "mkgsv" &&
      strategyId !== "sklearn_svm" &&
      strategyId !== "lag_logistic" &&
      strategyId !== "sparse_neural_ticket" &&
      strategyId !== "decision_tree_selector" &&
      strategyId !== "recurrence_dynamics" &&
      strategyId !== "svc_recurrence_hybrid" &&
      strategyId !== "svc_recurrence_proximity_hybrid" &&
      strategyId !== "srph_residual_diversity_hybrid",
  );
let enabledStrategyIds = new Set(defaultStrategyPluginIds);

const dashboardViews = [
  ["overview", "Overview"],
  ["numbers", "Numbers"],
  ["spaces", "Spaces"],
  ["space-groups", "Border Groups"],
  ["relationships", "Relationships"],
  ["randomness", "Randomness"],
  ["nonlinear-dynamics", "Nonlinear Dynamics"],
  ["autocorrelation", "Autocorrelation"],
  ["co-occurrence", "Co-occurrence"],
  ["prediction-audit", "Prediction Audit"],
  ["draw-comparison", "Latest Draw vs Predictions"],
  ["strategy-effectiveness", "Strategy Effectiveness"],
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

function analysisCachePath() {
  return path.join(app.getPath("userData"), "analysis-cache");
}

function portfolioBacktestCachePath() {
  return path.join(app.getPath("userData"), "portfolio-backtests");
}

function colorTemplatePreferencesPath() {
  return path.join(app.getPath("userData"), "active-color-template.json");
}

function validatedColorTemplate(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The selected file does not contain a color template.");
  }
  if (value.kind !== "rand-ai-color-template") {
    throw new Error("Template kind must be rand-ai-color-template.");
  }
  if (value.schemaVersion !== 1) {
    throw new Error(
      `Unsupported color-template schema version: ${String(value.schemaVersion)}.`,
    );
  }
  if (
    typeof value.name !== "string" ||
    value.name.trim().length === 0 ||
    value.name.length > 80
  ) {
    throw new Error("Template name must contain 1 to 80 characters.");
  }
  if (
    value.description !== undefined &&
    (typeof value.description !== "string" || value.description.length > 500)
  ) {
    throw new Error("Template description must contain at most 500 characters.");
  }
  if (!value.colors || typeof value.colors !== "object" || Array.isArray(value.colors)) {
    throw new Error("Template colors must be an object.");
  }
  const entries = Object.entries(value.colors);
  if (entries.length === 0 || entries.length > 5000) {
    throw new Error("Template must contain between 1 and 5,000 color tokens.");
  }
  for (const [id, color] of entries) {
    if (!/^[a-zA-Z0-9_.-]{1,160}$/.test(id)) {
      throw new Error(`Invalid color token identifier: ${id}`);
    }
    if (typeof color !== "string" || !/^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$/.test(color)) {
      throw new Error(`Color token ${id} must use #RRGGBB or #RRGGBBAA.`);
    }
  }
  const normalized = {
    kind: "rand-ai-color-template",
    schemaVersion: 1,
    name: value.name.trim(),
    colors: Object.fromEntries(
      entries.map(([id, color]) => [id, color.toUpperCase()]),
    ),
  };
  if (typeof value.description === "string" && value.description.trim()) {
    normalized.description = value.description.trim();
  }
  for (const key of ["createdWith", "exportedAt"]) {
    if (typeof value[key] === "string" && value[key].length <= 100) {
      normalized[key] = value[key];
    }
  }
  return normalized;
}

function loadStoredColorTemplate() {
  try {
    const raw = fs.readFileSync(colorTemplatePreferencesPath(), "utf8");
    if (Buffer.byteLength(raw, "utf8") > maximumColorTemplateBytes) {
      throw new Error("Stored color template exceeds 1 MiB.");
    }
    return validatedColorTemplate(JSON.parse(raw));
  } catch (error) {
    if (error?.code !== "ENOENT") {
      console.warn("Could not load the active color template:", error);
    }
    return null;
  }
}

async function writeJsonAtomically(filePath, value) {
  const contents = `${JSON.stringify(value, null, 2)}\n`;
  if (Buffer.byteLength(contents, "utf8") > maximumColorTemplateBytes) {
    throw new Error("Color template must not exceed 1 MiB.");
  }
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${crypto.randomUUID()}.tmp`;
  try {
    await fs.promises.writeFile(temporaryPath, contents, "utf8");
    try {
      await fs.promises.rename(temporaryPath, filePath);
    } catch (error) {
      if (!["EEXIST", "EPERM"].includes(error?.code)) throw error;
      await fs.promises.copyFile(temporaryPath, filePath);
    }
  } finally {
    await fs.promises.unlink(temporaryPath).catch(() => undefined);
  }
}

function safeColorTemplateStem(name) {
  return String(name ?? "rand-ai-template")
    .replace(/\.randai-theme\.json$/i, "")
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "rand-ai-template";
}

function validatedPortfolioCacheKey(value) {
  const key = String(value ?? "");
  if (!/^[a-f0-9]{64}-v\d+-p(?:[1-9]|[1-9]\d|100)$/.test(key)) {
    throw new Error("Invalid portfolio backtest cache key.");
  }
  return key;
}

function validPortfolioBacktestResult(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    Number.isInteger(value.algorithmVersion) &&
    Number.isInteger(value.portfolioSize) &&
    value.portfolioSize >= 1 &&
    value.portfolioSize <= 100 &&
    Number.isInteger(value.evaluatedTargets) &&
    Array.isArray(value.buckets) &&
    value.buckets.length === 7 &&
    Array.isArray(value.audit) &&
    value.audit.length === value.evaluatedTargets
  );
}

async function prunePortfolioBacktestCache() {
  const cachePath = portfolioBacktestCachePath();
  const entries = await fs.promises.readdir(cachePath, { withFileTypes: true });
  const files = await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json.gz"))
      .map(async (entry) => {
        const filePath = path.join(cachePath, entry.name);
        const stats = await fs.promises.stat(filePath);
        return { filePath, modified: stats.mtimeMs };
      }),
  );
  files.sort((left, right) => right.modified - left.modified);
  await Promise.all(
    files.slice(20).map((entry) => fs.promises.unlink(entry.filePath)),
  );
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
    const selected = new Set(
      parsed.enabledReports.filter(
        (reportId) => typeof reportId === "string" && knownIds.has(reportId),
      ),
    );
    if (legacyReportPluginIds.every((reportId) => selected.has(reportId))) {
      for (const plugin of reportPlugins) selected.add(plugin.id);
    }
    return selected;
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
      return new Set(defaultStrategyPluginIds);
    }
    const knownIds = new Set(strategyPlugins.map((plugin) => plugin.id));
    const selected = new Set(
      parsed.enabledStrategies.filter(
        (strategyId) =>
          typeof strategyId === "string" && knownIds.has(strategyId),
      ),
    );
    if (legacyStrategyPluginIds.every((strategyId) => selected.has(strategyId))) {
      for (const strategyId of defaultStrategyPluginIds) selected.add(strategyId);
    }
    return selected;
  } catch (error) {
    if (error?.code !== "ENOENT") {
      console.warn("Could not load strategy plugin preferences:", error);
    }
    return new Set(defaultStrategyPluginIds);
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

function recentDatasetPayload(dataset) {
  const extension = path.extname(dataset.path).toLowerCase();
  return {
    path: dataset.path,
    name: dataset.name,
    sizeBytes: dataset.sizeBytes,
    requiresTrust: ![".yaml", ".yml"].includes(extension),
    lastOpenedAt:
      typeof dataset.lastOpenedAt === "string" ? dataset.lastOpenedAt : undefined,
  };
}

async function resolveRecentDataset(requestedPath) {
  const resolvedPath = path.resolve(String(requestedPath ?? ""));
  const comparisonPath =
    process.platform === "win32" ? resolvedPath.toLowerCase() : resolvedPath;
  const dataset = recentDatasets.find((candidate) => {
    const candidatePath =
      process.platform === "win32"
        ? candidate.path.toLowerCase()
        : candidate.path;
    return candidatePath === comparisonPath;
  });
  if (!dataset) {
    throw new Error("This dataset is not in the recent-files list.");
  }
  try {
    const stats = await fs.promises.stat(dataset.path);
    if (!stats.isFile()) {
      throw new Error("The recent dataset is no longer a file.");
    }
    if (stats.size > maximumUploadBytes) {
      throw new Error("Dataset file must not exceed 100 MiB.");
    }
    const extension = path.extname(dataset.path).toLowerCase();
    if (![".yaml", ".yml", ".pkl", ".pickle"].includes(extension)) {
      throw new Error("The recent file is not a supported Draws dataset.");
    }
    return {
      path: dataset.path,
      name: path.basename(dataset.path),
      sizeBytes: stats.size,
      requiresTrust: ![".yaml", ".yml"].includes(extension),
    };
  } catch (error) {
    recentDatasets = recentDatasets.filter(
      (recentDataset) => recentDataset.path !== dataset.path,
    );
    saveRecentDatasets();
    buildApplicationMenu();
    throw error;
  }
}

async function selectRecentDataset(dataset) {
  try {
    sendMenuAction("datasetSelected", {
      dataset: await resolveRecentDataset(dataset.path),
    });
  } catch (error) {
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

function updateReportPlugins(nextEnabledReportIds) {
  enabledReportIds = new Set(nextEnabledReportIds);
  saveReportPreferences();
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

async function chooseDataset() {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Open Draws YAML or trusted pickle",
    properties: ["openFile"],
    filters: [
      { name: "Draws YAML or pickle", extensions: ["yaml", "yml", "pkl", "pickle"] },
      { name: "Draws YAML", extensions: ["yaml", "yml"] },
      { name: "Draws pickle", extensions: ["pkl", "pickle"] },
      { name: "All files", extensions: ["*"] },
    ],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  const filePath = result.filePaths[0];
  const extension = path.extname(filePath).toLowerCase();
  if (![".yaml", ".yml", ".pkl", ".pickle"].includes(extension)) {
    throw new Error("Select a .yaml, .yml, .pkl, or .pickle Draws dataset.");
  }
  const stats = await fs.promises.stat(filePath);
  if (stats.size > maximumUploadBytes) {
    throw new Error("Dataset file must not exceed 100 MiB.");
  }
  return {
    path: filePath,
    name: path.basename(filePath),
    sizeBytes: stats.size,
    requiresTrust: ![".yaml", ".yml"].includes(extension),
  };
}

function optionArguments(options = {}) {
  const selectedNumbers = Array.isArray(options.selectedNumbers)
    ? options.selectedNumbers.join(",")
    : "1,2,3,4,5,6";
  const arguments = [
    "--selected-numbers",
    selectedNumbers,
    "--trend-bins",
    String(options.trendBins ?? 100),
    "--correlation-method",
    String(options.correlationMethod ?? "pearson"),
    "--border-space",
    String(options.borderSpace ?? 7),
    "--reports",
    enabledReportsList().join(","),
    "--strategies",
    enabledStrategiesList().join(","),
  ];
  if (Number.isInteger(options.targetGroupCount)) {
    arguments.push("--target-group-count", String(options.targetGroupCount));
  }
  return arguments;
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
          label: "Open YAML or Dataset...",
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
      label: "Analyze",
      submenu: [
        {
          label: "Reanalyze",
          accelerator: "F5",
          enabled: activeDatasetPath !== null,
          click: () => sendMenuAction("reanalyze"),
        },
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
          label: "Last Seen Highlight",
          accelerator: "CmdOrCtrl+Shift+L",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("last-seen"),
          click: () => sendMenuAction("openWorkspaceTab", { tab: "last-seen" }),
        },
        {
          label: "Last Seen Gap Highlight",
          accelerator: "CmdOrCtrl+Shift+G",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("last-seen-gap"),
          click: () =>
            sendMenuAction("openWorkspaceTab", { tab: "last-seen-gap" }),
        },
        {
          label: "Last Seen Space Highlight",
          accelerator: "CmdOrCtrl+Shift+S",
          enabled:
            activeDatasetPath !== null &&
            enabledReportIds.has("last-seen-space"),
          click: () =>
            sendMenuAction("openWorkspaceTab", { tab: "last-seen-space" }),
        },
        {
          label: "Predictions",
          accelerator: "CmdOrCtrl+Shift+P",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("predictions"),
          click: () => sendMenuAction("openWorkspaceTab", { tab: "predictions" }),
        },
        {
          label: "Possible Draw",
          accelerator: "CmdOrCtrl+Shift+D",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("possible-draw"),
          click: () =>
            sendMenuAction("openWorkspaceTab", { tab: "possible-draw" }),
        },
        {
          label: "Draw Portfolio",
          enabled:
            activeDatasetPath !== null && enabledReportIds.has("draw-portfolio"),
          click: () =>
            sendMenuAction("openWorkspaceTab", { tab: "draw-portfolio" }),
        },
        {
          label: "Draw History Editor",
          accelerator: "CmdOrCtrl+Shift+H",
          enabled: activeDatasetPath !== null,
          click: () =>
            sendMenuAction("openWorkspaceTab", { tab: "draw-history" }),
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
  const storedTemplate = loadStoredColorTemplate();
  const storedBackground = storedTemplate?.colors?.["application.background"];
  mainWindow = new BrowserWindow({
    title: "Rand AI",
    width: 1480,
    height: 980,
    minWidth: 1080,
    minHeight: 720,
    backgroundColor: typeof storedBackground === "string"
      ? storedBackground.slice(0, 7)
      : "#eef3f7",
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

ipcMain.handle("dataset:recent-list", () =>
  recentDatasets.map(recentDatasetPayload),
);

ipcMain.handle("dataset:recent-open", async (_event, requestedPath) =>
  resolveRecentDataset(requestedPath),
);

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

ipcMain.handle("color-template:get", () => loadStoredColorTemplate());

ipcMain.handle("color-template:apply", async (_event, requestedTemplate) => {
  const template = validatedColorTemplate(requestedTemplate);
  await writeJsonAtomically(colorTemplatePreferencesPath(), template);
  if (mainWindow && !mainWindow.isDestroyed()) {
    const background = template.colors["application.background"];
    if (typeof background === "string") mainWindow.setBackgroundColor(background.slice(0, 7));
  }
  return template;
});

ipcMain.handle("color-template:load", async (event) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
  const result = await dialog.showOpenDialog(parent, {
    title: "Load Rand AI color template",
    properties: ["openFile"],
    filters: [
      { name: "Rand AI color template", extensions: ["randai-theme.json", "json"] },
      { name: "JSON document", extensions: ["json"] },
    ],
  });
  if (result.canceled || result.filePaths.length === 0) return { canceled: true };
  const filePath = result.filePaths[0];
  const stats = await fs.promises.stat(filePath);
  if (stats.size > maximumColorTemplateBytes) {
    throw new Error("Color template must not exceed 1 MiB.");
  }
  const template = validatedColorTemplate(
    JSON.parse(await fs.promises.readFile(filePath, "utf8")),
  );
  return { canceled: false, path: filePath, template };
});

ipcMain.handle("color-template:save", async (event, requestedTemplate) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
  const template = validatedColorTemplate(requestedTemplate);
  const result = await dialog.showSaveDialog(parent, {
    title: "Save Rand AI color template",
    defaultPath: `${safeColorTemplateStem(template.name)}.randai-theme.json`,
    filters: [{ name: "Rand AI color template", extensions: ["randai-theme.json", "json"] }],
  });
  if (result.canceled || !result.filePath) return { canceled: true };
  const filePath = result.filePath.toLowerCase().endsWith(".randai-theme.json")
    ? result.filePath
    : `${result.filePath.replace(/\.json$/i, "")}.randai-theme.json`;
  await writeJsonAtomically(filePath, template);
  return { canceled: false, path: filePath };
});

ipcMain.handle("dataset:analyze", async (event, request) => {
  const sendProgress = (progress) => {
    if (!event.sender.isDestroyed()) {
      event.sender.send("dataset:analysis-progress", progress);
    }
  };
  const mapProgress = (start, end) => (progress) => {
    const bridgePercent = Math.min(
      Math.max(Number(progress?.percent) || 0, 0),
      100,
    );
    sendProgress({
      percent: Math.round(start + ((end - start) * bridgePercent) / 100),
      message: progress.message,
    });
  };
  sendProgress({ percent: 2, message: "Checking the selected file and its size" });
  const selectedPath = path.resolve(String(request?.path ?? ""));
  const extension = path.extname(selectedPath).toLowerCase();
  if (![".yaml", ".yml", ".pkl", ".pickle"].includes(extension)) {
    throw new Error("Select a .yaml, .yml, .pkl, or .pickle Draws dataset.");
  }
  const selectedStats = await fs.promises.stat(selectedPath);
  if (selectedStats.size > maximumUploadBytes) {
    throw new Error("Dataset file must not exceed 100 MiB.");
  }

  const isYaml = [".yaml", ".yml"].includes(extension);
  let analysisPath = selectedPath;
  if (isYaml) {
    analysisPath = path.join(
      path.dirname(selectedPath),
      `${path.basename(selectedPath, extension)}.pkl`,
    );
    sendProgress({ percent: 3, message: "Opening the YAML draw history" });
    await runBridge(
      ["yaml-import", "--input", selectedPath, "--output", analysisPath],
      true,
      mapProgress(3, 10),
    );
    const pickleStats = await fs.promises.stat(analysisPath);
    if (pickleStats.size > maximumUploadBytes) {
      throw new Error("Generated pickle file must not exceed 100 MiB.");
    }
    sendProgress({
      percent: 11,
      message: "Managed pickle created; starting the Python analysis engine",
    });
  } else {
    sendProgress({ percent: 3, message: "Starting the Python analysis engine" });
  }

  const analysisArguments = [
    "analyze",
    "--input",
    analysisPath,
    ...optionArguments(request?.options),
    "--cache-dir",
    analysisCachePath(),
  ];
  if (request?.forceReanalysis === true) {
    analysisArguments.push("--refresh-cache");
  }
  const payload = await runBridge(
    analysisArguments,
    true,
    isYaml ? mapProgress(11, 97) : sendProgress,
  );
  sendProgress({ percent: 98, message: "Updating the tabbed workspace and application menus" });
  activeDatasetPath = analysisPath;
  addRecentDataset(selectedPath, selectedStats.size);
  buildApplicationMenu();
  sendProgress({ percent: 100, message: "Analysis ready" });
  return payload;
});

ipcMain.handle("portfolio-backtest:data", async (event, request) => {
  if (!activeDatasetPath) {
    throw new Error("Analyze a dataset before running a portfolio simulation.");
  }
  const requested = new Set(
    Array.isArray(request?.strategyIds) ? request.strategyIds : [],
  );
  const strategyIds = strategyPlugins
    .map((plugin) => plugin.id)
    .filter((strategyId) => requested.has(strategyId));
  const bridgeArguments = [
      "portfolio-backtest-data",
      "--input",
      activeDatasetPath,
      "--strategies",
      strategyIds.join(","),
      "--cache-dir",
      analysisCachePath(),
      "--border-space",
      String(request?.borderSpace ?? 7),
    ];
  if (Number.isInteger(request?.targetGroupCount)) {
    bridgeArguments.push(
      "--target-group-count",
      String(request.targetGroupCount),
    );
  }
  return runBridge(
    bridgeArguments,
    true,
    (progress) => {
      if (!event.sender.isDestroyed()) {
        event.sender.send("portfolio-backtest:progress", progress);
      }
    },
  );
});

ipcMain.handle("portfolio-backtest:cache-load", async (_event, requestedKey) => {
  const key = validatedPortfolioCacheKey(requestedKey);
  const filePath = path.join(portfolioBacktestCachePath(), `${key}.json.gz`);
  try {
    const compressed = await fs.promises.readFile(filePath);
    const decoded = JSON.parse((await gunzip(compressed)).toString("utf8"));
    if (!validPortfolioBacktestResult(decoded)) {
      throw new Error("Portfolio backtest cache has an invalid structure.");
    }
    await fs.promises.utimes(filePath, new Date(), new Date());
    return decoded;
  } catch (error) {
    if (error?.code !== "ENOENT") {
      await fs.promises.unlink(filePath).catch(() => undefined);
      console.warn("Discarded invalid portfolio backtest cache:", error);
    }
    return null;
  }
});

ipcMain.handle("portfolio-backtest:cache-save", async (_event, request) => {
  const key = validatedPortfolioCacheKey(request?.key);
  const result = request?.result;
  if (!validPortfolioBacktestResult(result)) {
    throw new Error("Portfolio backtest result is invalid.");
  }
  const cachePath = portfolioBacktestCachePath();
  await fs.promises.mkdir(cachePath, { recursive: true });
  const filePath = path.join(cachePath, `${key}.json.gz`);
  const temporaryPath = `${filePath}.${process.pid}.${crypto.randomUUID()}.tmp`;
  try {
    await fs.promises.writeFile(
      temporaryPath,
      await gzip(Buffer.from(JSON.stringify(result), "utf8")),
    );
    await fs.promises.rename(temporaryPath, filePath);
  } finally {
    await fs.promises.unlink(temporaryPath).catch(() => undefined);
  }
  await prunePortfolioBacktestCache();
});

ipcMain.handle("possible-draw:for-sure-limit-error", (event, requestedNumber) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
  return dialog.showMessageBox(parent, {
    type: "error",
    title: "Fixed-number limit reached",
    message: "No more than six numbers can be marked Fixed.",
    detail: `Number ${requestedNumber} was not marked Fixed. Clear a Fixed number before trying again.`,
  });
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
  return payload;
});

ipcMain.handle("draw-comparison:save-pdf", async (event, request) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
  const requestedName =
    typeof request?.suggestedName === "string"
      ? path.basename(request.suggestedName)
      : "rand-ai-draw-comparison.pdf";
  const safeStem =
    requestedName
      .replace(/\.pdf$/i, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "rand-ai-draw-comparison";
  const result = await dialog.showSaveDialog(parent, {
    title: "Save draw comparison report",
    defaultPath: `${safeStem}.pdf`,
    filters: [{ name: "PDF document", extensions: ["pdf"] }],
  });
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }
  const pdf = await event.sender.printToPDF({
    printBackground: true,
    landscape: true,
    pageSize: "A4",
    preferCSSPageSize: true,
  });
  fs.writeFileSync(result.filePath, pdf);
  return { canceled: false, path: result.filePath };
});

ipcMain.handle("draw-portfolio:save-pdf", async (event, request) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
  const requestedName =
    typeof request?.suggestedName === "string"
      ? path.basename(request.suggestedName)
      : "rand-ai-draw-portfolio.pdf";
  const safeStem =
    requestedName
      .replace(/\.pdf$/i, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "rand-ai-draw-portfolio";
  const result = await dialog.showSaveDialog(parent, {
    title: "Save draw portfolio",
    defaultPath: `${safeStem}.pdf`,
    filters: [{ name: "PDF document", extensions: ["pdf"] }],
  });
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }
  const pdf = await event.sender.printToPDF({
    printBackground: true,
    landscape: true,
    pageSize: "A4",
    preferCSSPageSize: true,
  });
  fs.writeFileSync(result.filePath, pdf);
  return { canceled: false, path: result.filePath };
});

ipcMain.handle("draw-comparison:print", async (event) => {
  await new Promise((resolve, reject) => {
    event.sender.print(
      {
        silent: false,
        printBackground: true,
        landscape: true,
      },
      (success, failureReason) => {
        if (success) resolve();
        else reject(new Error(failureReason || "Printing failed."));
      },
    );
  });
});

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
