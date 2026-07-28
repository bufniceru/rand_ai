const { app, BrowserWindow } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const projectRoot = "C:\\code_py\\rand_ai";
const pythonPath = path.join(projectRoot, ".venv", "Scripts", "python.exe");
const reportPath = path.join(
  projectRoot,
  "output",
  "pdf",
  "rand-ai-draw-comparison-2026-07-26.pdf",
);
const payloadCachePath = path.join(
  projectRoot,
  "tmp",
  "pdfs",
  "draw-comparison-payload.json",
);
const previewPath = path.join(
  projectRoot,
  "tmp",
  "pdfs",
  "preview-dist",
  "report-preview.html",
);
const previewDirectory = path.dirname(previewPath);
const strategies = [
  "emd",
  "randomness",
  "fresh_random",
  "chi_square",
  "entropy",
  "mkfr",
  "mksp",
  "bayesian",
  "predictive_grid",
  "co_occurrence",
  "mixed",
  "svc",
  "tbl",
  "cis",
  "residual_coverage",
  "chained",
].join(",");

app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.setPath(
  "userData",
  path.join(projectRoot, "tmp", "pdfs", "electron-user-data"),
);

function analyzeDataset() {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonPath,
      [
        "-m",
        "rand_ai.gui_bridge",
        "analyze",
        "--input",
        path.join(projectRoot, "data", "lotto_results.pkl"),
        "--reports",
        "draw-comparison",
        "--strategies",
        strategies,
      ],
      {
        cwd: projectRoot,
        env: {
          ...process.env,
          PYTHONPATH: path.join(projectRoot, "src"),
        },
        windowsHide: true,
      },
    );
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `Analysis exited with code ${code}:\n${Buffer.concat(stderr)}`,
          ),
        );
        return;
      }
      resolve(JSON.parse(Buffer.concat(stdout).toString("utf8")));
    });
  });
}

function startPreviewServer() {
  const contentTypes = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
  };
  const server = http.createServer((request, response) => {
    const requestedPath = request.url === "/" ? "/report-preview.html" : request.url;
    const filePath = path.resolve(
      previewDirectory,
      `.${decodeURIComponent(requestedPath)}`,
    );
    if (!filePath.startsWith(previewDirectory) || !fs.existsSync(filePath)) {
      response.writeHead(404);
      response.end("Not found");
      return;
    }
    response.writeHead(200, {
      "Content-Type":
        contentTypes[path.extname(filePath).toLowerCase()] ??
        "application/octet-stream",
    });
    fs.createReadStream(filePath).pipe(response);
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

app.whenReady().then(async () => {
  let previewServer;
  try {
    const payload = fs.existsSync(payloadCachePath)
      ? JSON.parse(fs.readFileSync(payloadCachePath, "utf8"))
      : await analyzeDataset();
    if (!fs.existsSync(payloadCachePath)) {
      fs.writeFileSync(payloadCachePath, JSON.stringify(payload));
    }
    const window = new BrowserWindow({
      width: 1400,
      height: 1000,
      show: false,
      webPreferences: {
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: false,
      },
    });
    previewServer = await startPreviewServer();
    const address = previewServer.address();
    await window.loadURL(
      `http://127.0.0.1:${address.port}/report-preview.html`,
    );
    await window.webContents.executeJavaScript(
      `window.renderDrawComparisonPreview(${JSON.stringify(payload)})`,
    );
    const pdf = await window.webContents.printToPDF({
      printBackground: true,
      landscape: true,
      pageSize: "A4",
      preferCSSPageSize: true,
    });
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, pdf);
    console.log(reportPath);
    window.destroy();
    previewServer.close();
    app.quit();
  } catch (error) {
    if (previewServer) previewServer.close();
    console.error(error);
    app.exit(1);
  }
});
