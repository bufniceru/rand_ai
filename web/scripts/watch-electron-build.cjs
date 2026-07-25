const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const webRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(webRoot, "..");
const debounceMilliseconds = 1500;
const sourceExtensions = new Set([".cjs", ".css", ".js", ".json", ".py", ".ts", ".vue"]);
const watchedTargets = [
  path.join(projectRoot, "src", "rand_ai"),
  path.join(webRoot, "src"),
  path.join(webRoot, "electron"),
  path.join(webRoot, "package.json"),
  path.join(webRoot, "package-lock.json"),
  path.join(webRoot, "tsconfig.json"),
  path.join(webRoot, "vite.config.ts"),
  path.join(projectRoot, "pyproject.toml"),
  path.join(projectRoot, "uv.lock"),
].filter((target) => fs.existsSync(target));

let buildProcess = null;
let debounceTimer = null;
let rebuildQueued = false;
const changedFiles = new Set();
const watchers = [];

function timestamp() {
  return new Date().toLocaleTimeString();
}

function isSourceChange(filePath) {
  const baseName = path.basename(filePath);
  return baseName === "uv.lock" || sourceExtensions.has(path.extname(baseName).toLowerCase());
}

function scheduleBuild(filePath) {
  if (!isSourceChange(filePath)) return;
  changedFiles.add(path.relative(projectRoot, filePath));
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runBuild, debounceMilliseconds);
}

function runBuild() {
  debounceTimer = null;
  if (buildProcess) {
    rebuildQueued = true;
    return;
  }

  const changes = [...changedFiles].sort();
  changedFiles.clear();
  console.log(`\n[${timestamp()}] Source change detected; rebuilding portable Electron executable.`);
  for (const change of changes) console.log(`  - ${change}`);

  const npmExecutable = process.platform === "win32" ? "npm.cmd" : "npm";
  buildProcess = spawn(npmExecutable, ["run", "electron:build"], {
    cwd: webRoot,
    stdio: "inherit",
    windowsHide: true,
  });

  buildProcess.on("error", (error) => {
    console.error(`[${timestamp()}] Could not start Electron rebuild:`, error);
  });
  buildProcess.on("exit", (code) => {
    console.log(
      code === 0
        ? `[${timestamp()}] Portable Electron executable rebuilt successfully.`
        : `[${timestamp()}] Electron rebuild failed with exit code ${code}.`,
    );
    buildProcess = null;
    if (rebuildQueued || changedFiles.size > 0) {
      rebuildQueued = false;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(runBuild, debounceMilliseconds);
    }
  });
}

function watchTarget(target) {
  const isDirectory = fs.statSync(target).isDirectory();
  const watcher = fs.watch(target, { recursive: isDirectory }, (_eventType, fileName) => {
    const changedPath = isDirectory && fileName ? path.join(target, String(fileName)) : target;
    scheduleBuild(changedPath);
  });
  watcher.on("error", (error) => {
    console.error(`[${timestamp()}] Watch error for ${target}:`, error);
  });
  watchers.push(watcher);
}

if (process.argv.includes("--check")) {
  console.log(`Electron rebuild hook is ready; ${watchedTargets.length} source targets found.`);
  process.exit(0);
}

for (const target of watchedTargets) watchTarget(target);
console.log("Electron executable rebuild hook is active.");
console.log("Save a source file to queue a debounced portable build; press Ctrl+C to stop.");

function shutdown() {
  clearTimeout(debounceTimer);
  for (const watcher of watchers) watcher.close();
  if (buildProcess) buildProcess.kill();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
