import { build } from "vite";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
await build({
  configFile: false, root, logLevel: "warn",
  build: { ssr: "scripts/portfolio-study-runner.ts", outDir: "dist/portfolio-study",
    emptyOutDir: false, rollupOptions: { output: { entryFileNames: "runner.mjs" } } },
});
