import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const outputDirectory = fileURLToPath(
  new URL("../tmp/pdfs/preview-dist/", import.meta.url),
);

export default defineConfig({
  base: "./",
  plugins: [vue()],
  build: {
    emptyOutDir: true,
    outDir: outputDirectory,
    rollupOptions: {
      input: fileURLToPath(new URL("./report-preview.html", import.meta.url)),
    },
  },
});
