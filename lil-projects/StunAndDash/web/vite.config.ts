import { defineConfig } from "vite";

export default defineConfig({
  publicDir: "../shared",
  server: { port: 5173 },
  build: { outDir: "dist", emptyOutDir: true },
});
