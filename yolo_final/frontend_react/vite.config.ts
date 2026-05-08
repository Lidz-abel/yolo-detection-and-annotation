import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:5000",
      "/model_predict": "http://127.0.0.1:5000",
      "/human_annotate": "http://127.0.0.1:5000"
    }
  }
});
