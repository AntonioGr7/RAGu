import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Build to ../<dist served by FastAPI>. In dev, proxy /api to the Python server
// so the frontend can call relative /api paths in both dev and production.
export default defineConfig({
    plugins: [react()],
    build: { outDir: "dist", emptyOutDir: true },
    server: {
        port: 5173,
        proxy: {
            "/api": "http://127.0.0.1:8000",
        },
    },
});
