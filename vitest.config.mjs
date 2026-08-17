import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/js/**/*.test.mjs"],
    // Each test builds its own jsdom through tests/js/helpers/appWindow.mjs, so
    // the default node environment is enough here and keeps the global scope of
    // one test out of the next one's way.
    environment: "node",
    globals: false,
    restoreMocks: true,
  },
});
