/// <reference lib="webworker" />

import { runPortfolioBacktest } from "../lib/drawPortfolioBacktest";
import type { PortfolioBacktestData } from "../types";

interface StartMessage {
  data: PortfolioBacktestData;
  portfolioSize: number;
}

self.onmessage = (event: MessageEvent<StartMessage>) => {
  try {
    const result = runPortfolioBacktest(
      event.data.data,
      event.data.portfolioSize,
      (progress) => self.postMessage({ type: "progress", progress }),
    );
    self.postMessage({ type: "result", result });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};

export {};
