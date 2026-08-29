import { describe, expect, it } from "vitest";
import type { NonlinearEvidenceStatus } from "../types";
import {
  NONLINEAR_STATUS_LABELS,
  RECURRENCE_FORECAST_VERSION_LABEL,
  recurrencePointPercent,
} from "./nonlinearDynamics";

describe("nonlinear dynamics report helpers", () => {
  it("labels every evidence state used by the report", () => {
    const statuses: NonlinearEvidenceStatus[] = [
      "insufficient",
      "weak",
      "suggestive",
      "supported",
    ];

    expect(Object.keys(NONLINEAR_STATUS_LABELS)).toEqual(statuses);
    expect(statuses.map((status) => NONLINEAR_STATUS_LABELS[status])).toEqual([
      "Insufficient evidence",
      "Weak evidence",
      "Suggestive evidence",
      "Supported evidence",
    ]);
  });

  it("places recurrence points at cell centers and handles an empty plot", () => {
    expect(recurrencePointPercent(0, 4)).toBe(12.5);
    expect(recurrencePointPercent(3, 4)).toBe(87.5);
    expect(recurrencePointPercent(0, 0)).toBe(50);
  });

  it("identifies the replacement forecast as V2", () => {
    expect(RECURRENCE_FORECAST_VERSION_LABEL).toBe("Causal validation · V2");
  });
});
