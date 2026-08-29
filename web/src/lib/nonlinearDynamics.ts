import type { NonlinearEvidenceStatus } from "../types";

export const NONLINEAR_STATUS_LABELS = {
  insufficient: "Insufficient evidence",
  weak: "Weak evidence",
  suggestive: "Suggestive evidence",
  supported: "Supported evidence",
} as const satisfies Record<NonlinearEvidenceStatus, string>;

export function recurrencePointPercent(position: number, size: number): number {
  return ((position + 0.5) / Math.max(size, 1)) * 100;
}
