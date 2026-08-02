import { describe, expect, it } from "vitest";
import {
  buildEfficacyChartScale,
  efficacyBarGeometry,
  formatEfficacyChartValue,
  type EfficacyChartRow,
} from "./efficacyChart";

const rows: EfficacyChartRow[] = [
  { id: "ahead", label: "Ahead", rate: 0.8, normalizedLift: 0.065 },
  { id: "behind", label: "Behind", rate: 0.7, normalizedLift: -0.035 },
];

describe("efficacy chart geometry", () => {
  it("scales rate bars from zero and positions the random reference", () => {
    const scale = buildEfficacyChartScale(rows, "rate", 36 / 49);
    const bar = efficacyBarGeometry(0.8, "rate", scale.maximum);

    expect(scale.maximum).toBeCloseTo(0.864);
    expect(scale.referencePercent).toBeCloseTo((36 / 49 / 0.864) * 100);
    expect(bar).toEqual({ leftPercent: 0, widthPercent: expect.any(Number) });
    expect(bar.widthPercent).toBeCloseTo((0.8 / 0.864) * 100);
  });

  it("uses a zero-centered symmetric scale for lift bars", () => {
    const scale = buildEfficacyChartScale(rows, "lift", 36 / 49);

    expect(scale).toEqual({ maximum: 0.065, referencePercent: 50 });
    expect(efficacyBarGeometry(0.065, "lift", scale.maximum)).toEqual({
      leftPercent: 50,
      widthPercent: 50,
    });
    expect(efficacyBarGeometry(-0.0325, "lift", scale.maximum)).toEqual({
      leftPercent: 25,
      widthPercent: 25,
    });
  });

  it("formats exact rate and signed lift values", () => {
    expect(formatEfficacyChartValue(0.735, "rate")).toBe("0.735");
    expect(formatEfficacyChartValue(0.0125, "lift")).toBe("+0.013");
    expect(formatEfficacyChartValue(-0.0125, "lift")).toBe("-0.013");
  });

  it("rejects invalid scale inputs", () => {
    expect(() => buildEfficacyChartScale(rows, "rate", -1)).toThrow(RangeError);
    expect(() => efficacyBarGeometry(1, "rate", 0)).toThrow(RangeError);
  });
});
