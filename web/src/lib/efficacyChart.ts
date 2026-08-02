export type EfficacyChartMode = "rate" | "lift";

export interface EfficacyChartRow {
  id: string;
  label: string;
  rate: number;
  normalizedLift: number;
  detail?: string;
}

export interface EfficacyChartScale {
  maximum: number;
  referencePercent: number;
}

export interface EfficacyBarGeometry {
  leftPercent: number;
  widthPercent: number;
}

const MINIMUM_SCALE = 0.001;
const RATE_HEADROOM = 1.08;

export function buildEfficacyChartScale(
  rows: readonly EfficacyChartRow[],
  mode: EfficacyChartMode,
  randomRate: number,
): EfficacyChartScale {
  if (!Number.isFinite(randomRate) || randomRate < 0) {
    throw new RangeError("Random rate must be a finite non-negative number");
  }

  if (mode === "lift") {
    const maximum = Math.max(
      MINIMUM_SCALE,
      ...rows.map((row) => Math.abs(row.normalizedLift)),
    );
    return { maximum, referencePercent: 50 };
  }

  const maximum =
    Math.max(MINIMUM_SCALE, randomRate, ...rows.map((row) => row.rate)) *
    RATE_HEADROOM;
  return {
    maximum,
    referencePercent: Math.min(100, (randomRate / maximum) * 100),
  };
}

export function efficacyBarGeometry(
  value: number,
  mode: EfficacyChartMode,
  maximum: number,
): EfficacyBarGeometry {
  if (!Number.isFinite(value)) {
    throw new RangeError("Chart value must be finite");
  }
  if (!Number.isFinite(maximum) || maximum <= 0) {
    throw new RangeError("Chart maximum must be a positive finite number");
  }

  if (mode === "rate") {
    return {
      leftPercent: 0,
      widthPercent: Math.min(100, Math.max(0, (value / maximum) * 100)),
    };
  }

  const widthPercent = Math.min(50, (Math.abs(value) / maximum) * 50);
  return {
    leftPercent: value < 0 ? 50 - widthPercent : 50,
    widthPercent,
  };
}

export function formatEfficacyChartValue(
  value: number,
  mode: EfficacyChartMode,
): string {
  const formatted = value.toFixed(3);
  return mode === "lift" && value > 0 ? `+${formatted}` : formatted;
}
