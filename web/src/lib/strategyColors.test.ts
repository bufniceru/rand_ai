import { describe, expect, it } from "vitest";
import type { StrategyId } from "../types";
import {
  STRATEGY_FAMILIES,
  STRATEGY_FAMILY_BY_ID,
} from "./strategyFamilies";
import {
  STRATEGY_TONE_POSITION_BY_ID,
  strategyColor,
  strategyFamilyColor,
} from "./strategyColors";

const strategyIds = Object.keys(STRATEGY_FAMILY_BY_ID) as StrategyId[];

describe("family-based strategy colors", () => {
  it("assigns every family a distinct Monokai base color", () => {
    const colors = STRATEGY_FAMILIES.map((family) => family.color);
    expect(new Set(colors).size).toBe(STRATEGY_FAMILIES.length);
    expect(colors.every((color) => /^#[0-9A-F]{6}$/.test(color))).toBe(true);
  });

  it("assigns every strategy an explicit tonal position and valid color", () => {
    expect(Object.keys(STRATEGY_TONE_POSITION_BY_ID).sort()).toEqual(
      [...strategyIds].sort(),
    );
    expect(strategyIds.every((id) => /^#[0-9A-F]{6}$/.test(strategyColor(id)))).toBe(
      true,
    );
  });

  it("uses unique evenly ordered tones inside each family", () => {
    for (const family of STRATEGY_FAMILIES) {
      const members = strategyIds.filter(
        (strategyId) => STRATEGY_FAMILY_BY_ID[strategyId] === family.id,
      );
      const positions = members.map(
        (strategyId) => STRATEGY_TONE_POSITION_BY_ID[strategyId],
      );
      const colors = members.map(strategyColor);

      expect([...positions].sort((left, right) => left - right)).toEqual(
        Array.from({ length: members.length }, (_value, index) => index),
      );
      expect(new Set(colors).size).toBe(members.length);
      expect(strategyFamilyColor(family.id)).toBe(family.color);
    }
  });

  it("keeps strategy colors stable", () => {
    expect(strategyColor("freshness")).toBe("#C87B56");
    expect(strategyColor("fresh_random")).toBe("#FFE085");
    expect(strategyColor("sparse_neural_ticket")).toBe("#93E3ED");
  });
});
