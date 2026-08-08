import { afterEach, describe, expect, it } from "vitest";
import type { ColorTemplate } from "../types";
import {
  COLOR_TEMPLATE_KIND,
  COLOR_TEMPLATE_SCHEMA_VERSION,
  beginColorTemplatePreview,
  cancelColorTemplatePreview,
  cloneColorTemplate,
  colorTokenDefinitions,
  commitColorTemplatePreview,
  contrastRatio,
  createDefaultColorTemplate,
  defaultColorMap,
  isHexColor,
  materializeColorTemplate,
  normalizeHexColor,
  previewColorTemplate,
  setActiveColorTemplate,
  themeColor,
  validateColorTemplate,
} from "./colorTemplates";
import "./strategyColors";

afterEach(() => setActiveColorTemplate(createDefaultColorTemplate()));

describe("color template catalog", () => {
  it("has unique stable identifiers and valid defaults", () => {
    const definitions = colorTokenDefinitions();
    expect(definitions.length).toBeGreaterThan(90);
    expect(new Set(definitions.map((definition) => definition.id)).size).toBe(
      definitions.length,
    );
    expect(definitions.every((definition) => isHexColor(definition.defaultValue))).toBe(true);
    expect(definitions.every((definition) => definition.cssVariable?.startsWith("--theme-"))).toBe(true);
    expect(definitions.some((definition) => definition.id === "strategy.categorical_chi_square")).toBe(true);
  });

  it("exports a complete immutable-default snapshot", () => {
    const template = createDefaultColorTemplate();
    expect(template.kind).toBe(COLOR_TEMPLATE_KIND);
    expect(template.schemaVersion).toBe(COLOR_TEMPLATE_SCHEMA_VERSION);
    expect(Object.keys(template.colors)).toHaveLength(colorTokenDefinitions().length);
    const clone = cloneColorTemplate(template);
    clone.colors["application.background"] = "#000000";
    expect(template.colors["application.background"]).toBe("#E9EFF4");
  });
});

describe("color values", () => {
  it("normalizes shorthand, opaque, and alpha hexadecimal values", () => {
    expect(normalizeHexColor("#abc")).toBe("#AABBCC");
    expect(normalizeHexColor("#abcd")).toBe("#AABBCCDD");
    expect(normalizeHexColor("#12ab34ef")).toBe("#12AB34EF");
    expect(() => normalizeHexColor("rgb(1, 2, 3)")).toThrow("Invalid color value");
  });

  it("calculates accessible contrast ratios without treating alpha as a channel", () => {
    expect(contrastRatio("#000000", "#FFFFFF")).toBeCloseTo(21, 5);
    expect(contrastRatio("#77777780", "#FFFFFF")).toBeCloseTo(
      contrastRatio("#777777", "#FFFFFF"),
      5,
    );
  });
});

describe("template validation", () => {
  function partialTemplate(colors: Record<string, string>): Record<string, unknown> {
    return {
      kind: COLOR_TEMPLATE_KIND,
      schemaVersion: COLOR_TEMPLATE_SCHEMA_VERSION,
      name: "Imported",
      colors,
    };
  }

  it("fills missing known tokens and ignores unknown tokens with warnings", () => {
    const result = validateColorTemplate(
      partialTemplate({
        "application.background": "#01020304",
        "future.color": "#ABCDEF",
      }),
    );
    expect(result.template.colors["application.background"]).toBe("#01020304");
    expect(result.template.colors["text.primary"]).toBe("#172033");
    expect(result.template.colors["future.color"]).toBeUndefined();
    expect(result.warnings).toHaveLength(2);
  });

  it.each([
    [null, "does not contain"],
    [partialTemplate({ "application.background": "red" }), "must use"],
    [{ ...partialTemplate({}), kind: "other" }, "Template kind"],
    [{ ...partialTemplate({}), schemaVersion: 2 }, "Unsupported"],
    [{ ...partialTemplate({}), name: "" }, "Template name"],
  ])("rejects invalid templates", (value, message) => {
    expect(() => validateColorTemplate(value)).toThrow(message as string);
  });

  it("round-trips complete templates without losing colors", () => {
    const exported: ColorTemplate = {
      ...createDefaultColorTemplate(),
      name: "Round trip",
      description: "Portable colors only",
      exportedAt: "2026-08-08T00:00:00.000Z",
      colors: {
        ...defaultColorMap(),
        "charts.series1": "#12345678",
      },
    };
    const validated = validateColorTemplate(JSON.parse(JSON.stringify(exported)));
    expect(validated.warnings).toEqual([]);
    expect(materializeColorTemplate(validated.template)).toEqual(exported);
  });
});

describe("preview lifecycle", () => {
  it("restores a canceled preview and keeps a committed preview", () => {
    const original = createDefaultColorTemplate();
    setActiveColorTemplate(original);
    beginColorTemplatePreview();
    const preview = cloneColorTemplate(original);
    preview.colors["application.accent"] = "#123456";
    previewColorTemplate(preview);
    expect(themeColor("application.accent")).toBe("#123456");
    cancelColorTemplatePreview();
    expect(themeColor("application.accent")).toBe("#2E63C5");

    beginColorTemplatePreview();
    commitColorTemplatePreview(preview);
    expect(themeColor("application.accent")).toBe("#123456");
  });
});
