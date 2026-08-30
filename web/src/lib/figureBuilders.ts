import type {
  AnalysisPayload,
  FigureSpec,
  TablePayload,
  TableRow,
} from "../types";
import { themeColor } from "./colorTemplates";

function chartColors(): string[] {
  return [1, 2, 3, 4, 5, 6].map((index) =>
    themeColor(`charts.series${index}`),
  );
}

function table(analysis: AnalysisPayload, name: string): TablePayload {
  const value = analysis.tables[name];
  if (!value) throw new Error(`Missing analysis table: ${name}`);
  return value;
}

function numberValue(row: TableRow, key: string): number {
  return Number(row[key] ?? 0);
}

function baseLayout(title: string, xTitle = "", yTitle = ""): Record<string, unknown> {
  return {
    title: { text: title, x: 0.02, xanchor: "left", font: { size: 18, color: themeColor("charts.title") } },
    paper_bgcolor: themeColor("charts.paper"),
    plot_bgcolor: themeColor("charts.plot"),
    font: { family: "Segoe UI, Helvetica Neue, sans-serif", color: themeColor("charts.text") },
    margin: { l: 62, r: 24, t: 62, b: 58 },
    xaxis: { title: { text: xTitle }, gridcolor: themeColor("charts.grid"), zerolinecolor: themeColor("charts.zeroLine") },
    yaxis: { title: { text: yTitle }, gridcolor: themeColor("charts.grid"), zerolinecolor: themeColor("charts.zeroLine") },
    hoverlabel: { bgcolor: themeColor("charts.hoverBackground"), bordercolor: themeColor("charts.hoverBorder"), font: { color: themeColor("charts.title") } },
    legend: { orientation: "h", y: 1.08, x: 1, xanchor: "right" },
  };
}

function heatmap(
  source: TablePayload,
  title: string,
  xTitle: string,
  yTitle: string,
  valueKey: string | null = null,
): FigureSpec {
  if (valueKey) {
    const rowKey = source.columns[0];
    const columnKey = source.columns[1];
    const x = [...new Set(source.rows.map((row) => String(row[columnKey])))];
    const y = [...new Set(source.rows.map((row) => String(row[rowKey])))];
    const lookup = new Map(
      source.rows.map((row) => [
        `${String(row[rowKey])}|${String(row[columnKey])}`,
        numberValue(row, valueKey),
      ]),
    );
    return {
      data: [{
        type: "heatmap",
        x,
        y,
        z: y.map((row) => x.map((column) => lookup.get(`${row}|${column}`) ?? 0)),
        colorscale: [[0, themeColor("charts.heatLow")], [0.25, themeColor("charts.heatMidLow")], [0.5, themeColor("charts.heatMid")], [0.75, themeColor("charts.heatMidHigh")], [1, themeColor("charts.heatHigh")]],
        hovertemplate: `${yTitle}: %{y}<br>${xTitle}: %{x}<br>Value: %{z}<extra></extra>`,
      }],
      layout: baseLayout(title, xTitle, yTitle),
    };
  }

  const x = source.columns.filter((column) => column !== "row");
  const y = source.rows.map((row) => String(row.row));
  return {
    data: [{
      type: "heatmap",
      x,
      y,
      z: source.rows.map((row) => x.map((column) => numberValue(row, column))),
      colorscale: [[0, themeColor("charts.negative")], [0.5, themeColor("charts.neutral")], [1, themeColor("charts.positive")]],
      zmin: -1,
      zmax: 1,
      hovertemplate: `${yTitle}: %{y}<br>${xTitle}: %{x}<br>Correlation: %{z:.3f}<extra></extra>`,
    }],
    layout: baseLayout(title, xTitle, yTitle),
  };
}

export function numberFrequencyFigure(source: TablePayload): FigureSpec {
  const rows = source.rows;
  const layout = baseLayout(
    "Number frequency compared with uniform expectation",
    "Number",
    "Appearances",
  );
  return {
    data: [
      {
        type: "bar",
        name: "Observed",
        x: rows.map((row) => numberValue(row, "number")),
        y: rows.map((row) => numberValue(row, "count")),
        marker: { color: themeColor("charts.series1") },
        hovertemplate: "Number %{x}<br>Count %{y}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "Expected",
        x: rows.map((row) => numberValue(row, "number")),
        y: rows.map((row) => numberValue(row, "expected_count")),
        line: { color: themeColor("charts.series2"), width: 3 },
        hovertemplate: "Number %{x}<br>Expected %{y:.2f}<extra></extra>",
      },
    ],
    layout,
  };
}

export function groupFrequencyFigure(
  source: TablePayload,
  metadata: { datasetName: string; borderSpace: number },
): FigureSpec {
  const rows = source.rows;
  const layout = baseLayout("Border Group Frequency", "Number of groups", "Draws");
  const rowsPerGroup = new Map<number, number>();
  for (const row of rows) {
    const groupCount = numberValue(row, "group_count");
    rowsPerGroup.set(groupCount, (rowsPerGroup.get(groupCount) ?? 0) + 1);
  }
  const groupOffsets = new Map<number, number>();
  return {
    data: rows.map((row, index) => {
      const groupCount = numberValue(row, "group_count");
      const signature = String(row.signature);
      const count = numberValue(row, "count");
      const barCount = rowsPerGroup.get(groupCount) ?? 1;
      const barIndex = groupOffsets.get(groupCount) ?? 0;
      groupOffsets.set(groupCount, barIndex + 1);
      const barWidth = Math.min(0.5, (0.72 - (barCount - 1) * 0.03) / barCount);
      const xPosition = groupCount
        + (barIndex - (barCount - 1) / 2) * (barWidth + 0.03);
      return {
        type: "bar",
        name: signature,
        x: [xPosition],
        y: [count],
        width: [barWidth],
        marker: { color: chartColors()[index % chartColors().length] },
        text: [`${signature} · ${count.toLocaleString()}`],
        texttemplate: "%{text}",
        textposition: "inside",
        insidetextanchor: "middle",
        constraintext: "inside",
        customdata: [[metadata.datasetName, metadata.borderSpace, groupCount, signature]],
        hovertemplate:
          "Dataset %{customdata[0]}<br>Border space %{customdata[1]}<br>" +
          "%{customdata[2]} groups<br>Signature %{customdata[3]}<br>" +
          "%{y} draws<extra></extra>",
      };
    }),
    layout: {
      ...layout,
      barmode: "group",
      bargap: 0.18,
      uniformtext: { minsize: 9, mode: "hide" },
      margin: {
        ...(layout.margin as Record<string, unknown>),
        t: 108,
      },
      legend: {
        ...(layout.legend as Record<string, unknown>),
        orientation: "h",
        x: 0,
        xanchor: "left",
        y: 1.18,
        traceorder: "normal",
      },
      xaxis: {
        ...(layout.xaxis as Record<string, unknown>),
        tickmode: "array",
        tickvals: [1, 2, 3, 4, 5, 6],
        range: [0.5, 6.5],
      },
      yaxis: {
        ...(layout.yaxis as Record<string, unknown>),
        rangemode: "tozero",
      },
    },
  };
}

function drawSumDistribution(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "draw_structure_distributions").rows.filter(
    (row) => row.measure === "draw_sum",
  );
  return {
    data: [{
      type: "bar",
      x: rows.map((row) => numberValue(row, "value")),
      y: rows.map((row) => numberValue(row, "count")),
      marker: { color: themeColor("charts.series1") },
    }],
    layout: baseLayout("Draw-sum distribution", "Sum of six numbers", "Draws"),
  };
}

function composition(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "draw_structure_distributions").rows;
  const measures = [
    ["odd_count", "Odd numbers"],
    ["low_count", "Numbers from 1 to 24"],
    ["consecutive_pair_count", "Consecutive pairs"],
  ];
  return {
    data: measures.map(([measure, label], index) => {
      const values = rows.filter((row) => row.measure === measure);
      return {
        type: "bar",
        name: label,
        x: values.map((row) => numberValue(row, "value")),
        y: values.map((row) => numberValue(row, "count")),
        marker: { color: chartColors()[index] },
        xaxis: index === 0 ? "x" : `x${index + 1}`,
        yaxis: index === 0 ? "y" : `y${index + 1}`,
        showlegend: false,
      };
    }),
    layout: {
      ...baseLayout("Draw composition distributions", "Count in draw", "Draws"),
      grid: { rows: 1, columns: 3, pattern: "independent" },
      annotations: measures.map(([, label], index) => ({
        text: label,
        x: (index + 0.5) / 3,
        y: 1.04,
        xref: "paper",
        yref: "paper",
        showarrow: false,
        font: { size: 13, color: themeColor("charts.text") },
      })),
    },
  };
}

function trends(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "number_trends").rows;
  const numbers = [...new Set(rows.map((row) => numberValue(row, "number")))];
  return {
    data: numbers.map((number, index) => {
      const values = rows.filter((row) => numberValue(row, "number") === number);
      return {
        type: "scatter",
        mode: "lines+markers",
        name: String(number),
        x: values.map((row) => numberValue(row, "bin")),
        y: values.map((row) => numberValue(row, "appearance_rate")),
        customdata: values.map((row) => [
          numberValue(row, "start_draw"),
          numberValue(row, "end_draw"),
          numberValue(row, "count"),
        ]),
        line: { color: chartColors()[index % chartColors().length], width: 2 },
        hovertemplate:
          "Bin %{x}<br>Rate %{y:.2f}%<br>Draws %{customdata[0]}–%{customdata[1]}<br>Count %{customdata[2]}<extra></extra>",
      };
    }),
    layout: baseLayout("Appearance rate by draw-index bin", "Draw-index bin", "Appearance rate (%)"),
  };
}

function distanceFigure(
  source: TablePayload,
  title: string,
  xKey: string,
  yKey: string,
): FigureSpec {
  return {
    data: [{
      type: "bar",
      x: source.rows.map((row) => numberValue(row, xKey)),
      y: source.rows.map((row) => numberValue(row, yKey)),
      marker: { color: themeColor("charts.series1") },
    }],
    layout: {
      ...baseLayout(title, "Distance (0–43)", "Occurrences"),
      xaxis: {
        title: { text: "Distance (0–43)" },
        range: [-0.5, 43.5],
        dtick: 1,
        gridcolor: themeColor("charts.grid"),
      },
    },
  };
}

function spaceBox(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "sampled_spaces").rows;
  const positions = [...new Set(rows.map((row) => String(row.position)))];
  return {
    data: positions.map((position, index) => ({
      type: "box",
      name: position,
      y: rows
        .filter((row) => row.position === position)
        .map((row) => numberValue(row, "space")),
      marker: { color: chartColors()[index % chartColors().length] },
      boxpoints: false,
    })),
    layout: baseLayout(
      `Space distributions (deterministic sample: ${analysis.dataset.sampleSize.toLocaleString()} draws)`,
      "Space position",
      "Space value",
    ),
  };
}

function spaceExtremes(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "space_extreme_distributions").rows;
  const measures = [...new Set(rows.map((row) => String(row.measure)))];
  return {
    data: measures.map((measure, index) => {
      const values = rows.filter((row) => row.measure === measure);
      return {
        type: "bar",
        name: measure,
        x: values.map((row) => numberValue(row, "value")),
        y: values.map((row) => numberValue(row, "count")),
        marker: { color: chartColors()[index] },
      };
    }),
    layout: {
      ...baseLayout("Minimum and maximum space distributions", "Space value", "Draws"),
      barmode: "group",
    },
  };
}

function matchingPairs(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "randomness_diagnostics").rows;
  const lookup = new Map(rows.map((row) => [String(row.diagnostic), numberValue(row, "value")]));
  return {
    data: [{
      type: "bar",
      x: ["Observed", "Expected"],
      y: [
        lookup.get("observed_matching_combination_pairs") ?? 0,
        lookup.get("expected_matching_combination_pairs") ?? 0,
      ],
      marker: { color: [themeColor("charts.series1"), themeColor("charts.series2")] },
    }],
    layout: baseLayout("Matching six-number combination pairs", "", "Matching pairs"),
  };
}

function freshnessGapDistribution(analysis: AnalysisPayload): FigureSpec {
  const rows = table(analysis, "freshness_gap_distribution").rows;
  const layout = baseLayout(
    "Number hits by exact freshness gap",
    "Gap (intervening draws since the previous hit)",
    "Number hits",
  );
  return {
    data: [{
      type: "bar",
      x: rows.map((row) => numberValue(row, "gap")),
      y: rows.map((row) => numberValue(row, "hits")),
      customdata: rows.map((row) => [
        numberValue(row, "opportunities"),
        numberValue(row, "hit_rate"),
        numberValue(row, "hit_percentage"),
      ]),
      marker: {
        color: rows.map((row) => numberValue(row, "hit_rate")),
        colorscale: [
          [0, themeColor("charts.neutral")],
          [0.5, themeColor("charts.series1")],
          [1, themeColor("charts.series4")],
        ],
        colorbar: { title: { text: "Hit rate %" } },
      },
      hovertemplate:
        "Gap %{x}<br>Hits %{y:,}<br>Opportunities %{customdata[0]:,}<br>Hit rate %{customdata[1]:.3f}%<br>Share of all hits %{customdata[2]:.3f}%<extra></extra>",
    }],
    layout: {
      ...layout,
      bargap: 0.08,
      xaxis: {
        title: { text: "Gap (intervening draws since the previous hit)" },
        dtick: rows.length > 60 ? 5 : 1,
        gridcolor: themeColor("charts.grid"),
      },
    },
  };
}

function borderGroupFigures(analysis: AnalysisPayload): Record<string, FigureSpec> {
  const signatureRows = table(analysis, "space_group_signature_distribution").rows;
  const historyRows = table(analysis, "space_group_history").rows;
  const sensitivityRows = table(analysis, "space_group_threshold_sensitivity").rows;
  const metricRows = table(analysis, "space_group_model_metrics").rows.filter(
    (row) => row.log_loss !== null,
  );
  return {
    border_group_signatures: {
      data: [
        {
          type: "bar",
          name: "Observed",
          x: signatureRows.map((row) => String(row.signature)),
          y: signatureRows.map((row) => numberValue(row, "percentage")),
          marker: { color: themeColor("charts.series3") },
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "Exact random null",
          x: signatureRows.map((row) => String(row.signature)),
          y: signatureRows.map((row) => numberValue(row, "null_probability")),
          line: { color: themeColor("charts.series2"), width: 2 },
        },
      ],
      layout: baseLayout("Canonical group signatures", "Group sizes", "Draws (%)"),
    },
    border_group_history: {
      data: [
        {
          type: "scatter",
          mode: "markers",
          name: "Draw group count",
          x: historyRows.map((row) => numberValue(row, "draw_number")),
          y: historyRows.map((row) => numberValue(row, "group_count")),
          marker: { color: themeColor("charts.series1"), size: 5 },
        },
        {
          type: "scatter",
          mode: "lines",
          name: "Rolling 25",
          x: historyRows.map((row) => numberValue(row, "draw_number")),
          y: historyRows.map((row) => numberValue(row, "rolling_25_group_count")),
          line: { color: themeColor("charts.series4"), width: 2 },
        },
        {
          type: "scatter",
          mode: "lines",
          name: "Rolling 100",
          x: historyRows.map((row) => numberValue(row, "draw_number")),
          y: historyRows.map((row) => numberValue(row, "rolling_100_group_count")),
          line: { color: themeColor("charts.series5"), width: 2 },
        },
      ],
      layout: baseLayout("Border groups through history", "Draw", "Group count"),
    },
    border_group_transitions: heatmap(
      table(analysis, "space_group_transitions"),
      "Signature transition lift",
      "Next signature",
      "Current signature",
      "lift",
    ),
    border_group_sensitivity: {
      data: [
        {
          type: "scatter",
          mode: "lines",
          name: "Observed 1–3 groups",
          x: sensitivityRows.map((row) => numberValue(row, "border_space")),
          y: sensitivityRows.map((row) => numberValue(row, "one_to_three_percentage")),
          line: { color: themeColor("charts.series1"), width: 2 },
        },
        {
          type: "scatter",
          mode: "lines",
          name: "Exact random null",
          x: sensitivityRows.map((row) => numberValue(row, "border_space")),
          y: sensitivityRows.map((row) => numberValue(row, "null_one_to_three_percentage")),
          line: { color: themeColor("charts.series2"), width: 2, dash: "dot" },
        },
      ],
      layout: baseLayout("Border sensitivity", "Inclusive border space", "Draws with 1–3 groups (%)"),
    },
    border_group_models: {
      data: [{
        type: "bar",
        x: metricRows.map((row) => String(row.name)),
        y: metricRows.map((row) => numberValue(row, "log_loss")),
        marker: { color: metricRows.map((_row, index) => chartColors()[index % chartColors().length]) },
      }],
      layout: baseLayout("Walk-forward structure log loss (lower is better)", "Model", "Log loss"),
    },
  };
}

export function buildFigures(analysis: AnalysisPayload): Record<string, FigureSpec> {
  const enabled = new Set(analysis.options.enabledReports);
  const figures: Record<string, FigureSpec> = {};

  if (enabled.has("overview")) {
    figures.draw_sum_distribution = drawSumDistribution(analysis);
    figures.draw_composition = composition(analysis);
  }

  if (enabled.has("numbers")) {
    figures.position_frequencies = heatmap(
      table(analysis, "position_frequencies"),
      "Number frequency by sorted position",
      "Number",
      "Position",
      "count",
    );
    figures.pair_cooccurrence = heatmap(
      table(analysis, "pair_cooccurrence"),
      "Number pair co-occurrence",
      "Second number",
      "First number",
      "count",
    );
    figures.number_trends = trends(analysis);
  }

  if (enabled.has("spaces")) {
    const spaceFrequencies = table(analysis, "space_frequencies");
    figures.distance_frequencies = distanceFigure(
      table(analysis, "distance_frequencies"),
      "Distance occurrences across all six positions",
      "distance",
      "occurrences",
    );
    figures.space_frequencies = heatmap(
      spaceFrequencies,
      "Space frequency by position",
      "Space value",
      "Position",
      "count",
    );
    figures.space_box_plots = spaceBox(analysis);
    figures.space_extremes = spaceExtremes(analysis);
    for (let position = 1; position <= 6; position += 1) {
      const rows = spaceFrequencies.rows.filter(
        (row) => row.position === `dist${position}`,
      );
      figures[`dist${position}_frequencies`] = distanceFigure(
        { columns: spaceFrequencies.columns, rows },
        `dist${position} distance occurrences`,
        "space",
        "count",
      );
    }
  }

  if (enabled.has("space-groups")) {
    Object.assign(figures, borderGroupFigures(analysis));
  }

  if (enabled.has("relationships")) {
    const method = analysis.options.correlationMethod;
    const suffix = method === "spearman" ? "sampled" : "exact";
    figures.number_correlations = heatmap(
      table(analysis, `number_correlations_${method}`),
      `Number-position ${method[0].toUpperCase()}${method.slice(1)} correlations (${suffix})`,
      "Position",
      "Position",
    );
    figures.space_correlations = heatmap(
      table(analysis, `space_correlations_${method}`),
      `Space-position ${method[0].toUpperCase()}${method.slice(1)} correlations (${suffix})`,
      "Space",
      "Space",
    );
    figures.number_space_correlations = heatmap(
      table(analysis, `number_space_correlations_${method}`),
      `Number-to-space ${method[0].toUpperCase()}${method.slice(1)} correlations (${suffix})`,
      "Space",
      "Number position",
    );
  }

  if (enabled.has("randomness")) {
    figures.matching_pairs = matchingPairs(analysis);
  }
  if (enabled.has("gaps")) {
    figures.freshness_gap_distribution = freshnessGapDistribution(analysis);
  }
  return figures;
}

export function summaryLookup(analysis: AnalysisPayload): Map<string, number> {
  return new Map(
    table(analysis, "summary").rows.map((row) => [
      String(row.metric),
      numberValue(row, "value"),
    ]),
  );
}
