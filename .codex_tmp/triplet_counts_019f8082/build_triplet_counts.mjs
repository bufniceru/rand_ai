import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "C:\\code_py\\rand_ai";
const sourcePath = path.join(projectRoot, "data", "lotto_results.yaml");
const outputDir = path.join(
  projectRoot,
  "outputs",
  "019f8082-01dc-7521-a62a-57ccc0edb08a",
);
const outputPath = path.join(outputDir, "lotto_triplet_counts.xlsx");
const previewCountsPath = path.join(outputDir, "triplet_counts_preview.png");
const previewAboutPath = path.join(outputDir, "triplet_counts_about_preview.png");

function parseHistory(yamlText) {
  const lines = yamlText.split(/\r?\n/);
  const draws = [];
  let firstDraw = "";
  let lastDraw = "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("first_draw:")) {
      firstDraw = trimmed.split(":", 2)[1].trim().replaceAll("'", "");
    } else if (trimmed.startsWith("last_draw:")) {
      lastDraw = trimmed.split(":", 2)[1].trim().replaceAll("'", "");
    }
  }

  for (let index = 0; index < lines.length; index += 1) {
    if (lines[index].trim() !== "numbers:") continue;
    const numbers = [];
    for (let offset = 1; offset <= 6; offset += 1) {
      const match = lines[index + offset]?.trim().match(/^-\s+(\d+)$/);
      if (!match) {
        throw new Error(`Invalid number list near YAML line ${index + 1}`);
      }
      numbers.push(Number(match[1]));
    }
    draws.push(numbers.sort((a, b) => a - b));
    index += 6;
  }

  if (draws.length === 0) throw new Error("No draws found in the YAML history");
  return { draws, firstDraw, lastDraw };
}

function aggregateTriplets(draws) {
  const lastDrawn = Array(50).fill(null);
  const counts = new Map();
  let maximumGap = 0;

  draws.forEach((numbers, drawIndex) => {
    const leftDistances = [
      (numbers[0] - 1) + (49 - numbers[5]),
      ...numbers.slice(1).map((value, index) => value - numbers[index] - 1),
    ];
    const rightDistances = [...leftDistances.slice(1), leftDistances[0]];

    numbers.forEach((number, position) => {
      const previousIndex = lastDrawn[number];
      const gap =
        previousIndex === null ? drawIndex : drawIndex - previousIndex - 1;
      maximumGap = Math.max(maximumGap, gap);
      const key = `${leftDistances[position]}|${rightDistances[position]}|${gap}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });

    numbers.forEach((number) => {
      lastDrawn[number] = drawIndex;
    });
  });

  const rows = [...counts.entries()].map(([key, count]) => {
    const [leftDist, rightDist, gap] = key.split("|").map(Number);
    return [leftDist, rightDist, gap, count];
  });
  rows.sort(
    (first, second) =>
      first[0] - second[0] ||
      first[1] - second[1] ||
      first[2] - second[2],
  );

  return { rows, maximumGap };
}

const yamlText = await fs.readFile(sourcePath, "utf8");
const history = parseHistory(yamlText);
const aggregation = aggregateTriplets(history.draws);
const encounterCount = history.draws.length * 6;
const rowStart = 5;
const rowEnd = rowStart + aggregation.rows.length - 1;

if (
  aggregation.rows.reduce((sum, row) => sum + row[3], 0) !== encounterCount
) {
  throw new Error("Triplet counts do not reconcile to the drawn-number total");
}

const workbook = Workbook.create();
const countsSheet = workbook.worksheets.add("Triplet Counts");
const aboutSheet = workbook.worksheets.add("About");

countsSheet.showGridLines = false;
countsSheet.mergeCells("A1:D1");
countsSheet.getRange("A1").values = [["Draw-history triplet frequencies"]];
countsSheet.getRange("A1:D1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  verticalAlignment: "center",
};
countsSheet.getRange("A1:D1").format.rowHeight = 32;

countsSheet.mergeCells("A2:D2");
countsSheet.getRange("A2").values = [[
  "One row per distinct (left_dist, right_dist, gap) observed in the official draw history",
]];
countsSheet.getRange("A2:D2").format = {
  fill: "#D9EAF7",
  font: { color: "#17365D", italic: true },
  verticalAlignment: "center",
  wrapText: true,
};
countsSheet.getRange("A2:D2").format.rowHeight = 38;

countsSheet.getRange("A4:D4").values = [[
  "left_dist",
  "right_dist",
  "gap",
  "encounter_count",
]];
countsSheet.getRange(`A5:D${rowEnd}`).values = aggregation.rows;
countsSheet.getRange(`A4:D${rowEnd}`).format.font = {
  name: "Aptos",
  size: 10,
};
countsSheet.getRange(`A5:D${rowEnd}`).format.numberFormat = "#,##0";
countsSheet.getRange(`A5:D${rowEnd}`).format.verticalAlignment = "center";

const table = countsSheet.tables.add(
  `A4:D${rowEnd}`,
  true,
  "TripletCountsTable",
);
table.style = "TableStyleMedium2";
table.showBandedRows = true;
table.showFilterButton = true;

countsSheet.getRange(`D5:D${rowEnd}`).conditionalFormats.add("dataBar", {
  color: "#5B9BD5",
  gradient: true,
});

countsSheet.getRange("F1:G1").values = [["History summary", "Value"]];
countsSheet.getRange("F2:F7").values = [
  ["Draws"],
  ["Number encounters"],
  ["Distinct triplets"],
  ["Largest gap"],
  ["First draw"],
  ["Last draw"],
];
countsSheet.getRange("G2").values = [[history.draws.length]];
countsSheet.getRange("G3").formulas = [[`=SUM(D5:D${rowEnd})`]];
countsSheet.getRange("G4").formulas = [[`=COUNTA(A5:A${rowEnd})`]];
countsSheet.getRange("G5").values = [[aggregation.maximumGap]];
countsSheet.getRange("G6:G7").values = [
  [history.firstDraw],
  [history.lastDraw],
];
countsSheet.getRange("F1:G1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
};
countsSheet.getRange("F2:F7").format = {
  fill: "#EAF2F8",
  font: { bold: true, color: "#17365D" },
};
countsSheet.getRange("F1:G7").format.borders = {
  preset: "outside",
  style: "thin",
  color: "#9FBAD0",
};
countsSheet.getRange("G2:G5").format.numberFormat = "#,##0";

countsSheet.getRange("A:D").format.columnWidth = 15;
countsSheet.getRange("A:C").format.columnWidth = 13;
countsSheet.getRange("D:D").format.columnWidth = 18;
countsSheet.getRange("E:E").format.columnWidth = 3;
countsSheet.getRange("F:F").format.columnWidth = 22;
countsSheet.getRange("G:G").format.columnWidth = 16;
countsSheet.freezePanes.freezeRows(4);

aboutSheet.showGridLines = false;
aboutSheet.mergeCells("A1:B1");
aboutSheet.getRange("A1").values = [["About this workbook"]];
aboutSheet.getRange("A1:B1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  verticalAlignment: "center",
};
aboutSheet.getRange("A1:B1").format.rowHeight = 32;
aboutSheet.getRange("A3:B9").values = [
  ["Item", "Details"],
  ["Source", sourcePath],
  ["History period", `${history.firstDraw} through ${history.lastDraw}`],
  ["Draws analyzed", history.draws.length],
  ["Number encounters", encounterCount],
  [
    "Triplet definition",
    "For each drawn number: unused values to its left, unused values to its right, and intervening draws since its prior appearance.",
  ],
  [
    "Inclusion rule",
    "The table contains every distinct triplet observed at least once. Unobserved combinations are omitted because the historical gap has no fixed theoretical upper bound.",
  ],
];
aboutSheet.getRange("A3:B3").format = {
  fill: "#5B9BD5",
  font: { bold: true, color: "#FFFFFF" },
};
aboutSheet.getRange("A4:A9").format = {
  fill: "#EAF2F8",
  font: { bold: true, color: "#17365D" },
};
aboutSheet.getRange("A3:B9").format.font.name = "Aptos";
aboutSheet.getRange("A3:B9").format.verticalAlignment = "top";
aboutSheet.getRange("B4:B9").format.wrapText = true;
aboutSheet.getRange("B4:B9").format.horizontalAlignment = "left";
aboutSheet.getRange("B6:B7").format.numberFormat = "#,##0";
aboutSheet.getRange("A3:B9").format.borders = {
  preset: "outside",
  style: "thin",
  color: "#9FBAD0",
};
aboutSheet.getRange("A:A").format.columnWidth = 24;
aboutSheet.getRange("B:B").format.columnWidth = 88;
aboutSheet.getRange("4:9").format.rowHeight = 30;
aboutSheet.getRange("8:9").format.rowHeight = 54;
aboutSheet.freezePanes.freezeRows(3);

await fs.mkdir(outputDir, { recursive: true });

const countsInspection = await workbook.inspect({
  kind: "table",
  range: "Triplet Counts!A1:G15",
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 7,
  maxChars: 10000,
});
const aboutInspection = await workbook.inspect({
  kind: "table",
  range: "About!A1:B9",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 4,
  maxChars: 8000,
});
const errorInspection = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});

const countsPreview = await workbook.render({
  sheetName: "Triplet Counts",
  range: "A1:G24",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  previewCountsPath,
  new Uint8Array(await countsPreview.arrayBuffer()),
);
const aboutPreview = await workbook.render({
  sheetName: "About",
  range: "A1:B9",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  previewAboutPath,
  new Uint8Array(await aboutPreview.arrayBuffer()),
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

console.log(
  JSON.stringify(
    {
      outputPath,
      previewCountsPath,
      previewAboutPath,
      drawCount: history.draws.length,
      encounterCount,
      distinctTriplets: aggregation.rows.length,
      maximumGap: aggregation.maximumGap,
      rowEnd,
      countsInspection: countsInspection.ndjson,
      aboutInspection: aboutInspection.ndjson,
      errorInspection: errorInspection.ndjson,
    },
    null,
    2,
  ),
);
