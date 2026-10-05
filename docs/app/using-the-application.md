# Using the application

## Datasets and recent files

Open a dataset from the welcome screen or the File menu. The application
accepts YAML and pickle extensions and rejects files larger than 100 MiB.
Recently opened entries show their path and last-opened time. Use
**File > Recent Datasets** to reopen or clear the list.

The active dataset remains selected until another dataset is opened. Press
**F5** or choose **Analyze > Reanalyze** to rebuild the current analysis.

## Navigation and report configuration

The main navigation exposes enabled report views. The workspace tabs provide
related tables, highlights, generated grids, portfolio tools, possible-draw
tools, and the draw-history editor. The **View** menu offers keyboard shortcuts
for frequently used tabs.

Use **File > Settings** or the **Reports** menu to control which report areas
are enabled. Disabling a report hides its related navigation entry without
altering the source dataset. Reanalysis applies changed calculation options to
the active dataset.

## Strategy Hit Statistics

Enable **Strategy Hit Statistics** in Settings or the Reports menu, then open
its navigation entry. It works independently of Prediction Audit and Strategy
Effectiveness and uses the same walk-forward predictions: a hit means the
strategy's prior Top 6 included an actual winning number.

Choose all evaluated draws or the latest 100, 250, or 500. The selected range
applies to draw selection, strategy totals, and relationships. The latest
evaluated draw is selected initially. Draw details list every winning number
and its successful strategies, plus each strategy's matched numbers and hit
count, including zero hits. Totals show hits, draws with hits, averages per
evaluated draw, and the distribution of exactly zero through six hits.

The relationship heatmap and sortable pair table show **Shared winning
numbers** (distinct draw-and-number events) and **Both hit in the same draw**
(including hits on different numbers). The table also counts draws with shared
hits. Click a heatmap cell or pair to inspect contributing draws and matched
numbers; click a draw in the detail table to select it above. Self-pairs are
excluded and each unordered pair appears once in the table. These observed
relationships do not establish predictive causation. Enabled random strategies
are included. Only draws with recorded prior predictions are evaluated.

## Command palette

Press **Ctrl+Shift+P** (or **Cmd+Shift+P**), press **F1**, or choose **View >
Command Palette…** to search and run application commands. Use the arrow keys
to select a result, **Enter** to execute it, and **Esc** to close the palette.
Commands that require analysis remain visible but disabled until a dataset is
active.

The initial commands generate Number Frequency and detailed Border Group
Frequency charts against the complete active database. Results cover the full
application and are dismissed with **Esc**. See {ref}`command-system` for the
complete interaction, calculation, and extension guide.

## Draw-history editor

The editor is available when the active pickle has a matching `.yaml` or
`.yml` source beside it. It supports adding a draw or updating an existing draw
by ISO date.

In viewing mode, hover over a drawn number in the **7×7 Grid** or **PyLotto
Circle**, or focus it with the keyboard, to see which enabled strategies
included it in their prior Top 6. The hover window shows full strategy names,
the hit count, draw, and date. Move onto the window to read or scroll a long
list; press **Esc** to dismiss it. Non-drawn numbers have no hit window.

These records are available independently of the Prediction Audit and Strategy
Hit Statistics report switches. Enabled strategies are evaluated during
analysis and cached results are reused. A draw without a recorded forecast
shows “No prior prediction available for this draw.” After editing history,
reanalyze before inspecting refreshed hits.

When a draw is saved, Rand AI validates that it contains six unique numbers,
sorts the history by date, updates the YAML metadata, and regenerates the
paired pickle. Duplicate dates and invalid draw values are rejected.

```{important}
The YAML file is the editable source of truth. Do not edit the managed pickle
directly.
```

## Appearance

The appearance dialog changes the application's color template. Templates can
be loaded from or saved to JSON, and the selected template is retained in the
application data directory. Invalid template kinds, schema versions, or color
values are rejected before application.

## Printing, PDF, and ZIP output

Supported comparison and portfolio views can be saved as PDF. The latest-draw
comparison can also be sent to the system print dialog. Background colors are
included in both operations.

Choose **Export** or **File > Export Analysis** to create a ZIP archive. The
archive contains:

- `metadata.json`, describing the dataset and analysis options;
- one CSV file per exported table under `tables/`.

Exports represent the current active analysis. If configuration changes, run
the analysis again before exporting the updated result.
