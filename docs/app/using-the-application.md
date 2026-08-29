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

## Draw-history editor

The editor is available when the active pickle has a matching `.yaml` or
`.yml` source beside it. It supports adding a draw or updating an existing draw
by ISO date.

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
