# Application overview

Rand AI brings the dataset lifecycle and its visual analysis into one desktop
workspace. Users can open a historical draw file, configure the reports they
want, inspect charts and tables, maintain a YAML-backed draw history, and save
portable outputs without running Python commands manually.

## Main capabilities

- Import `.yaml` or `.yml` draw histories and create a managed `.pkl` file.
- Open an existing `.pkl` or `.pickle` file after an explicit trust decision.
- Browse overview, number, spacing, relationship, randomness, correlation,
  comparison, audit, gap, and co-occurrence reports.
- Use workspace tabs for statistics, last-seen highlights, generated number
  grids, possible draws, portfolio review, and draw-history maintenance.
- Enable or disable report areas from Settings or the Reports menu.
- Customize the interface through importable and exportable color templates.
- Save analysis tables and metadata in a ZIP archive, save supported views as
  PDF, or send a supported comparison view to the system print dialog.

## Application lifecycle

1. Select a YAML or trusted pickle dataset.
2. Choose the reports and analysis options to enable.
3. Let the Python engine validate the data and build the analysis payload.
4. Navigate the enabled dashboard views and workspace tabs.
5. Reanalyze after an option or dataset change, or export the current result.

The welcome screen keeps up to ten recently opened datasets. A recent pickle
still requires trust confirmation when it is opened again.

## Boundaries

Rand AI is a local desktop application. It does not require a hosted service to
analyze a dataset. The renderer presents returned results but does not directly
read arbitrary files or start processes; those operations belong to the
Electron main process and the Python bridge.
