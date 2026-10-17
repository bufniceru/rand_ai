## Rand AI desktop

Rand AI is a Vue 3 and Electron desktop application backed by the existing
Python draw-statistics engine. The renderer owns all charts, tables,
navigation, controls, the application status bar, and the pylotto-compatible
last-seen and last-seen-gap highlight views. Dataset import also precomputes a
walk-forward prediction that combines exact freshness gaps with exact left/right
proximity spaces; its navigable 7x7 grid is available from the Prediction button
or View > Raw Combined Prediction.

### Development

Install the Python and desktop dependencies:

```powershell
uv sync
cd web
npm install
```

Build and open the Electron application:

```powershell
npm run electron
```

For live Vue development, start Vite and Electron in separate terminals:

```powershell
npm run dev
$env:VITE_DEV_SERVER_URL = "http://127.0.0.1:5173"
npx electron .
```

The Electron shell invokes `rand-ai-gui-bridge` with the project virtual
environment. Set `RAND_AI_PYTHON` to a different Python executable when
needed.

### Application documentation

The [Sphinx application guide](docs/app/index.md) covers installation, desktop
workflows, data handling, security, architecture, and contributor operations.
It also documents the [command palette and its on-demand statistics](docs/app/commands.md)
and the [Border Group SVC strategy](docs/app/strategies/border-group-svc.md).
The published guide is available at
[bufniceru.github.io/rand_ai](https://bufniceru.github.io/rand_ai/).
Build it locally with:

```powershell
uv run --group docs sphinx-build -W --keep-going -b html docs/app docs/_build/html
```

Open `docs/_build/html/index.html` after the build completes. The application
guide is intentionally separate from the algorithm-specific papers below.

### Statistics simplification study

Run the frozen exploratory comparison with:

```powershell
uv run python scripts/study_statistics.py
```

The runner uses `data/lotto_results_2019.yaml`, reserves 120 initial draws,
and compares six-number predictions across five chronological blocks. It
writes the [study report](reports/statistics_study.md), machine-readable
results, full rankings and predictions, and hit counts under `reports/`.
The report covers strategy redundancy, input removal experiments in SVC,
TBL and the online SVM, and up to three rank-blend experiments. Historical
results remain exploratory because this dataset previously influenced
strategy selection. Original statistics remain accessible in expandable
panels; saved settings, exports and internal strategy dependencies are preserved.

### Portfolio coverage research

With Python dependencies and `web/node_modules` installed, run from the repository root:

```powershell
uv run python scripts/study_portfolio.py --stage smoke
uv run python scripts/study_portfolio.py --stage historical
uv run python scripts/study_portfolio.py --stage synthetic
```

`--stage all` runs both full stages. Smoke evaluates 30 targets after 120 warm-up
draws, with ten tickets, on real history and three independent synthetic histories.
Full stages compare five, ten and twenty tickets; synthetic uses twenty histories
of 360 draws. The production default strategy roster is frozen in the manifest.

The runner bundles the actual TypeScript portfolio engine using Vite. It compares
the unchanged blended default with a research-only coverage-first objective,
100 equal-budget random portfolios per target from 1–49, and 100 from the same
selected pool. Random portfolios contain distinct tickets. It reports best-ticket
hits, at least one 3+ match, exact triple coverage, separate model-weighted coverage,
conditional guarantees, uncovered triples and runtime. Greedy coverage-first is
not a minimum-wheel solver and does not improve fair-draw jackpot odds per ticket.

Results live under `reports/portfolio_study/smoke/` and
`reports/portfolio_study/full/`: `report.md`, `aggregate.json`, `protocol.json`,
per-target audit and uncovered-triple JSONL, and dataset/budget checkpoints.
Rerunning resumes completed compatible jobs; changing data, source, roster or
protocol invalidates them. Historical and synthetic full stages share checkpoints
and combine completed results in the report. Each job is checkpointed after its
full dataset/budget comparison; an interrupted job restarts that job.

Paired comparisons use five chronological blocks, moving-block bootstrap intervals
and Holm-adjusted tests. Historical results remain exploratory; synthetic results
diagnose leakage and false signals. UI controls and application defaults are
unchanged. No Electron executable rebuild is needed for this study.

### Strategy documentation

- [Chi-Square Frequency strategy](docs/chi-square-frequency-strategy.md) —
  mathematical explanation, Python implementation, worked example, tests, and
  interpretation limits.
- [Exact-State Categorical Chi-Square strategy](docs/categorical-chi-square-strategy.md)
  — per-number gap and retained left/right-space dependency probabilities with
  hierarchical sparse-state backoff.
- [Nonlinear Dynamics, Recurrence Dynamics, and the SVC residual hybrids](docs/nonlinear-dynamics-strategy.md)
  — delay-embedded recurrence diagnostics, causal analogue ranking, the
  effectiveness-weighted experimental hybrids, limitations, and benchmarks.

### Opening a dataset

From the welcome screen or **File > Open YAML or Dataset**, select either:

- A `.yaml` or `.yml` Draws file. Rand AI generates a same-name `.pkl` beside
  the YAML source and starts analysis immediately.
- An existing `.pkl` or `.pickle` dataset. Rand AI asks for explicit trust
  confirmation before loading it.

The generated pickle remains paired with its YAML source for the Draw History
editor.

### Portable Windows build

```powershell
cd web
npm run electron:build
```

This builds the Python bridge with PyInstaller, creates the Vue bundle, and
packages both into a portable Electron executable.

To rebuild the portable executable automatically after source-file saves, run:

```powershell
cd web
npm run electron:build:hook
```

The hook debounces related file events and queues a follow-up build when files
change while an existing package build is still running.

### Security

YAML imports use the safe YAML loader and Rand AI trusts only the pickle it
generates from that selected source. Existing `.pkl` and `.pickle` datasets
still require explicit trust confirmation because Python pickle loading can
execute code; never analyze an unknown or untrusted pickle file.
