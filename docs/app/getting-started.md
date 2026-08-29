# Getting started

## Requirements

- Windows for the packaged desktop executable and the documented Electron
  packaging workflow.
- Python 3.14 or newer, managed with `uv`.
- Node.js and npm for the Vue and Electron application.

From the repository root, install the Python environment and documentation
tools:

```powershell
uv sync --group docs
```

Then install the desktop dependencies:

```powershell
Set-Location web
npm install
```

## Run the desktop application

Build the Electron-mode Vue bundle and open the desktop shell:

```powershell
Set-Location web
npm run electron
```

For live renderer development, use two terminals. Start Vite in the first:

```powershell
Set-Location web
npm run dev
```

Start Electron in the second:

```powershell
Set-Location web
$env:VITE_DEV_SERVER_URL = "http://127.0.0.1:5173"
npx electron .
```

Electron normally uses the project virtual environment to start the Python
bridge. Set `RAND_AI_PYTHON` to another Python executable when a different
environment is required.

## Open the first dataset

Choose **Open YAML or trusted pickle** on the welcome screen, or select
**File > Open YAML or Dataset**.

- Selecting YAML immediately creates a same-name managed pickle beside the
  source and starts analysis.
- Selecting an existing pickle opens a confirmation dialog. Continue only when
  the file came from a trusted source.

Analysis progress appears in the application while the dataset is prepared.
When processing completes, the configured workspace and report views become
available.

## Build the application documentation

From the repository root, run:

```powershell
uv run --group docs sphinx-build -W --keep-going -b html docs/app docs/_build/html
```

The result is `docs/_build/html/index.html`. On Windows, `docs/app/make.bat
html` runs the same warning-as-error build. On systems with Make, run `make
-C docs/app html`.
