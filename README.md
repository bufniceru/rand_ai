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

The desktop app accepts `.pkl` and `.pickle` datasets only after an explicit
trust confirmation. Python pickle loading can execute code; never analyze an
unknown or untrusted file.
