# Application architecture

Rand AI separates desktop privileges, presentation, and data processing across
four runtime layers.

| Layer | Primary location | Responsibility |
| --- | --- | --- |
| Electron main process | `web/electron/main.cjs` | Windows, menus, file dialogs, settings, local caches, printing, and Python process management |
| Preload bridge | `web/electron/preload.cjs` | Narrow, typed-by-convention IPC surface exposed to the renderer |
| Vue renderer | `web/src/` | Navigation, dialogs, charts, tables, progress, and user interaction |
| Python engine | `src/rand_ai/` | Dataset validation, statistical analysis, persistence helpers, and export assembly |

## Request flow

1. The renderer requests an operation through `window.randAiDesktop`.
2. The sandboxed preload layer forwards the request to a named Electron IPC
   handler.
3. The main process validates paths and request shape, opens native dialogs
   when required, and starts the Python bridge for analysis operations.
4. The bridge writes progress records and a JSON result through its process
   streams.
5. Electron returns the parsed payload to Vue, which updates the active views.

Large portfolio calculations use a renderer worker to keep the interface
responsive. Electron stores compressed reusable inputs in its application data
directory, while the worker performs browser-side computation from those
inputs.

## Security boundary

The renderer runs with context isolation and Chromium sandboxing enabled, and
Node integration disabled. It cannot use Electron or Node APIs directly. The
preload file exposes only the operations required by the interface, such as
dataset selection, analysis, export, printing, settings, and draw editing.

The main process owns native privileges and limits file extensions and sizes
before invoking Python. The Python process receives explicit command-line
arguments and emits machine-readable JSON; it is not embedded in the renderer.

## Repository structure

```text
rand_ai/
|-- src/rand_ai/       Python data and analysis engine
|-- web/
|   |-- electron/      Desktop main and preload processes
|   `-- src/           Vue renderer, components, views, and workers
|-- tests/             Python tests
|-- docs/app/          Curated Sphinx application guide
|-- scripts/           Maintenance and build helpers
|-- pyproject.toml     Python project and dependency groups
|-- Jenkinsfile        Windows CI and packaging pipeline
`-- README.md          Repository entry point
```

This guide is written manually and does not import application modules during
the Sphinx build. That keeps documentation generation deterministic and limits
the site to application behavior and architecture.
