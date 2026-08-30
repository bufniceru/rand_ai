# Contributor guide

## Development responsibilities

The Python package owns dataset objects, validation, statistics, persistence,
and the JSON/ZIP bridge contract. Electron owns operating-system integration
and privileged operations. Vue owns presentation state and interaction.

Keep those boundaries intact when adding application features:

- expose the smallest necessary IPC operation through the preload layer;
- validate privileged requests again in the Electron main process;
- keep renderer code independent of direct Node access;
- return serializable bridge payloads and report long-operation progress;
- preserve YAML as the editable source for managed datasets.

## Python checks

From the repository root:

```powershell
uv run pytest
uv run ruff check .
uv run ty check src
```

The Python test configuration enforces 100% package coverage. Add tests for new
bridge inputs, validation failures, payload fields, and persistence behavior.

## Frontend checks

From `web/`:

```powershell
npm test
npm run build
```

The production build runs TypeScript checking before Vite creates the renderer
bundle. Cover data transformations and state-independent helpers with Vitest;
exercise desktop IPC changes through the existing Electron-facing tests where
available.

## Documentation checks

Build the site after changing application behavior or contributor commands:

```powershell
uv run --group docs sphinx-build -W --keep-going -b html docs/app docs/_build/html
```

Warnings fail the build. Keep internal links valid, ensure examples match the
current interface, and keep the general application chapters independent of
ranking algorithms. Put reviewed implementation-specific strategy material in
the dedicated **Strategies** section.

## Packaging and CI

Jenkins performs a locked Python sync including the documentation group, builds
this guide, installs frontend dependencies, and creates the portable Electron
executable. After the complete pipeline succeeds for `master`, Jenkins publishes
the generated Sphinx HTML to the root of the orphan `gh-pages` branch. Other
source branches validate the documentation but do not update the public site.
The generated output remains ignored on `master` and is never merged back from
`gh-pages`.

Pages publication uses the Jenkins SSH Agent plugin and the existing
`github-rand-ai-ssh` credential. That GitHub SSH key must have repository write
access. Configure GitHub Pages to deploy from the `gh-pages` branch and its
repository root. Jenkins skips the publication commit when the generated site
has not changed.

Portable executable packaging is handled by Jenkins. Local contributors should
run the Python suite, frontend tests, frontend production build, and Sphinx
build unless a packaging change specifically requires a local executable.
