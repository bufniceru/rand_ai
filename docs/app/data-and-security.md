# Data and security

## YAML format

YAML is the preferred editable dataset format. The document contains a
`lotto_results` mapping and a chronological list of draws:

```yaml
lotto_results:
  total_draws: 2
  first_draw: '2026-01-03'
  last_draw: '2026-01-10'
  draws:
    - date: '2026-01-03'
      numbers: [3, 11, 18, 27, 35, 44]
    - date: '2026-01-10'
      numbers: [5, 14, 22, 31, 38, 47]
```

Rand AI uses the safe YAML loader. Importing YAML creates a pickle with the
same base name in the same directory. Keeping the pair together enables the
draw-history editor.

## Pickle trust

```{warning}
Python pickle files can execute code while loading. Never approve a pickle from
an unknown or untrusted source.
```

A pickle generated from YAML selected during the current import flow is
managed by the application. An existing `.pkl` or `.pickle` file requires an
explicit trust confirmation each time it is selected for analysis.

## Local application data

Electron stores recent dataset metadata, the active color template, analysis
caches, and compressed portfolio inputs beneath its platform-specific user-data
directory. These files improve startup and repeat-analysis performance; they
are not substitutes for the source dataset.

Recent entries may contain local filesystem paths. Clear the Recent Datasets
menu before sharing an application profile or diagnostic capture if those paths
are sensitive.

## Validation and failure handling

- Dataset selection is limited to `.yaml`, `.yml`, `.pkl`, and `.pickle`.
- Selected inputs and generated managed pickles must not exceed 100 MiB.
- Draws must contain six unique, valid values and a valid ISO date.
- Draw-history updates reject duplicate dates and stale edit targets.
- Invalid or obsolete caches are discarded and regenerated.
- Export and PDF writes use native save dialogs so the destination is explicit.

When analysis fails, the application reports the bridge error and leaves the
source dataset untouched. YAML editing writes the source first and then
regenerates its paired pickle, so the two formats remain synchronized after a
successful save.
