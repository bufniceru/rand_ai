(command-system)=
# Command palette and on-demand statistics

Rand AI includes an application command system modeled on the Visual Studio
Code command palette. Commands are searched and run when needed instead of
occupying a permanent report or workspace panel. The initial command catalog
contains two whole-database statistics, and the registry is designed to accept
future statistical and general application commands.

## Opening and using the palette

Open the command palette with any of these methods:

- press **Ctrl+Shift+P** on Windows or Linux;
- press **Cmd+Shift+P** on macOS;
- press **F1**; or
- choose **View > Command Palette…**.

Type any part of a command's category, title, or keywords. Matching supports
ordinary substring matching and ordered fuzzy characters. A leading `>` is
optional. For example, `statistics`, `number expected`, and `border count`
locate relevant commands.

Use **Up Arrow** and **Down Arrow** to move through results, **Enter** to run
the selected command, and **Esc** to close the palette. Commands that need an
active dataset remain visible before analysis but display **Analyze a dataset
first** and cannot be executed.

## Result overlay

Running a statistics command closes the palette and opens a fixed result
overlay over the complete renderer, including its toolbar, workspaces, and
status bar. The overlay has loading, chart, and error states. It is not a
separate Electron window and has no window controls or close button.

Press **Esc** to discard the result and return focus to the control that was
active before the command palette opened. Palette shortcuts are ignored while
the result overlay is active. Clicking outside the result does not dismiss it.

Statistics commands always reload the complete active database and calculate
a fresh result. They do not use cached Statistics workspace tables, prepare
predictions, or persist their result. Report enablement, Statistics filters,
trend selections, and the currently visible workspace do not reduce the data
scope.

## Statistics: Number Frequency

Command identifier: `statistics.number-frequency`

**Statistics: Number Frequency** counts appearances of every lottery number
from 1 through 49 across every draw in the active database. The returned
statistical table contains all 49 rows and these fields:

| Field | Meaning |
|---|---|
| `number` | Lottery number from 1 through 49 |
| `count` | Draws containing the number |
| `appearance_rate` | Percentage of draws containing the number |
| `observation_percentage` | Percentage of all six-number observations |
| `expected_count` | Uniform 6/49 expectation, `draw_count × 6 / 49` |
| `deviation` | Observed count minus expected count |
| `standardized_residual` | Deviation divided by its binomial standard deviation |

The full-screen result displays observed counts as bars and the common uniform
expected count as a line. Hover details show the number and observed or
expected count. The structured table is used to construct the chart but is
not displayed as a separate result table.

Number Frequency is intentionally absent from the normal Overview and Numbers
workspaces. It is generated only when this command is run. Analysis export
archives may still contain the frequency table.

## Statistics: Group Frequency

Command identifier: `statistics.group-frequency`

**Statistics: Group Frequency** classifies every draw using the **Border
space** value currently selected in Settings. A circular space less than or
equal to the border connects adjacent numbers; a larger space separates two
groups. Each draw therefore receives one of the 11 canonical group-size
signatures:

| Groups | Signatures |
|---:|---|
| 1 | `6` |
| 2 | `5+1`, `4+2`, `3+3` |
| 3 | `4+1+1`, `3+2+1`, `2+2+2` |
| 4 | `3+1+1+1`, `2+2+1+1` |
| 5 | `2+1+1+1+1` |
| 6 | `1+1+1+1+1+1` |

The returned table always contains all 11 signatures in group-count order,
including signatures with zero draws. Each row contains `group_count`,
`signature`, and `count`. Every database draw contributes to exactly one row.

The chart keeps group counts 1 through 6 on the x-axis and draw counts on the
y-axis. Within each group-count category, one colored bar is shown for every
possible signature. Visible bars contain a label such as `5+1 · 123`; the
hover detail also reports the dataset, Border space, group count, signature,
and draw count. There is no separate total bar: signature bars within a group
count add up to that category's total.

Changing **Border space** and running the command again reclassifies the
complete database. **Predicted groups**, strategy settings, report enablement,
and workspace filters do not affect this statistic.

## Command architecture for contributors

Renderer commands are registered in `web/src/lib/commands.ts`. Each command
defines an identifier, title, category, search keywords, availability rule,
disabled reason, and asynchronous executor. The registry is generic even
though the first commands are statistics.

Statistics requests use a discriminated TypeScript request type. Electron
validates the command identifier and its parameters against a whitelist, then
invokes the Python `statistics-command` bridge operation. Python validates the
same whitelist, reloads the trusted active dataset without preparing
predictions, calculates the table, and returns the command identifier, dataset
name, draw count, optional Border space, and serialized table.

To add a command safely:

1. add its identifier and request/result shape to the public renderer types;
2. register its metadata, availability, and executor in the command registry;
3. validate any bridge-bound identifier and parameters in Electron and Python;
4. construct a renderer result such as a responsive Plotly figure; and
5. test filtering, disabled behavior, execution, payload validation, result
   construction, keyboard operation, and failure handling.

Unknown statistics identifiers are rejected at both bridge boundaries rather
than being interpreted as arbitrary operations.

## Proposed command catalog

The ordered backlog for future statistical, strategy-diagnostic, and general
application commands is documented in {ref}`proposed-command-catalog`. Each
entry defines its purpose, calculation or effect, parameters, presentation,
dependencies, failure behavior, strategy associations, and acceptance test.
The proposal does not make a command executable; this page remains the source
of truth for commands that are currently available.
