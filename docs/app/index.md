# Rand AI application guide

Rand AI is a Windows desktop application for importing historical lottery
draws, exploring statistical reports, reviewing generated number sets, editing
managed draw history, and exporting analysis results. A Vue interface runs
inside a security-hardened Electron shell and delegates data processing to a
Python engine.

This guide serves both application users and contributors. It describes what
the application does, how its processes exchange data, and how to operate and
build it. The numbered application-guide chapters remain independent of
strategy implementation details. Detailed descriptions of selected ranking
engines are collected separately under **Strategies**.

```{admonition} Interpretation limit
:class: caution

Rand AI summarizes and transforms historical data. Lottery draws are random,
and the application's reports or generated number sets cannot guarantee a
future result. Use the software for exploration rather than as evidence of a
predictable outcome.
```

```{toctree}
:maxdepth: 2
:numbered:
:caption: Application guide

overview
getting-started
using-the-application
architecture
data-and-security
contributing
```

```{toctree}
:maxdepth: 2
:caption: Strategies

strategies/index
```
