# Strategies

This catalog documents selected Rand AI ranking engines separately from the
general application guide. Each entry explains the production behavior,
mathematical model, causal safeguards, interpretation, and limitations of one
strategy.

Strategy documentation is descriptive, not a claim that lottery outcomes are
predictable. Historical replay results are retrospective measurements and do
not guarantee future performance.

## Catalog

The category order and names below match **Settings → Strategies**. Only
strategies with a dedicated guide are linked; an empty category is retained so
the documentation catalog and application settings use the same structure.

### Frequency & Recency

```{toctree}
:maxdepth: 1

chi-square-frequency
categorical-chi-square
entropy
bayesian
```

### Shape & Similarity

```{toctree}
:maxdepth: 1

earth-mover-distance
recurrence-dynamics
predictive-score-grid
```

### Markov & Sequence

```{toctree}
:maxdepth: 1

markov-freshness
markov-spaces
markov-normalized-positions
markov-relative-dispersion
doublet-triplet-markov
```

### Relationships & Machine Learning

```{toctree}
:maxdepth: 1

support-vector-classifier
temporal-behavior-learning
scikit-online-svm
lagged-logistic
```

### Ensembles & Coverage

```{toctree}
:maxdepth: 1

svc-recurrence-hybrid
svc-recurrence-proximity-hybrid
srph-residual-diversity-hybrid
decision-tree-selector
residual-coverage
```

### Border Space Groups

```{toctree}
:maxdepth: 1

border-group-statistical
border-group-markov
border-group-bayesian
border-group-ml
border-group-hybrid
```

### Random Baselines

```{toctree}
:maxdepth: 1

random-baseline
fresh-random
```
