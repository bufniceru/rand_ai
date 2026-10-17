# LSTM (Experimental)

**LSTM (Experimental)** (`lstm`) is an optional standalone strategy in
**Settings → Strategies → Relationships & Machine Learning**. Enable it and
apply settings to calculate its forecasts alongside the other selected methods.
It is disabled by default and does not join existing fixed ensemble rosters.

## Model and availability

The desktop bundles a frozen five-seed ensemble: seven completed 49-element
presence vectors enter one 32-unit LSTM layer followed by 49 sigmoid outputs.
It uses NumPy only, without training or importing PyTorch in the application.
All 49 numbers are ranked by raw model estimate; ties prefer smaller numbers.
The first six ranks form the distinct-number selection. The full rankings
participate in prediction grids, comparisons, effectiveness histories,
portfolios and Possible Draw when available. Exports retain the selected ID.

The bundled model was selected using the dated prefix through reference draw
521 of `data/lotto_results_2019.yaml`. Its first eligible target is draw 522.
Earlier targets have no LSTM forecast and are excluded from its performance
denominator. Different, reordered, date-edited or truncated fitted prefixes
produce no LSTM ranking. The application never substitutes random or uniform
predictions for an unavailable forecast. Appending completed draws after the
matching fitted prefix is supported; the weights remain frozen.

## Evidence and interpretation

The historical research gate **failed**. In the 250-draw holdout the ensemble
averaged 0.716 matches per draw versus Bayesian's 0.744 and the fixed random
ticket baseline's 0.776. The comparisons did not establish an advantage.
Candidate details display the failed gate, activation reference and frozen
model description. These are uncalibrated model estimates, not guarantees or
a joint distribution over valid tickets.

The [research guide](../lstm-research.md) describes training, chronological
validation, synthetic controls, metrics, artifact validation and reproducibility.
The approved artifact is explicitly included in wheel and portable-bridge
packaging. A missing or corrupt packaged model is an installation error, not
a reason to silently fabricate forecasts. Research runs do not replace it.
