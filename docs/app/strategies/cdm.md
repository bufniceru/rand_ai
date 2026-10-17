# CDM — Compound Dirichlet–Multinomial

CDM (`cdm`) is a default-enabled standalone **Frequency & Recency** strategy.
It uses lifetime occurrence counts, including the first completed draw. It is
independent of Bayesian and is not added to fixed ensemble expert rosters.
Explicit saved strategy selections remain respected.

## Posterior predictive mean

Following the posterior-mean idea in
[Predicting Winning Lottery Numbers](https://arxiv.org/pdf/2403.12836), use a
symmetric Dirichlet prior with one pseudo-count per number. After \(T\) completed
six-number draws, let \(c_j\) be the number of appearances of number \(j\):

\[
q_j=\frac{c_j+1}{6T+49},\qquad E[Z_j]=6q_j.
\]

The categorical predictive shares sum to one; the expected counts sum to six.
Before any observations, every share is \(1/49\). After one draw, each of its
six numbers has share \(2/55\) and expected count \(12/55\); each other number
has share \(1/55\) and expected count \(6/55\).

## Ranking and application behavior

Larger expected counts rank first. Ties use larger current recurrence gap,
then smaller number. All 49 numbers are ranked; the first six form the selection.
Because every number has the same prior and denominator, this ordering is
identical to lifetime frequency ordering with the same tie-break.

The displayed score is min–max scaled to 0–1, with zero for an entirely tied
ranking. Candidate details retain the occurrence count, history length, prior,
predictive share, and expected count. These display scores are not probabilities.
CDM participates in the normal comparisons, effectiveness histories, portfolio
selection, Possible Draw, and exports when enabled. Forecasts use only completed
draws before the target; no future draw is used to update its counts.

## Interpretation limits

The multinomial approximation allows repeat categories within a draw. The
application's lottery draws contain six distinct numbers, so neither \(q_j\)
nor \(E[Z_j]\) is a calibrated probability of inclusion in a valid ticket.
CDM supplies a deterministic ranking, not a joint distribution over valid tickets.
The prior strength is fixed and no fitting or parameter controls are provided.

Smoothing does not establish an advantage over random selection. This strategy
does not implement the paper's diagonal estimator, positional rounding,
waiting-time rules, or escalating stakes.
