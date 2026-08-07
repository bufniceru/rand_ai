# Chi-Square Frequency Strategy

## 1. Purpose

The **Chi-Square Frequency** strategy ranks the 49 lottery numbers by how far
their historical appearance counts are above or below the count expected from
a uniform 6-from-49 lottery.

In one sentence, the strategy does this:

> Count how often every number has appeared, compare every count with the same
> fair-lottery expectation, give the highest scores to the most overrepresented
> numbers, and select the first six numbers after sorting.

The strategy ID used by the application is `chi_square`. Its short display name
is `Chi²`.

This document explains:

- the probability and chi-square mathematics;
- what a signed Pearson residual means;
- how the mathematical values become scores from 0 to 1;
- how ties are resolved and the Top 6 is chosen;
- how historical draws flow through the Python implementation;
- which Python libraries are and are not used;
- how the automated test verifies the calculation; and
- the statistical limits of the method.

## 2. What the strategy predicts—and what it does not

The result is a **ranking**, not a calculated probability that a number will be
drawn next. Each number receives:

- a score between 0 and 1;
- a rank from 1 through 49;
- its current gap, meaning the number of completed draws since it last appeared;
- explanatory text containing its observed count, expected count, signed
  residual, and chi-square contribution.

The first six ranked numbers become `top_numbers`.

The strategy follows a “hot-number” idea: a number that has appeared more often
than expected receives a larger signed residual and therefore a better rank.
This describes the **past data**. In a genuinely independent and fair lottery,
past overrepresentation does not increase the number's probability in the next
draw. Every number still has probability `6 / 49` of appearing in that draw.

## 3. The 6-from-49 mathematical model

### 3.1 Symbols

The implementation uses these quantities:

| Symbol | Python value | Meaning |
|---|---:|---|
| $N$ | `_NUMBER_COUNT = 49` | Number of possible values, 1 through 49 |
| $K$ | `_NUMBERS_PER_DRAW = 6` | Unique numbers selected in one draw |
| $D$ | `self.draw_count` | Number of historical draws remembered so far |
| $O_i$ | `self.appearances[i]` | Observed appearances of number $i$ in those draws |
| $p$ | `6 / 49` | Probability that one particular number appears in a fair draw |
| $E$ | `D * 6 / 49` | Expected appearances of every particular number |

The source constants are defined in
[`strategy_prediction.py`](../src/rand_ai/strategy_prediction.py#L25-L28).

### 3.2 Why the probability is 6/49

A draw selects 6 distinct numbers from 49. Because the lottery treats all 49
numbers equally, one fixed number occupies 6 of the 49 available selection
places in probability terms. Therefore,

$$
p = \frac{K}{N} = \frac{6}{49} \approx 0.122449.
$$

So a particular number has about a **12.2449%** chance of being present in one
fair draw.

This can also be proved with combinations. There are
$\binom{49}{6}$ possible six-number sets. The sets containing a chosen number
are formed by choosing the other five numbers from the remaining 48, so there
are $\binom{48}{5}$ such sets. Thus,

$$
\frac{\binom{48}{5}}{\binom{49}{6}} = \frac{6}{49}.
$$

### 3.3 Expected number of appearances

If there are $D$ historical draws, the expected appearance count for one
number is

$$
E = Dp = D\left(\frac{6}{49}\right).
$$

For example, after 49 draws,

$$
E = 49\left(\frac{6}{49}\right) = 6.
$$

An expectation does not have to be a whole number. After 100 draws, for
example, $E = 600/49 \approx 12.245$. It is a long-run average, not a claim
that every number must appear exactly that many times.

As a useful check, the 49 expected counts add up to the total number of balls
observed:

$$
49E = 49\left(D\frac{6}{49}\right) = 6D.
$$

The 49 actual counts also add up to $6D$, because every stored draw contains
exactly six unique values.

## 4. From frequency difference to chi-square mathematics

### 4.1 Raw difference

For number $i$, the simplest comparison is

$$
d_i = O_i - E.
$$

- $d_i > 0$: the number appeared more often than expected;
- $d_i < 0$: the number appeared less often than expected;
- $d_i = 0$: the count exactly matches the expectation.

A raw difference is hard to compare between histories of different sizes. A
difference of 5 is large in a short history but can be small in a very long
history. Pearson scaling addresses this by dividing by $\sqrt{E}$.

### 4.2 Signed Pearson residual

The strategy's main mathematical value is the **signed Pearson residual**:

$$
r_i = \frac{O_i-E}{\sqrt{E}}.
$$

The word “signed” is important. The sign preserves direction:

- a positive residual means **over** the expected frequency;
- a negative residual means **under** the expected frequency.

The code calculates it as:

```python
difference = observed - expected
residual = difference / math.sqrt(expected) if expected > 0 else 0.0
```

This is implemented in
[`_chi_square_scores()`](../src/rand_ai/strategy_prediction.py#L2377-L2405).

### 4.3 Chi-square contribution

The ordinary Pearson chi-square contribution for number $i$ is

$$
c_i = \frac{(O_i-E)^2}{E}.
$$

Because

$$
r_i^2 = \left(\frac{O_i-E}{\sqrt{E}}\right)^2
      = \frac{(O_i-E)^2}{E},
$$

the contribution is exactly the square of the signed residual:

$$
c_i = r_i^2.
$$

The contribution is always zero or positive. It measures the **size** of a
deviation but loses its direction. For example, residuals `-2` and `+2` both
have a chi-square contribution of `4`.

The implementation shows the contribution in the details for the user, but it
does **not** rank by contribution. Ranking by contribution would mix strongly
underrepresented and strongly overrepresented numbers. Ranking by the signed
residual deliberately puts overrepresented numbers first.

### 4.4 Relationship to the global chi-square statistic

A traditional goodness-of-fit statistic adds all 49 contributions:

$$
\chi^2 = \sum_{i=1}^{49}\frac{(O_i-E)^2}{E}
       = \sum_{i=1}^{49}r_i^2.
$$

That global statistic answers a question such as “How different is the complete
frequency distribution from a uniform distribution?” A hypothesis test may
then convert it to a p-value.

The prediction strategy does not calculate that sum and does not calculate a
p-value. It needs a directional score for each individual number, so it retains
the 49 signed residuals instead. The separate statistics module does calculate
a global statistic and p-value with SciPy; see
[`Statistics.randomness_diagnostics()`](../src/rand_ai/statistics.py#L461-L495).
That diagnostic and this ranking strategy are related mathematically but serve
different purposes.

### 4.5 Descriptive bands

Each residual receives a label using fixed thresholds:

| Residual $r_i$ | Label | Plain-language meaning |
|---:|---|---|
| $r_i \le -2$ | `Strong under` | Far below the expected count |
| $-2 < r_i \le -1$ | `Mild under` | Moderately below the expected count |
| $-1 < r_i < 1$ | `Near expected` | Relatively close to expectation |
| $1 \le r_i < 2$ | `Mild over` | Moderately above the expected count |
| $r_i \ge 2$ | `Strong over` | Far above the expected count |

These labels are descriptive categories. They are not p-values and are not
proof that the lottery is biased.

## 5. Converting residuals to application scores

The raw residual can be negative and has no fixed upper bound. The application
converts all 49 residuals to the interval from 0 to 1 with min-max scaling.

Let

$$
r_{\min}=\min_i(r_i), \qquad r_{\max}=\max_i(r_i).
$$

Then the displayed score is

$$
s_i = \frac{r_i-r_{\min}}{r_{\max}-r_{\min}}.
$$

Consequently:

- the smallest residual receives score 0;
- the largest residual receives score 1;
- all other values are placed proportionally between them.

The reusable implementation is
[`_scale_scores()`](../src/rand_ai/strategy_prediction.py#L376-L382).
If every raw value is equal, the spread is zero and the function returns score
`0.0` for every number, avoiding division by zero.

### 5.1 A useful simplification

Within one prediction, every number has the same $E$ and therefore the same
$\sqrt{E}$. Substituting the residual formula into the scaling formula causes
the common expectation terms to cancel:

$$
s_i = \frac{O_i-O_{\min}}{O_{\max}-O_{\min}}.
$$

Therefore, for a fixed history, min-max-scaled signed Pearson residuals have the
same ordering and the same normalized values as min-max-scaled appearance
counts. The residual calculation is still valuable because it supplies:

- an interpretable comparison with the fair-lottery expectation;
- the under/near/over bands;
- the signed standardized value; and
- the chi-square contribution shown in the details.

It is important not to read `score = 0.8` as an 80% chance of appearing. It is
only a relative position between the smallest and largest residual in the
current 49-number set.

## 6. Ranking and selecting the Top 6

After scoring, the shared ranking helper sorts using this key:

```python
key=lambda number: (-scores[number], -gaps[number], number)
```

The complete rule is:

1. Higher score first.
2. If scores are equal, longer current gap first.
3. If both score and gap are equal, smaller number first.

This is implemented by
[`_ranking_from_scores()`](../src/rand_ai/strategy_prediction.py#L425-L432).
The first six entries become the strategy's `top_numbers` in
[`_strategy()`](../src/rand_ai/strategy_prediction.py#L385-L410).

Because equal scores in this strategy normally mean equal historical counts,
the gap acts only as a tie-breaker between equally frequent numbers. It does not
otherwise change their chi-square scores.

## 7. Fully worked example

The unit test uses a 49-draw history. The expected count is therefore 6. It
assigns sample observed counts of 0, 2, 6, 10, and 12 to numbers 1 through 5.

Since $\sqrt{6}\approx2.44949$, the calculations are:

| Number | Observed $O_i$ | Difference $O_i-E$ | Residual $r_i$ | Contribution $r_i^2$ | Band | Scaled score* |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 0 | -6 | -2.449 | 6.000 | Strong under | 0.000 |
| 2 | 2 | -4 | -1.633 | 2.667 | Mild under | 0.167 |
| 3 | 6 | 0 | 0.000 | 0.000 | Near expected | 0.500 |
| 4 | 10 | +4 | +1.633 | 2.667 | Mild over | 0.833 |
| 5 | 12 | +6 | +2.449 | 6.000 | Strong over | 1.000 |

`*` In the test state, the remaining numbers retain count 0, so the minimum
count is 0 and the maximum is 12. The simplified score is therefore
$s_i=O_i/12$.

For number 5, the exact steps are:

$$
E=6,
$$

$$
r_5=\frac{12-6}{\sqrt{6}}\approx2.449,
$$

$$
c_5=\frac{(12-6)^2}{6}=\frac{36}{6}=6,
$$

$$
s_5=\frac{12-0}{12-0}=1.
$$

The automated version of this example is
[`test_chi_square_ranks_signed_frequency_residuals()`](../tests/test_strategy_prediction.py#L669-L688).

## 8. Python implementation, step by step

### 8.1 Validated input draws

`Draw` represents exactly six unique, sorted integers from 1 through 49. Input
validation occurs before strategy processing. This guarantees the 6/49 model
used by the strategy matches the stored data format.

### 8.2 Constants

The module defines:

```python
_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASE_PROBABILITY = _NUMBERS_PER_DRAW / _NUMBER_COUNT
```

The chi-square expectation uses the first two constants directly. Centralizing
them prevents unrelated hard-coded copies of 49 and 6 inside the method.

### 8.3 Incremental state

`_StrategyState` stores frequency information without repeatedly scanning the
whole draw history:

```python
self.draw_count = 0
self.appearances = [0] * (_NUMBER_COUNT + 1)
```

The list contains 50 slots so that number `i` can be read directly from index
`i`. Index 0 is unused. See
[`_StrategyState.__init__()`](../src/rand_ai/strategy_prediction.py#L673-L695).

When `remember()` receives a completed draw, it loops over its six sorted
numbers and increments each corresponding counter:

```python
self.appearances[number] += 1
```

After all current-draw state is updated, it increments `self.draw_count`. See
[`remember()`](../src/rand_ai/strategy_prediction.py#L2151-L2170) and the final
increment in that method.

### 8.4 Score calculation

`_chi_square_scores()` performs one pass over numbers 1 through 49:

1. Calculate the common expected count.
2. Read the observed count for one number.
3. Calculate difference, residual, and contribution.
4. Assign the descriptive band.
5. Save the residual and four detail strings.
6. Min-max scale all residuals after the loop.

Its return type is a pair:

```python
tuple[dict[int, float], dict[int, tuple[str, ...]]]
```

The first dictionary maps each number to its normalized score. The second maps
each number to text suitable for display, for example:

```text
Strong over
Observed 12 vs expected 6.00
Signed Pearson residual +2.449
Chi-square contribution 6.000
```

The formatting deliberately uses two decimal places for expectation and three
for the residual and contribution.

### 8.5 Building the strategy result

`build_strategies()` calculates chi-square data only when `chi_square` is in
the enabled set. If the strategy was requested for display, it calls the shared
`_strategy()` constructor with:

- strategy ID: `chi_square`;
- display name: `Chi²`;
- description: `Signed Pearson residual from the uniform 6/49 frequency expectation.`;
- the 49 scores;
- the 49 current gaps; and
- the 49 detail tuples.

See the
[`chi_square` build block](../src/rand_ai/strategy_prediction.py#L4440-L4454).
The resulting immutable `StrategyPrediction` contains 49
`StrategyNumberPrediction` records plus the Top 6 tuple. The data classes are
defined in
[`strategy_prediction.py`](../src/rand_ai/strategy_prediction.py#L307-L340).

### 8.6 Walk-forward timing and protection from future data

`build_prediction_suites()` processes draws in chronological order. For each
reference draw, it:

1. trains models that require a pre-draw state;
2. remembers the current completed draw;
3. builds rankings intended for the following draw;
4. compares the ranking with the following draw only for evaluation; and
5. moves forward to the next reference draw.

For the chi-square strategy, this means the ranking for target draw $t+1$
uses frequency counts through reference draw $t$, including draw $t$, but
does not include the target draw. This is a walk-forward or leakage-safe
evaluation. The orchestration is in
[`build_prediction_suites()`](../src/rand_ai/strategy_prediction.py#L5015-L5083).

## 9. Libraries and language features

### 9.1 Libraries used directly by this strategy

The core chi-square ranking is intentionally small and uses only Python
features plus one standard-library function:

| Tool | Use in the strategy |
|---|---|
| `math.sqrt` | Calculates $\sqrt{E}$ for the Pearson residual |
| `dict` | Stores scores and display details by number |
| `list` | Stores incremental appearance counts and the final ordering |
| `sorted` | Ranks all numbers with deterministic tie-breakers |
| type hints | State and return-shape documentation checked by development tools |
| `dataclasses.dataclass` | Defines immutable output records shared by all strategies |

No third-party numerical library is required by `_chi_square_scores()`.

### 9.2 Project libraries that are not part of this calculation

The strategy module also imports NumPy and scikit-learn because other strategies
in the same file need them. The project's dependency list includes SciPy,
pandas, and other packages. They do not calculate the per-number chi-square
strategy scores.

In particular:

- **NumPy** is not used by `_chi_square_scores()`.
- **scikit-learn** is not used by `_chi_square_scores()`.
- **SciPy's `scipy.stats.chisquare`** is used by the separate randomness
  diagnostics, not by this prediction strategy.
- **pytest** is a development dependency used to verify the behavior.

The Python and development dependencies are declared in
[`pyproject.toml`](../pyproject.toml).

## 10. Calling the strategy from Python

The strategy normally runs as part of the complete application pipeline. A
small program can request only this strategy as follows:

```python
from rand_ai import Draw, Draws
from rand_ai.strategy_prediction import build_prediction_suites

draws = Draws()
draws.add(Draw(1, 2, 8, 17, 31, 49))
draws.add(Draw(3, 6, 12, 22, 36, 47))
draws.add(Draw(1, 9, 18, 27, 38, 45))

# build_prediction_suites expects each Draw to have its combined prediction
# prepared, even when only the chi-square plugin is requested.
draws.prepare_predictions()

suites = build_prediction_suites(
    draws.draws,
    enabled_strategy_ids=("chi_square",),
)

latest = suites[-1].strategies[0]
print(latest.name)         # Chi²
print(latest.top_numbers)  # Six highest-ranked numbers

for candidate in latest.numbers[:6]:
    print(
        candidate.rank,
        candidate.number,
        candidate.score,
        candidate.gap,
        candidate.details,
    )
```

`latest.numbers` is in rank order. A candidate's `score` is the normalized
relative score, while its raw residual and contribution are available in the
human-readable `details` tuple.

## 11. Automated test and verification

The focused unit test constructs `_StrategyState(("chi_square",))`, sets a
controlled 49-draw state, and checks all major behaviors:

- minimum residual maps to score 0;
- expected frequency maps to score 0.5 in the chosen data;
- maximum residual maps to score 1;
- all five descriptive bands are reached at the intended values;
- expected count formatting is `6.00`; and
- the chi-square contribution for the 12-count example is `6.000`.

Run only this focused test from the repository root with:

```powershell
uv run pytest tests/test_strategy_prediction.py::test_chi_square_ranks_signed_frequency_residuals
```

Run the complete test suite with:

```powershell
uv run pytest
```

The project configures pytest to collect tests from `tests/` and requires 100%
coverage for the complete run.

## 12. Edge cases

### No remembered draws

When `draw_count == 0`, the expectation is 0. The code explicitly returns a raw
residual and contribution of 0 for every number instead of dividing by zero.
Min-max scaling then returns score 0 for every number. Ranking is resolved by
gap and then number; with an entirely empty state, that ultimately means
ascending number order.

### Every number has the same count

If all residuals are identical, `maximum - minimum == 0`. `_scale_scores()`
returns all-zero scores. Gaps and then numeric order decide the ranking.

### Small histories

In a short history, the expected count can be much less than 1 and the residual
bands can change sharply after a single appearance. The code intentionally has
no minimum-history gate for this strategy. Early rankings should therefore be
interpreted cautiously.

### Tied frequencies

Equal observed counts produce equal residuals and equal normalized scores.
Their current gaps decide the order. If their gaps are also equal, the smaller
lottery number is listed first, making the result deterministic.

## 13. Complexity and performance

Let $N=49$.

- Updating appearance counts for one draw touches its 6 numbers: $O(6)$,
  effectively constant time.
- Calculating residuals touches all 49 numbers: $O(N)$.
- Min-max scaling touches all 49 numbers: $O(N)$.
- Sorting the ranking costs $O(N\log N)$.
- The frequency counters require $O(N)$ memory.

Since `N` is fixed at 49, the real runtime and memory cost are very small. The
incremental counters also avoid rescanning all $D$ historical draws every
time a new ranking is built.

## 14. Statistical interpretation and limitations

### 14.1 A high residual describes history, not future probability

Suppose number 5 has residual `+2.449`. This says its observed historical count
is 2.449 Pearson units above the uniform expectation. It does not say number 5
has a 2.449-times greater probability in the next draw.

### 14.2 Ordinary lottery independence does not “correct” hot or cold numbers

If draws are independent, a number does not become more likely because it is
hot and does not become “due” because it is cold. The strategy is a historical
frequency heuristic whose predictive value must be judged by walk-forward
results against a random baseline.

### 14.3 Residual is not a binomial z-score

For one number across independent draws, its count has marginal variance
$Dp(1-p)$. A binomial z-score would divide by
$\sqrt{Dp(1-p)}$. The Pearson cell residual implemented here divides by
$\sqrt{E}=\sqrt{Dp}$, matching a cell contribution in the Pearson chi-square
formula. These quantities are related but not identical.

### 14.4 The six values within a draw are selected without replacement

Numbers in one draw are not independent categorical observations: selecting
one value prevents that same value from being selected again in the draw. This
matters for strict hypothesis-test assumptions. The application uses the
Pearson-style per-number values descriptively and as ranking features; it does
not present the strategy's bands as formal significance claims.

### 14.5 Scores are relative to the current history

A score of 1 only means “largest residual among these 49 numbers now.” The raw
residual behind score 1 could be small in one history and large in another.
Compare raw residuals or contributions when the magnitude matters; compare
normalized scores when the ranking matters.

## 15. Formula and code reference sheet

| Stage | Formula or rule | Python location |
|---|---|---|
| Fair per-draw probability | $p=6/49$ | Constants near module start |
| Expected count | $E=D(6/49)$ | `_chi_square_scores()` |
| Difference | $d_i=O_i-E$ | `_chi_square_scores()` |
| Signed residual | $r_i=(O_i-E)/\sqrt{E}$ | `_chi_square_scores()` |
| Contribution | $c_i=(O_i-E)^2/E=r_i^2$ | `_chi_square_scores()` |
| Normalized score | $s_i=(r_i-r_{min})/(r_{max}-r_{min})$ | `_scale_scores()` |
| Ranking | score desc, gap desc, number asc | `_ranking_from_scores()` |
| Selection | first six ranked numbers | `_strategy()` |
| Historical update | increment `appearances[number]` | `remember()` |
| Walk-forward orchestration | remember reference, predict target | `build_prediction_suites()` |

The most important practical distinction is this:

> The chi-square **contribution** explains how unusual the magnitude of a
> historical deviation is, while the signed Pearson **residual** supplies the
> direction used for ranking. The normalized strategy score is a relative
> ranking value, not a probability.
