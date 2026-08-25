"""Compute descriptive and diagnostic statistics for Draws datasets."""

from collections.abc import Collection
from functools import cached_property
from dataclasses import asdict
from math import comb, log, sqrt
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import chisquare

from rand_ai.draw import Draw
from rand_ai.draws import Draws
from rand_ai.space_groups import (
    MAX_BORDER_SPACE,
    MODEL_IDS,
    MODEL_NAMES,
    SIGNATURES,
    exact_null_group_probabilities,
    exact_null_probabilities,
    profile_from_spaces,
    signature_chi_square,
    transition_diagnostics,
    validate_border_space,
    walk_forward_models,
)

type CorrelationMethod = Literal["pearson", "spearman"]
type UInt8Array = npt.NDArray[np.uint8]
type Int64Array = npt.NDArray[np.int64]

NUMBER_NAMES = tuple(f"num{position}" for position in range(1, 7))
SPACE_NAMES = tuple(f"dist{position}" for position in range(1, 7))


class DrawsStatistics:
    """Analyze an immutable snapshot of up to one million draws."""

    MAX_DRAWS = 1_000_000
    MAX_TREND_BINS = 500

    def __init__(
        self,
        draws: Draws,
        *,
        heavy_sample_size: int = 100_000,
        trend_bins: int = 500,
    ) -> None:
        """Validate configuration and snapshot number and space values."""
        if not isinstance(draws, Draws):
            raise TypeError("draws must be a Draws instance")
        draw_count = len(draws)
        if draw_count == 0:
            raise ValueError("At least one draw is required")
        if draw_count > self.MAX_DRAWS:
            raise ValueError(f"At most {self.MAX_DRAWS} draws are supported")
        if type(heavy_sample_size) is not int or heavy_sample_size < 1:
            raise ValueError("heavy_sample_size must be a positive integer")
        if type(trend_bins) is not int or not 1 <= trend_bins <= self.MAX_TREND_BINS:
            raise ValueError(f"trend_bins must be between 1 and {self.MAX_TREND_BINS}")

        self._draw_count = draw_count
        self._heavy_sample_size = min(heavy_sample_size, draw_count)
        self._trend_bins = min(trend_bins, draw_count)
        self._numbers = self._snapshot_numbers(draws)
        self._dates = tuple(draw.date for draw in draws)
        self._spaces = self._calculate_spaces(self._numbers)
        self._sample_indices = self._select_sample_indices()

    def _snapshot_numbers(self, draws: Draws) -> UInt8Array:
        """Copy validated Draw values into a compact read-only array."""
        numbers = np.empty((self._draw_count, 6), dtype=np.uint8)
        for index, draw in enumerate(draws):
            if not isinstance(draw, Draw):
                raise TypeError("Draws collection contains a non-Draw value")
            values = tuple(ball.value for ball in draw.balls)
            if (
                values != tuple(sorted(values))
                or len(set(values)) != 6
                or values[0] < 1
                or values[-1] > 49
            ):
                raise ValueError("Draws collection contains an invalid Draw")
            numbers[index] = values
        numbers.setflags(write=False)
        return numbers

    @staticmethod
    def _calculate_spaces(numbers: UInt8Array) -> UInt8Array:
        """Calculate the six circular gap values for every draw."""
        spaces = np.empty_like(numbers)
        spaces[:, 0] = (numbers[:, 0] - 1) + (49 - numbers[:, 5])
        spaces[:, 1:] = numbers[:, 1:] - numbers[:, :-1] - 1
        if not np.all(spaces.sum(axis=1) == 43):
            raise ValueError("Every draw must have spaces that sum to 43")
        spaces.setflags(write=False)
        return spaces

    def _select_sample_indices(self) -> Int64Array:
        """Choose a deterministic sample for expensive calculations."""
        if self._heavy_sample_size == self._draw_count:
            return np.arange(self._draw_count, dtype=np.int64)
        generator = np.random.default_rng(0)
        indices = generator.choice(
            self._draw_count, size=self._heavy_sample_size, replace=False
        )
        return np.sort(indices.astype(np.int64))

    @staticmethod
    def _describe(
        values: npt.NDArray[np.generic], names: tuple[str, ...]
    ) -> pd.DataFrame:
        """Return population descriptive statistics for named columns."""
        numeric_values = values.astype(np.float64)
        quantiles = np.quantile(numeric_values, (0.25, 0.5, 0.75), axis=0)
        return pd.DataFrame(
            {
                "variable": names,
                "count": values.shape[0],
                "mean": numeric_values.mean(axis=0),
                "std": numeric_values.std(axis=0, ddof=0),
                "min": numeric_values.min(axis=0),
                "q25": quantiles[0],
                "median": quantiles[1],
                "q75": quantiles[2],
                "max": numeric_values.max(axis=0),
            }
        )

    @staticmethod
    def _distribution(measure: str, values: npt.NDArray[np.integer]) -> pd.DataFrame:
        """Return count and percentage rows for one discrete measure."""
        unique_values, counts = np.unique(values, return_counts=True)
        return pd.DataFrame(
            {
                "measure": measure,
                "value": unique_values.astype(np.int64),
                "count": counts.astype(np.int64),
                "percentage": counts / values.size * 100,
            }
        )

    @cached_property
    def _number_counts(self) -> Int64Array:
        """Return exact aggregate counts for values 1 through 49."""
        return np.bincount(self._numbers.ravel(), minlength=50)[1:].astype(np.int64)

    @cached_property
    def _combination_counts(self) -> Int64Array:
        """Return occurrence counts for each unique six-number combination."""
        codes = np.zeros(self._draw_count, dtype=np.int64)
        for position in range(6):
            codes = codes * 50 + self._numbers[:, position].astype(np.int64)
        return np.unique(codes, return_counts=True)[1].astype(np.int64)

    @property
    def draw_count(self) -> int:
        """Return the number of draws in the analyzed snapshot."""
        return self._draw_count

    @property
    def sample_size(self) -> int:
        """Return the deterministic sample size used by heavy analyses."""
        return self._heavy_sample_size

    @property
    def snapshot_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return number and space snapshot shapes for diagnostics."""
        return self._numbers.shape, self._spaces.shape

    def summary(self) -> pd.DataFrame:
        """Return high-level dataset and draw-sum statistics."""
        draw_sums = self._numbers.sum(axis=1, dtype=np.int64)
        combination_counts = self._combination_counts
        repeated_draws = int(np.maximum(combination_counts - 1, 0).sum())
        values: list[int | float] = [
            self._draw_count,
            self._draw_count * 6,
            int(combination_counts.size),
            repeated_draws,
            int(np.count_nonzero(combination_counts > 1)),
            float(draw_sums.mean()),
            float(draw_sums.std(ddof=0)),
            int(draw_sums.min()),
            float(np.quantile(draw_sums, 0.25)),
            float(np.median(draw_sums)),
            float(np.quantile(draw_sums, 0.75)),
            int(draw_sums.max()),
        ]
        metrics = (
            "draw_count",
            "number_observations",
            "unique_combinations",
            "repeated_draws",
            "repeated_combinations",
            "draw_sum_mean",
            "draw_sum_std",
            "draw_sum_min",
            "draw_sum_q25",
            "draw_sum_median",
            "draw_sum_q75",
            "draw_sum_max",
        )
        return pd.DataFrame({"metric": metrics, "value": values})

    def number_frequencies(self) -> pd.DataFrame:
        """Return exact frequency statistics for values 1 through 49."""
        expected_count = self._draw_count * 6 / 49
        inclusion_probability = 6 / 49
        standard_deviation = (
            self._draw_count * inclusion_probability * (1 - inclusion_probability)
        ) ** 0.5
        deviations = self._number_counts - expected_count
        return pd.DataFrame(
            {
                "number": np.arange(1, 50, dtype=np.int64),
                "count": self._number_counts,
                "appearance_rate": self._number_counts / self._draw_count * 100,
                "observation_percentage": self._number_counts
                / (self._draw_count * 6)
                * 100,
                "expected_count": expected_count,
                "deviation": deviations,
                "standardized_residual": deviations / standard_deviation,
            }
        )

    def position_frequencies(self) -> pd.DataFrame:
        """Return exact number frequencies for each sorted draw position."""
        counts = np.stack(
            [
                np.bincount(self._numbers[:, position], minlength=50)[1:]
                for position in range(6)
            ]
        )
        return pd.DataFrame(
            {
                "position": np.repeat(NUMBER_NAMES, 49),
                "number": np.tile(np.arange(1, 50, dtype=np.int64), 6),
                "count": counts.ravel().astype(np.int64),
                "appearance_rate": counts.ravel() / self._draw_count * 100,
            }
        )

    def number_descriptive(self) -> pd.DataFrame:
        """Return descriptive statistics for positions and total draw sum."""
        draw_sums = self._numbers.sum(axis=1, dtype=np.int64).reshape(-1, 1)
        values = np.hstack((self._numbers, draw_sums))
        return self._describe(values, (*NUMBER_NAMES, "draw_sum"))

    def freshness_gap_distribution(self) -> pd.DataFrame:
        """Return whole-history hits and opportunities for every exact gap."""
        hit_counts = np.zeros(self._draw_count, dtype=np.int64)
        opportunity_delta = np.zeros(self._draw_count + 1, dtype=np.int64)

        for number in range(1, 50):
            occurrences = np.flatnonzero(np.any(self._numbers == number, axis=1))
            if occurrences.size:
                hit_gaps = np.concatenate(
                    (
                        occurrences[:1],
                        np.diff(occurrences) - 1,
                    )
                ).astype(np.int64)
                np.add.at(hit_counts, hit_gaps, 1)
                interval_maxima = hit_gaps
                trailing_gap = self._draw_count - int(occurrences[-1]) - 2
                if trailing_gap >= 0:
                    interval_maxima = np.append(interval_maxima, trailing_gap)
            else:
                interval_maxima = np.array(
                    [self._draw_count - 1],
                    dtype=np.int64,
                )

            opportunity_delta[0] += interval_maxima.size
            np.add.at(opportunity_delta, interval_maxima + 1, -1)

        opportunity_counts = np.cumsum(opportunity_delta[:-1])
        encountered = np.flatnonzero(opportunity_counts)
        maximum_gap = int(encountered[-1])
        gaps = np.arange(maximum_gap + 1, dtype=np.int64)
        hits = hit_counts[: maximum_gap + 1]
        opportunities = opportunity_counts[: maximum_gap + 1]
        hit_rate = np.divide(
            hits,
            opportunities,
            out=np.zeros(maximum_gap + 1, dtype=np.float64),
            where=opportunities != 0,
        )
        return pd.DataFrame(
            {
                "gap": gaps,
                "hits": hits,
                "opportunities": opportunities,
                "hit_rate": hit_rate * 100,
                "hit_percentage": hits / (self._draw_count * 6) * 100,
            }
        )

    def draw_structure_distributions(self) -> pd.DataFrame:
        """Return sum, parity, range, and consecutive-pair distributions."""
        draw_sums = self._numbers.sum(axis=1, dtype=np.int64)
        odd_counts = (self._numbers % 2 == 1).sum(axis=1, dtype=np.int64)
        low_counts = (self._numbers <= 24).sum(axis=1, dtype=np.int64)
        consecutive_pairs = (np.diff(self._numbers, axis=1) == 1).sum(
            axis=1, dtype=np.int64
        )
        return pd.concat(
            (
                self._distribution("draw_sum", draw_sums),
                self._distribution("odd_count", odd_counts),
                self._distribution("low_count", low_counts),
                self._distribution("consecutive_pair_count", consecutive_pairs),
            ),
            ignore_index=True,
        )

    def pair_cooccurrence(self) -> pd.DataFrame:
        """Return the exact 49-by-49 number co-occurrence table."""
        matrix = np.zeros((49, 49), dtype=np.int64)
        for first_position in range(5):
            for second_position in range(first_position + 1, 6):
                first = self._numbers[:, first_position].astype(np.int64) - 1
                second = self._numbers[:, second_position].astype(np.int64) - 1
                np.add.at(matrix, (first, second), 1)
                np.add.at(matrix, (second, first), 1)

        number_a = np.repeat(np.arange(1, 50, dtype=np.int64), 49)
        number_b = np.tile(np.arange(1, 50, dtype=np.int64), 49)
        diagonal = number_a == number_b
        expected = self._draw_count * 30 / (49 * 48)
        expected_counts = np.where(diagonal, 0.0, expected)
        lift = np.divide(
            matrix.ravel(),
            expected_counts,
            out=np.full(49 * 49, np.nan),
            where=expected_counts != 0,
        )
        return pd.DataFrame(
            {
                "number_a": number_a,
                "number_b": number_b,
                "count": matrix.ravel(),
                "expected_count": expected_counts,
                "lift": lift,
            }
        )

    def space_frequencies(self) -> pd.DataFrame:
        """Return exact frequency tables for all six space positions."""
        counts = np.stack(
            [
                np.bincount(self._spaces[:, position], minlength=44)
                for position in range(6)
            ]
        )
        return pd.DataFrame(
            {
                "position": np.repeat(SPACE_NAMES, 44),
                "space": np.tile(np.arange(44, dtype=np.int64), 6),
                "count": counts.ravel().astype(np.int64),
                "percentage": counts.ravel() / self._draw_count * 100,
            }
        )

    def distance_frequencies(self) -> pd.DataFrame:
        """Return exact aggregate frequencies for distance values 0 through 43."""
        counts = np.bincount(self._spaces.ravel(), minlength=44).astype(np.int64)
        occurrence_count = self._draw_count * 6
        return pd.DataFrame(
            {
                "distance": np.arange(44, dtype=np.int64),
                "occurrences": counts,
                "occurrence_percentage": counts / occurrence_count * 100,
            }
        )

    def space_descriptive(self) -> pd.DataFrame:
        """Return descriptive statistics for spaces and their extrema."""
        minimum = self._spaces.min(axis=1).reshape(-1, 1)
        maximum = self._spaces.max(axis=1).reshape(-1, 1)
        totals = self._spaces.sum(axis=1, dtype=np.int64).reshape(-1, 1)
        values = np.hstack((self._spaces, minimum, maximum, totals))
        return self._describe(
            values, (*SPACE_NAMES, "minimum_space", "maximum_space", "space_sum")
        )

    def space_extreme_distributions(self) -> pd.DataFrame:
        """Return exact minimum- and maximum-space distributions."""
        minimum = self._spaces.min(axis=1)
        maximum = self._spaces.max(axis=1)
        return pd.concat(
            (
                self._distribution("minimum_space", minimum),
                self._distribution("maximum_space", maximum),
            ),
            ignore_index=True,
        )

    def sampled_spaces(self) -> pd.DataFrame:
        """Return deterministic sampled raw spaces for Plotly box plots."""
        sample = self._spaces[self._sample_indices]
        return pd.DataFrame(
            {
                "position": np.repeat(SPACE_NAMES, sample.shape[0]),
                "space": sample.T.ravel().astype(np.int64),
            }
        )

    def space_group_analysis(
        self,
        border_space: int,
        target_group_count: int | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
        """Return historical, null, transition, and forecast group analysis."""
        border = validate_border_space(border_space)
        profiles = [
            profile_from_spaces(
                tuple(int(value) for value in spaces),
                border,
                numbers=tuple(int(value) for value in numbers),
            )
            for numbers, spaces in zip(self._numbers, self._spaces, strict=True)
        ]
        null_signatures = exact_null_probabilities(border)
        null_groups = exact_null_group_probabilities(border)
        signature_counts = {
            signature: sum(profile.signature == signature for profile in profiles)
            for signature in SIGNATURES
        }
        group_counts = np.asarray(
            [profile.group_count for profile in profiles], dtype=np.int64
        )
        group_size_counts = np.bincount(
            np.asarray(
                [size for profile in profiles for size in profile.signature],
                dtype=np.int64,
            ),
            minlength=7,
        )
        signature_rows = []
        for signature, probability in zip(
            SIGNATURES, null_signatures, strict=True
        ):
            count = signature_counts[signature]
            expected = self._draw_count * probability
            signature_rows.append(
                {
                    "signature": "+".join(str(value) for value in signature),
                    "group_count": len(signature),
                    "count": count,
                    "percentage": count / self._draw_count * 100,
                    "null_probability": probability * 100,
                    "expected_count": expected,
                    "standardized_residual": (
                        (count - expected) / sqrt(expected) if expected else 0.0
                    ),
                }
            )

        matrix, mutual_information, permutation_p = transition_diagnostics(profiles)
        signature_totals = [sum(row) for row in matrix]
        target_totals = [sum(matrix[row][column] for row in range(len(SIGNATURES))) for column in range(len(SIGNATURES))]
        transition_total = max(sum(signature_totals), 1)
        transition_rows = []
        for left_index, left in enumerate(SIGNATURES):
            for right_index, right in enumerate(SIGNATURES):
                count = matrix[left_index][right_index]
                expected = signature_totals[left_index] * target_totals[right_index] / transition_total
                transition_rows.append(
                    {
                        "from_signature": "+".join(map(str, left)),
                        "to_signature": "+".join(map(str, right)),
                        "count": count,
                        "row_percentage": (
                            count / signature_totals[left_index] * 100
                            if signature_totals[left_index]
                            else 0.0
                        ),
                        "expected_count": expected,
                        "lift": count / expected if expected else 0.0,
                    }
                )

        rolling_rows = []
        for index, profile in enumerate(profiles):
            row: dict[str, int | float | str | None] = {
                "draw_number": index + 1,
                "date": self._dates[index],
                "numbers": ",".join(str(int(value)) for value in self._numbers[index]),
                "spaces": ",".join(str(value) for value in profile.spaces),
                "large_spaces": ",".join(str(value) for value in profile.large_spaces),
                "separator_indices": ",".join(
                    str(value) for value in profile.separator_indices
                ),
                "separator_count": profile.separator_count,
                "group_count": profile.group_count,
                "ordered_groups": " | ".join(
                    ",".join(str(number) for number in group)
                    for group in profile.ordered_groups
                ),
                "ordered_group_sizes": "+".join(map(str, profile.ordered_group_sizes)),
                "signature": profile.signature_text,
                "maximum_space": profile.maximum_space,
            }
            for window in (25, 100):
                start = max(0, index + 1 - window)
                row[f"rolling_{window}_group_count"] = float(
                    group_counts[start : index + 1].mean()
                )
            rolling_rows.append(row)

        sensitivity_rows = []
        for candidate_border in range(MAX_BORDER_SPACE + 1):
            separators = (self._spaces > candidate_border).sum(
                axis=1, dtype=np.int64
            )
            observed_groups = np.maximum(separators, 1)
            null_distribution = exact_null_group_probabilities(candidate_border)
            sensitivity_rows.append(
                {
                    "border_space": candidate_border,
                    "average_group_count": float(observed_groups.mean()),
                    "one_to_three_percentage": float(
                        np.mean(observed_groups <= 3) * 100
                    ),
                    "null_average_group_count": sum(
                        (index + 1) * probability
                        for index, probability in enumerate(null_distribution)
                    ),
                    "null_one_to_three_percentage": sum(
                        null_distribution[:3]
                    )
                    * 100,
                }
            )

        model_analysis = walk_forward_models(
            profiles, border, target_group_count
        )
        metrics = model_analysis["metrics"]
        latest = model_analysis["latest"]
        forecasts = []
        for model_id in MODEL_IDS:
            probabilities = latest[model_id]
            top_index = max(
                range(len(SIGNATURES)), key=probabilities.__getitem__
            )
            forecasts.append(
                {
                    "modelId": model_id,
                    "name": MODEL_NAMES[model_id],
                    "topSignature": "+".join(map(str, SIGNATURES[top_index])),
                    "topGroupCount": len(SIGNATURES[top_index]),
                    "topProbability": probabilities[top_index],
                    "probabilities": [
                        {
                            "signature": "+".join(map(str, signature)),
                            "groupCount": len(signature),
                            "probability": probability,
                        }
                        for signature, probability in zip(
                            SIGNATURES, probabilities, strict=True
                        )
                    ],
                }
            )
        chi_square_statistic, chi_square_p = signature_chi_square(
            profiles, border
        )
        tables = {
            "space_group_history": pd.DataFrame(rolling_rows),
            "space_group_count_distribution": pd.DataFrame(
                {
                    "group_count": np.arange(1, 7, dtype=np.int64),
                    "count": np.bincount(group_counts, minlength=7)[1:],
                    "percentage": np.bincount(group_counts, minlength=7)[1:]
                    / self._draw_count
                    * 100,
                    "null_probability": np.asarray(null_groups) * 100,
                    "expected_count": np.asarray(null_groups) * self._draw_count,
                }
            ),
            "space_group_size_distribution": pd.DataFrame(
                {
                    "group_size": np.arange(1, 7, dtype=np.int64),
                    "count": group_size_counts[1:],
                    "percentage": group_size_counts[1:]
                    / max(group_size_counts[1:].sum(), 1)
                    * 100,
                }
            ),
            "space_group_signature_distribution": pd.DataFrame(signature_rows),
            "space_group_transitions": pd.DataFrame(transition_rows),
            "space_group_threshold_sensitivity": pd.DataFrame(sensitivity_rows),
            "space_group_model_metrics": pd.DataFrame(
                [asdict(metric) for metric in metrics]
            ),
        }
        payload: dict[str, object] = {
            "borderSpace": border,
            "targetGroupCount": target_group_count,
            "smallSpaceDefinition": f"0–{border}",
            "largeSpaceDefinition": f">{border}",
            "bestModelId": model_analysis["best_model_id"],
            "provisional": model_analysis["provisional"],
            "forecasts": forecasts,
            "hybridWeights": model_analysis["hybrid_weights"],
            "signatureChiSquare": chi_square_statistic,
            "signatureChiSquarePValue": chi_square_p,
            "transitionMutualInformation": mutual_information,
            "transitionPermutationPValue": permutation_p,
        }
        return tables, payload

    def correlations(
        self, method: CorrelationMethod = "pearson"
    ) -> dict[str, pd.DataFrame]:
        """Return number, space, and cross-correlation matrices."""
        if method not in ("pearson", "spearman"):
            raise ValueError("method must be 'pearson' or 'spearman'")
        indices = (
            np.arange(self._draw_count, dtype=np.int64)
            if method == "pearson"
            else self._sample_indices
        )
        numbers = pd.DataFrame(self._numbers[indices], columns=NUMBER_NAMES)
        spaces = pd.DataFrame(self._spaces[indices], columns=SPACE_NAMES)
        combined = pd.concat((numbers, spaces), axis=1)
        combined_correlations = combined.corr(method=method)
        number_names = list(NUMBER_NAMES)
        space_names = list(SPACE_NAMES)
        return {
            "numbers": combined_correlations.loc[number_names, number_names].copy(),
            "spaces": combined_correlations.loc[space_names, space_names].copy(),
            "number_space": combined_correlations.loc[number_names, space_names].copy(),
        }

    def trend(
        self,
        selected_numbers: Collection[int] | None = None,
        *,
        bins: int | None = None,
    ) -> pd.DataFrame:
        """Return exact number appearance trends grouped by draw-index bins."""
        numbers = (
            tuple(range(1, 50))
            if selected_numbers is None
            else tuple(sorted(set(selected_numbers)))
        )
        if not numbers:
            raise ValueError("At least one number must be selected")
        if any(type(number) is not int or not 1 <= number <= 49 for number in numbers):
            raise ValueError("Selected numbers must be integers from 1 through 49")

        requested_bins = self._trend_bins if bins is None else bins
        if type(requested_bins) is not int or not 1 <= requested_bins <= 500:
            raise ValueError("bins must be between 1 and 500")
        bin_count = min(requested_bins, self._draw_count)
        edges = np.linspace(0, self._draw_count, bin_count + 1, dtype=np.int64)
        rows: list[dict[str, int | float]] = []
        for bin_index, (start, end) in enumerate(
            zip(edges[:-1], edges[1:], strict=True), start=1
        ):
            counts = np.bincount(self._numbers[start:end].ravel(), minlength=50)
            size = int(end - start)
            for number in numbers:
                count = int(counts[number])
                rows.append(
                    {
                        "bin": bin_index,
                        "start_draw": int(start + 1),
                        "end_draw": int(end),
                        "number": number,
                        "count": count,
                        "appearance_rate": count / size * 100,
                    }
                )
        return pd.DataFrame(rows)

    def randomness_diagnostics(self) -> pd.DataFrame:
        """Return descriptive diagnostics of frequency and serial randomness."""
        chi_square_result = chisquare(self._number_counts)
        probabilities = self._number_counts / self._number_counts.sum()
        nonzero_probabilities = probabilities[probabilities > 0]
        normalized_entropy = float(
            -(nonzero_probabilities * np.log(nonzero_probabilities)).sum() / log(49)
        )
        draw_sums = self._numbers.sum(axis=1, dtype=np.int64)
        lag_one = (
            float("nan")
            if self._draw_count < 2 or float(draw_sums.std()) == 0
            else float(np.corrcoef(draw_sums[:-1], draw_sums[1:])[0, 1])
        )
        combination_counts = self._combination_counts
        observed_matching_pairs = int(
            (combination_counts * (combination_counts - 1) // 2).sum()
        )
        expected_matching_pairs = (
            self._draw_count * (self._draw_count - 1) / (2 * comb(49, 6))
        )
        expected_number_count = self._draw_count * 6 / 49
        return pd.DataFrame(
            {
                "diagnostic": (
                    "number_frequency_chi_square",
                    "number_frequency_p_value",
                    "normalized_frequency_entropy",
                    "draw_sum_lag_one_autocorrelation",
                    "observed_matching_combination_pairs",
                    "expected_matching_combination_pairs",
                ),
                "value": (
                    float(chi_square_result.statistic),
                    float(chi_square_result.pvalue),
                    normalized_entropy,
                    lag_one,
                    observed_matching_pairs,
                    expected_matching_pairs,
                ),
                "reference": (
                    "0 is a perfect frequency match",
                    "Interpret only when expected count is at least 5",
                    "1 is maximum entropy",
                    "0 indicates no linear lag-one relationship",
                    "Observed equal-combination pairs",
                    "Expected under uniform independent draws",
                ),
                "reliable": (
                    expected_number_count >= 5,
                    expected_number_count >= 5,
                    True,
                    self._draw_count >= 3,
                    True,
                    True,
                ),
            }
        )

    def export_tables(self) -> dict[str, pd.DataFrame]:
        """Return all compact statistical tables with stable export names."""
        pearson = self.correlations("pearson")
        spearman = self.correlations("spearman")
        return {
            "summary": self.summary(),
            "number_frequencies": self.number_frequencies(),
            "position_frequencies": self.position_frequencies(),
            "number_descriptive": self.number_descriptive(),
            "freshness_gap_distribution": self.freshness_gap_distribution(),
            "draw_structure_distributions": self.draw_structure_distributions(),
            "pair_cooccurrence": self.pair_cooccurrence(),
            "space_frequencies": self.space_frequencies(),
            "distance_frequencies": self.distance_frequencies(),
            "space_descriptive": self.space_descriptive(),
            "space_extreme_distributions": self.space_extreme_distributions(),
            "number_correlations_pearson": pearson["numbers"],
            "space_correlations_pearson": pearson["spaces"],
            "number_space_correlations_pearson": pearson["number_space"],
            "number_correlations_spearman": spearman["numbers"],
            "space_correlations_spearman": spearman["spaces"],
            "number_space_correlations_spearman": spearman["number_space"],
            "number_trends": self.trend(),
            "randomness_diagnostics": self.randomness_diagnostics(),
        }
