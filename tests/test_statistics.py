"""Test descriptive and diagnostic analysis of Draws datasets."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from rand_ai import Ball, Draw, Draws, DrawsStatistics
from rand_ai.statistics import CorrelationMethod


def _sample_draws() -> Draws:
    """Return three deterministic draws with known numbers and spaces."""
    draws = Draws()
    draws.add(Draw(1, 2, 3, 4, 5, 6))
    draws.add(Draw(1, 10, 20, 30, 40, 49))
    draws.add(Draw(5, 12, 19, 27, 36, 45))
    return draws


class TestDrawsStatisticsInitialization:
    """Test snapshot construction, validation, and scale boundaries."""

    def test_snapshots_compact_values_independently_of_source(self) -> None:
        """Verify compact shapes and independence from later collection changes."""
        draws = _sample_draws()
        statistics = DrawsStatistics(draws, heavy_sample_size=2, trend_bins=2)

        draws.add(Draw(7, 8, 9, 10, 11, 12))

        assert statistics.draw_count == 3
        assert statistics.sample_size == 2
        assert statistics.snapshot_shapes == ((3, 6), (3, 6))

    def test_sampling_is_deterministic(self) -> None:
        """Verify sampled raw spaces are reproducible for equal datasets."""
        first = DrawsStatistics(_sample_draws(), heavy_sample_size=2)
        second = DrawsStatistics(_sample_draws(), heavy_sample_size=2)

        pd.testing.assert_frame_equal(first.sampled_spaces(), second.sampled_spaces())

    def test_uses_all_rows_when_sample_limit_exceeds_dataset(self) -> None:
        """Verify small datasets use every row for heavy calculations."""
        statistics = DrawsStatistics(_sample_draws(), heavy_sample_size=100)

        assert statistics.sample_size == 3
        assert len(statistics.sampled_spaces()) == 18

    def test_rejects_value_that_is_not_draws(self) -> None:
        """Verify the analyzer requires a Draws instance."""
        with pytest.raises(TypeError, match="draws must be a Draws instance"):
            DrawsStatistics(cast(Draws, object()))

    def test_rejects_empty_dataset(self) -> None:
        """Verify an empty collection has no analyzable statistics."""
        with pytest.raises(ValueError, match="At least one draw is required"):
            DrawsStatistics(Draws())

    def test_rejects_dataset_above_one_million_draws(self) -> None:
        """Verify the documented one-million-draw ceiling."""
        draws = Draws()
        setattr(draws, "_draws", [Draw()] * (DrawsStatistics.MAX_DRAWS + 1))

        with pytest.raises(ValueError, match="At most 1000000 draws are supported"):
            DrawsStatistics(draws)

    @pytest.mark.parametrize("invalid_value", (0, -1, True, 1.5))
    def test_rejects_invalid_heavy_sample_size(self, invalid_value: object) -> None:
        """Verify heavy sample size must be a positive integer."""
        with pytest.raises(
            ValueError, match="heavy_sample_size must be a positive integer"
        ):
            DrawsStatistics(_sample_draws(), heavy_sample_size=cast(int, invalid_value))

    @pytest.mark.parametrize("invalid_value", (0, 501, True, 1.5))
    def test_rejects_invalid_default_trend_bins(self, invalid_value: object) -> None:
        """Verify default trend bins stay within the supported range."""
        with pytest.raises(ValueError, match="trend_bins must be between 1 and 500"):
            DrawsStatistics(_sample_draws(), trend_bins=cast(int, invalid_value))

    def test_rejects_non_draw_item_in_corrupted_collection(self) -> None:
        """Verify corrupted collections cannot inject arbitrary values."""
        draws = Draws()
        setattr(draws, "_draws", [object()])

        with pytest.raises(TypeError, match="contains a non-Draw value"):
            DrawsStatistics(draws)

    def test_rejects_invalid_mutated_draw(self) -> None:
        """Verify snapshot validation catches a corrupted slotted Draw."""
        draw = Draw()
        setattr(draw, "_num6", Ball(5))
        draws = Draws()
        draws.add(draw)

        with pytest.raises(ValueError, match="contains an invalid Draw"):
            DrawsStatistics(draws)

    def test_rejects_spaces_that_do_not_sum_to_43(self) -> None:
        """Verify the circular-space invariant is checked defensively."""
        invalid_numbers = np.array([[1, 1, 2, 3, 4, 5]], dtype=np.uint8)

        with pytest.raises(ValueError, match="spaces that sum to 43"):
            DrawsStatistics._calculate_spaces(invalid_numbers)

    def test_supports_one_million_draw_snapshot(self) -> None:
        """Verify compact array construction at the supported upper boundary."""
        draws = Draws()
        setattr(draws, "_draws", [Draw()] * DrawsStatistics.MAX_DRAWS)

        statistics = DrawsStatistics(draws, heavy_sample_size=1, trend_bins=1)

        assert statistics.snapshot_shapes == (
            (DrawsStatistics.MAX_DRAWS, 6),
            (DrawsStatistics.MAX_DRAWS, 6),
        )


class TestDrawsStatisticsTables:
    """Test exact number, space, structure, and relationship tables."""

    def test_summary_counts_unique_and_repeated_combinations(self) -> None:
        """Verify overview counts and draw-sum statistics."""
        draws = _sample_draws()
        draws.add(Draw(1, 2, 3, 4, 5, 6))
        summary = (
            DrawsStatistics(draws).summary().set_index("metric")["value"].to_dict()
        )

        assert summary["draw_count"] == 4
        assert summary["number_observations"] == 24
        assert summary["unique_combinations"] == 3
        assert summary["repeated_draws"] == 1
        assert summary["repeated_combinations"] == 1
        assert summary["draw_sum_min"] == 21
        assert summary["draw_sum_max"] == 150

    def test_number_frequency_values(self) -> None:
        """Verify exact frequency and expectation values."""
        frequencies = DrawsStatistics(_sample_draws()).number_frequencies()
        number_one = frequencies.set_index("number").loc[1]

        assert number_one["count"] == 2
        assert number_one["appearance_rate"] == pytest.approx(200 / 3)
        assert number_one["expected_count"] == pytest.approx(18 / 49)
        assert frequencies["count"].sum() == 18

    def test_position_frequency_values(self) -> None:
        """Verify position tables contain a complete six-by-49 grid."""
        frequencies = DrawsStatistics(_sample_draws()).position_frequencies()
        first_position = frequencies[frequencies["position"] == "num1"].set_index(
            "number"
        )

        assert len(frequencies) == 6 * 49
        assert first_position.loc[1, "count"] == 2
        assert first_position["count"].sum() == 3

    def test_number_descriptive_values(self) -> None:
        """Verify positional and draw-sum descriptive statistics."""
        descriptive = (
            DrawsStatistics(_sample_draws()).number_descriptive().set_index("variable")
        )

        assert descriptive.loc["num1", "mean"] == pytest.approx(7 / 3)
        assert descriptive.loc["draw_sum", "min"] == 21
        assert descriptive.loc["draw_sum", "max"] == 150

    def test_freshness_gap_distribution_covers_every_exact_gap(self) -> None:
        """Verify hits, candidate opportunities, and rates by pre-draw gap."""
        distribution = DrawsStatistics(
            _sample_draws()
        ).freshness_gap_distribution()

        assert distribution["gap"].tolist() == [0, 1, 2]
        assert distribution["hits"].tolist() == [7, 6, 5]
        assert distribution["opportunities"].tolist() == [61, 48, 38]
        assert distribution["hits"].sum() == 18
        assert distribution["opportunities"].sum() == 3 * 49
        assert distribution.loc[0, "hit_rate"] == pytest.approx(700 / 61)
        assert distribution.loc[2, "hit_percentage"] == pytest.approx(250 / 9)

    def test_freshness_gaps_support_numbers_never_drawn_again(self) -> None:
        """Verify trailing and never-hit candidate gaps remain opportunities."""
        draws = Draws()
        draws.add(Draw())
        draws.add(Draw())
        distribution = DrawsStatistics(draws).freshness_gap_distribution()

        assert distribution["gap"].tolist() == [0, 1]
        assert distribution["hits"].tolist() == [12, 0]
        assert distribution["opportunities"].tolist() == [55, 43]

    def test_draw_structure_distributions(self) -> None:
        """Verify sums, parity, ranges, and consecutive-pair counts."""
        distributions = DrawsStatistics(_sample_draws()).draw_structure_distributions()

        assert set(distributions["measure"]) == {
            "draw_sum",
            "odd_count",
            "low_count",
            "consecutive_pair_count",
        }
        consecutive = distributions[
            distributions["measure"] == "consecutive_pair_count"
        ].set_index("value")
        assert consecutive.loc[0, "count"] == 2
        assert consecutive.loc[5, "count"] == 1

    def test_pair_cooccurrence_is_symmetric_and_exact(self) -> None:
        """Verify exact pair counts, symmetry, expectations, and diagonal values."""
        pairs = DrawsStatistics(_sample_draws()).pair_cooccurrence()
        indexed = pairs.set_index(["number_a", "number_b"])

        assert len(pairs) == 49 * 49
        assert indexed.loc[(1, 2), "count"] == 1
        assert indexed.loc[(2, 1), "count"] == 1
        assert indexed.loc[(1, 1), "count"] == 0
        assert np.isnan(indexed.loc[(1, 1), "lift"])
        assert indexed.loc[(1, 2), "expected_count"] == pytest.approx(
            3 * 30 / (49 * 48)
        )

    def test_space_frequency_and_descriptive_values(self) -> None:
        """Verify exact space grids, summaries, extrema, and invariant totals."""
        statistics = DrawsStatistics(_sample_draws())
        frequencies = statistics.space_frequencies()
        distance_frequencies = statistics.distance_frequencies().set_index("distance")
        descriptive = statistics.space_descriptive().set_index("variable")
        extremes = statistics.space_extreme_distributions()

        assert len(frequencies) == 6 * 44
        assert frequencies["count"].sum() == 18
        assert len(distance_frequencies) == 44
        assert distance_frequencies["occurrences"].sum() == 18
        assert distance_frequencies.loc[0, "occurrences"] == 6
        assert distance_frequencies.loc[8, "occurrences"] == 5
        assert distance_frequencies.loc[9, "occurrences"] == 3
        assert distance_frequencies["occurrence_percentage"].sum() == pytest.approx(100)
        assert descriptive.loc["space_sum", "mean"] == 43
        assert descriptive.loc["space_sum", "std"] == 0
        assert set(extremes["measure"]) == {"minimum_space", "maximum_space"}
        assert extremes.groupby("measure")["count"].sum().eq(3).all()

    @pytest.mark.parametrize("method", ("pearson", "spearman"))
    def test_correlation_tables(self, method: CorrelationMethod) -> None:
        """Verify all three typed correlation matrices."""
        correlations = DrawsStatistics(_sample_draws()).correlations(method)

        assert correlations["numbers"].shape == (6, 6)
        assert correlations["spaces"].shape == (6, 6)
        assert correlations["number_space"].shape == (6, 6)
        assert list(correlations["numbers"].columns) == [
            "num1",
            "num2",
            "num3",
            "num4",
            "num5",
            "num6",
        ]

    def test_rejects_unknown_correlation_method(self) -> None:
        """Verify only Pearson and Spearman correlations are supported."""
        with pytest.raises(ValueError, match="method must be"):
            DrawsStatistics(_sample_draws()).correlations(
                cast(CorrelationMethod, "kendall")
            )

    def test_trend_uses_exact_binned_counts(self) -> None:
        """Verify selected-number trends and draw-index boundaries."""
        statistics = DrawsStatistics(_sample_draws(), trend_bins=2)
        trend = statistics.trend([1], bins=2)

        assert trend.to_dict("records") == [
            {
                "bin": 1,
                "start_draw": 1,
                "end_draw": 1,
                "number": 1,
                "count": 1,
                "appearance_rate": 100.0,
            },
            {
                "bin": 2,
                "start_draw": 2,
                "end_draw": 3,
                "number": 1,
                "count": 1,
                "appearance_rate": 50.0,
            },
        ]

    def test_trend_defaults_to_every_number_and_configured_bins(self) -> None:
        """Verify default trend selection and bin configuration."""
        trend = DrawsStatistics(_sample_draws(), trend_bins=2).trend()

        assert len(trend) == 2 * 49
        assert set(trend["number"]) == set(range(1, 50))

    @pytest.mark.parametrize("invalid_numbers", ([], [0], [50], [True]))
    def test_trend_rejects_invalid_number_selection(
        self, invalid_numbers: list[object]
    ) -> None:
        """Verify trend selections are nonempty integers from 1 through 49."""
        with pytest.raises(ValueError):
            DrawsStatistics(_sample_draws()).trend(cast(list[int], invalid_numbers))

    @pytest.mark.parametrize("invalid_bins", (0, 501, True, 1.5))
    def test_trend_rejects_invalid_bin_count(self, invalid_bins: object) -> None:
        """Verify trend bin overrides remain within one through 500."""
        with pytest.raises(ValueError, match="bins must be between 1 and 500"):
            DrawsStatistics(_sample_draws()).trend([1], bins=cast(int, invalid_bins))


class TestDrawsRandomnessAndExports:
    """Test descriptive randomness diagnostics and complete table exports."""

    def test_randomness_diagnostics_include_expected_measures(self) -> None:
        """Verify diagnostics, finite entropy, serial result, and reliability."""
        diagnostics = (
            DrawsStatistics(_sample_draws())
            .randomness_diagnostics()
            .set_index("diagnostic")
        )

        assert set(diagnostics.index) == {
            "number_frequency_chi_square",
            "number_frequency_p_value",
            "normalized_frequency_entropy",
            "draw_sum_lag_one_autocorrelation",
            "observed_matching_combination_pairs",
            "expected_matching_combination_pairs",
        }
        assert 0 <= diagnostics.loc["normalized_frequency_entropy", "value"] <= 1
        assert np.isfinite(diagnostics.loc["draw_sum_lag_one_autocorrelation", "value"])
        assert not bool(diagnostics.loc["number_frequency_p_value", "reliable"])

    def test_single_draw_has_undefined_lag_one_diagnostic(self) -> None:
        """Verify lag-one correlation is undefined with fewer than two draws."""
        draws = Draws()
        draws.add(Draw())
        diagnostics = (
            DrawsStatistics(draws).randomness_diagnostics().set_index("diagnostic")
        )

        assert np.isnan(diagnostics.loc["draw_sum_lag_one_autocorrelation", "value"])

    def test_constant_draw_sums_have_undefined_lag_one_diagnostic(self) -> None:
        """Verify lag-one correlation is undefined for a constant sum series."""
        draws = Draws()
        for _ in range(41):
            draws.add(Draw())
        diagnostics = (
            DrawsStatistics(draws).randomness_diagnostics().set_index("diagnostic")
        )

        assert np.isnan(diagnostics.loc["draw_sum_lag_one_autocorrelation", "value"])
        assert bool(diagnostics.loc["number_frequency_p_value", "reliable"])
        assert diagnostics.loc[
            "observed_matching_combination_pairs", "value"
        ] == pytest.approx(820)

    def test_export_tables_contains_every_compact_result(self) -> None:
        """Verify stable export names and DataFrame result types."""
        tables = DrawsStatistics(_sample_draws(), trend_bins=2).export_tables()

        assert set(tables) == {
            "summary",
            "number_frequencies",
            "position_frequencies",
            "number_descriptive",
            "freshness_gap_distribution",
            "draw_structure_distributions",
            "pair_cooccurrence",
            "space_frequencies",
            "distance_frequencies",
            "space_descriptive",
            "space_extreme_distributions",
            "number_correlations_pearson",
            "space_correlations_pearson",
            "number_space_correlations_pearson",
            "number_correlations_spearman",
            "space_correlations_spearman",
            "number_space_correlations_spearman",
            "number_trends",
            "randomness_diagnostics",
        }
        assert all(isinstance(table, pd.DataFrame) for table in tables.values())
