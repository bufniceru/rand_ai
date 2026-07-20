"""Test Plotly figure construction and downloadable archive generation."""

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import plotly.graph_objects as go

from rand_ai import Draw, Draws, DrawsStatistics
from rand_ai.visualizations import build_export_zip, build_figures


def _statistics() -> DrawsStatistics:
    """Return deterministic statistics suitable for every chart."""
    draws = Draws()
    draws.add(Draw(1, 2, 3, 4, 5, 6))
    draws.add(Draw(1, 10, 20, 30, 40, 49))
    draws.add(Draw(5, 12, 19, 27, 36, 45))
    return DrawsStatistics(draws, heavy_sample_size=2, trend_bins=2)


class TestPlotlyFigures:
    """Test complete and bounded Plotly dashboard figures."""

    def test_builds_every_dashboard_figure(self) -> None:
        """Verify stable names, Plotly types, traces, and titles."""
        figures = build_figures(
            _statistics(),
            selected_numbers=(1, 5),
            trend_bins=2,
            correlation_method="pearson",
        )

        assert set(figures) == {
            "number_frequencies",
            "draw_sum_distribution",
            "draw_composition",
            "position_frequencies",
            "pair_cooccurrence",
            "number_trends",
            "distance_frequencies",
            "dist1_frequencies",
            "dist2_frequencies",
            "dist3_frequencies",
            "dist4_frequencies",
            "dist5_frequencies",
            "dist6_frequencies",
            "space_frequencies",
            "space_box_plots",
            "space_extremes",
            "matching_pairs",
            "number_correlations",
            "space_correlations",
            "number_space_correlations",
        }
        assert all(isinstance(figure, go.Figure) for figure in figures.values())
        assert all(len(figure.data) > 0 for figure in figures.values())
        assert all(figure.layout.title.text for figure in figures.values())

    def test_heatmaps_have_expected_compact_shapes(self) -> None:
        """Verify heatmaps contain aggregate matrices rather than raw rows."""
        figures = build_figures(_statistics(), trend_bins=2)

        assert figures["position_frequencies"].data[0].z.shape == (6, 49)
        assert figures["pair_cooccurrence"].data[0].z.shape == (49, 49)
        assert figures["space_frequencies"].data[0].z.shape == (6, 44)
        assert figures["number_correlations"].data[0].z.shape == (6, 6)
        assert figures["number_space_correlations"].data[0].z.shape == (6, 6)

    def test_distance_frequency_counts_repeated_occurrences(self) -> None:
        """Verify aggregate distance bars include all values and repetitions."""
        statistics = _statistics()
        figure = build_figures(statistics, trend_bins=2)["distance_frequencies"]

        assert list(figure.data[0].x) == list(range(44))
        assert sum(figure.data[0].y) == statistics.draw_count * 6
        assert figure.data[0].y[0] == 6
        assert figure.data[0].y[8] == 5
        assert figure.layout.xaxis.range == (-0.5, 43.5)

    def test_individual_distance_frequency_figures_are_isolated(self) -> None:
        """Verify each distance-position chart counts only its own occurrences."""
        statistics = _statistics()
        figures = build_figures(statistics, trend_bins=2)

        for position in range(1, 7):
            figure = figures[f"dist{position}_frequencies"]
            assert list(figure.data[0].x) == list(range(44))
            assert sum(figure.data[0].y) == statistics.draw_count
            assert figure.layout.xaxis.range == (-0.5, 43.5)

        assert figures["dist1_frequencies"].data[0].y[43] == 1
        assert figures["dist6_frequencies"].data[0].y[8] == 2

    def test_box_plot_and_trend_are_bounded(self) -> None:
        """Verify expensive figures contain only sampled or binned values."""
        statistics = _statistics()
        figures = build_figures(statistics, selected_numbers=(1, 2), trend_bins=2)
        box_value_count = sum(len(trace.y) for trace in figures["space_box_plots"].data)
        trend_point_count = sum(len(trace.y) for trace in figures["number_trends"].data)

        assert box_value_count <= statistics.sample_size * 6
        assert trend_point_count <= 2 * 2

    def test_spearman_titles_disclose_sampling(self) -> None:
        """Verify sampled correlation charts explicitly identify sampling."""
        figures = build_figures(
            _statistics(), trend_bins=2, correlation_method="spearman"
        )

        assert "sampled" in figures["number_correlations"].layout.title.text
        assert "sampled" in figures["space_correlations"].layout.title.text


class TestVisualizationExport:
    """Test CSV and self-contained Plotly HTML ZIP exports."""

    def test_builds_zip_with_range_and_indexed_tables(self) -> None:
        """Verify table index policy and standalone chart content."""
        range_table = pd.DataFrame({"value": [1, 2]})
        indexed_table = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=("num1", "num2"),
            columns=("num1", "num2"),
        )
        figure = go.Figure(data=go.Bar(x=(1, 2), y=(3, 4)))

        archive_data = build_export_zip(
            {"range_table": range_table, "indexed_table": indexed_table},
            {"bar_chart": figure},
        )

        with ZipFile(BytesIO(archive_data)) as archive:
            assert set(archive.namelist()) == {
                "tables/range_table.csv",
                "tables/indexed_table.csv",
                "charts/bar_chart.html",
            }
            range_csv = archive.read("tables/range_table.csv").decode()
            indexed_csv = archive.read("tables/indexed_table.csv").decode()
            chart_html = archive.read("charts/bar_chart.html").decode()

        assert range_csv.startswith("value")
        assert indexed_csv.startswith("row,num1,num2")
        assert "plotly.js" in chart_html
        assert "<html>" in chart_html
