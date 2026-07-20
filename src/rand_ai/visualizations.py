"""Build Plotly figures and downloadable statistics archives."""

from collections.abc import Collection, Mapping
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rand_ai.statistics import SPACE_NAMES, CorrelationMethod, DrawsStatistics


def _heatmap(
    values: pd.DataFrame,
    *,
    title: str,
    x_title: str,
    y_title: str,
    color_title: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Build a consistently labeled interactive heatmap."""
    figure = go.Figure(
        data=go.Heatmap(
            z=values.to_numpy(),
            x=[str(value) for value in values.columns],
            y=[str(value) for value in values.index],
            colorscale="Viridis",
            colorbar={"title": color_title},
            zmin=zmin,
            zmax=zmax,
            hovertemplate=f"{y_title}: %{{y}}<br>{x_title}: %{{x}}"
            f"<br>{color_title}: %{{z}}<extra></extra>",
        )
    )
    figure.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return figure


def _number_frequency_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build observed and expected number-frequency traces."""
    frequencies = statistics.number_frequencies()
    figure = go.Figure()
    figure.add_bar(
        x=frequencies["number"],
        y=frequencies["count"],
        name="Observed",
        hovertemplate="Number %{x}<br>Count %{y}<extra></extra>",
    )
    figure.add_scatter(
        x=frequencies["number"],
        y=frequencies["expected_count"],
        name="Expected",
        mode="lines",
        hovertemplate="Number %{x}<br>Expected %{y:.2f}<extra></extra>",
    )
    figure.update_layout(
        title="Number frequency compared with uniform expectation",
        xaxis_title="Number",
        yaxis_title="Appearances",
        barmode="overlay",
    )
    return figure


def _draw_sum_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build an exact pre-aggregated draw-sum distribution."""
    distributions = statistics.draw_structure_distributions()
    draw_sums = distributions[distributions["measure"] == "draw_sum"]
    return px.bar(
        draw_sums,
        x="value",
        y="count",
        title="Draw-sum distribution",
        labels={"value": "Sum of six numbers", "count": "Draws"},
    )


def _composition_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build small multiples for common draw composition measures."""
    distributions = statistics.draw_structure_distributions()
    measures = (
        ("odd_count", "Odd numbers"),
        ("low_count", "Numbers from 1 to 24"),
        ("consecutive_pair_count", "Consecutive pairs"),
    )
    figure = make_subplots(
        rows=1, cols=3, subplot_titles=[item[1] for item in measures]
    )
    for column, (measure, label) in enumerate(measures, start=1):
        values = distributions[distributions["measure"] == measure]
        figure.add_trace(
            go.Bar(
                x=values["value"],
                y=values["count"],
                name=label,
                showlegend=False,
                hovertemplate=f"{label}: %{{x}}<br>Draws: %{{y}}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="Count in draw", row=1, col=column)
    figure.update_yaxes(title_text="Draws", row=1, col=1)
    figure.update_layout(title="Draw composition distributions")
    return figure


def _position_frequency_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build a number-by-position frequency heatmap."""
    frequencies = statistics.position_frequencies()
    matrix = frequencies.pivot(index="position", columns="number", values="count")
    return _heatmap(
        matrix,
        title="Number frequency by sorted position",
        x_title="Number",
        y_title="Position",
        color_title="Count",
    )


def _pair_cooccurrence_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build an exact pair co-occurrence heatmap."""
    pairs = statistics.pair_cooccurrence()
    matrix = pairs.pivot(index="number_a", columns="number_b", values="count")
    return _heatmap(
        matrix,
        title="Number pair co-occurrence",
        x_title="Second number",
        y_title="First number",
        color_title="Count",
    )


def _trend_figure(
    statistics: DrawsStatistics,
    selected_numbers: Collection[int],
    trend_bins: int,
) -> go.Figure:
    """Build exact binned appearance-rate trends for selected numbers."""
    trend = statistics.trend(selected_numbers, bins=trend_bins)
    figure = px.line(
        trend,
        x="bin",
        y="appearance_rate",
        color=trend["number"].astype(str),
        markers=True,
        title="Appearance rate by draw-index bin",
        labels={
            "bin": "Draw-index bin",
            "appearance_rate": "Appearance rate (%)",
            "color": "Number",
        },
        hover_data={"start_draw": True, "end_draw": True, "count": True},
    )
    return figure


def _space_frequency_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build a space-value frequency heatmap."""
    frequencies = statistics.space_frequencies()
    matrix = frequencies.pivot(index="position", columns="space", values="count")
    return _heatmap(
        matrix,
        title="Space frequency by position",
        x_title="Space value",
        y_title="Space position",
        color_title="Count",
    )


def _distance_frequency_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build aggregate occurrence counts for distance values 0 through 43."""
    frequencies = statistics.distance_frequencies()
    figure = px.bar(
        frequencies,
        x="distance",
        y="occurrences",
        title="Distance occurrences across all six positions",
        labels={
            "distance": "Distance (0–43)",
            "occurrences": "Occurrences",
            "occurrence_percentage": "Share of occurrences (%)",
        },
        hover_data={"occurrence_percentage": ":.2f"},
    )
    figure.update_xaxes(range=(-0.5, 43.5), dtick=1)
    return figure


def _distance_position_frequency_figures(
    statistics: DrawsStatistics,
) -> dict[str, go.Figure]:
    """Build one occurrence chart for each individual distance position."""
    frequencies = statistics.space_frequencies()
    figures: dict[str, go.Figure] = {}
    for position in SPACE_NAMES:
        position_frequencies = frequencies[frequencies["position"] == position]
        figure = px.bar(
            position_frequencies,
            x="space",
            y="count",
            title=f"{position} distance occurrences",
            labels={
                "space": "Distance (0–43)",
                "count": "Occurrences",
                "percentage": "Draws with this distance (%)",
            },
            hover_data={"percentage": ":.2f"},
        )
        figure.update_xaxes(range=(-0.5, 43.5), dtick=1)
        figures[f"{position}_frequencies"] = figure
    return figures


def _space_box_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build sampled space box plots with an explicit sample-size title."""
    sampled_spaces = statistics.sampled_spaces()
    return px.box(
        sampled_spaces,
        x="position",
        y="space",
        points=False,
        title=f"Space distributions (deterministic sample: {statistics.sample_size:,} draws)",
        labels={"position": "Space position", "space": "Space value"},
    )


def _space_extremes_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build minimum- and maximum-space distributions."""
    extremes = statistics.space_extreme_distributions()
    return px.bar(
        extremes,
        x="value",
        y="count",
        color="measure",
        barmode="group",
        title="Minimum and maximum space distributions",
        labels={"value": "Space value", "count": "Draws", "measure": "Measure"},
    )


def _correlation_figures(
    statistics: DrawsStatistics, method: CorrelationMethod
) -> dict[str, go.Figure]:
    """Build number, space, and cross-correlation heatmaps."""
    correlations = statistics.correlations(method)
    suffix = "sampled" if method == "spearman" else "exact"
    return {
        "number_correlations": _heatmap(
            correlations["numbers"],
            title=f"Number-position {method.title()} correlations ({suffix})",
            x_title="Position",
            y_title="Position",
            color_title="Correlation",
            zmin=-1,
            zmax=1,
        ),
        "space_correlations": _heatmap(
            correlations["spaces"],
            title=f"Space-position {method.title()} correlations ({suffix})",
            x_title="Space",
            y_title="Space",
            color_title="Correlation",
            zmin=-1,
            zmax=1,
        ),
        "number_space_correlations": _heatmap(
            correlations["number_space"],
            title=f"Number-to-space {method.title()} correlations ({suffix})",
            x_title="Space",
            y_title="Number position",
            color_title="Correlation",
            zmin=-1,
            zmax=1,
        ),
    }


def _matching_pairs_figure(statistics: DrawsStatistics) -> go.Figure:
    """Build observed-versus-expected matching-combination bars."""
    diagnostics = statistics.randomness_diagnostics().set_index("diagnostic")
    values = pd.DataFrame(
        {
            "measure": ("Observed", "Expected"),
            "matching_pairs": (
                diagnostics.loc["observed_matching_combination_pairs", "value"],
                diagnostics.loc["expected_matching_combination_pairs", "value"],
            ),
        }
    )
    return px.bar(
        values,
        x="measure",
        y="matching_pairs",
        title="Matching six-number combination pairs",
        labels={"measure": "", "matching_pairs": "Matching pairs"},
    )


def build_figures(
    statistics: DrawsStatistics,
    *,
    selected_numbers: Collection[int] = (1, 2, 3, 4, 5, 6),
    trend_bins: int = 100,
    correlation_method: CorrelationMethod = "pearson",
) -> dict[str, go.Figure]:
    """Build every dashboard figure from compact or bounded data."""
    figures = {
        "number_frequencies": _number_frequency_figure(statistics),
        "draw_sum_distribution": _draw_sum_figure(statistics),
        "draw_composition": _composition_figure(statistics),
        "position_frequencies": _position_frequency_figure(statistics),
        "pair_cooccurrence": _pair_cooccurrence_figure(statistics),
        "number_trends": _trend_figure(
            statistics, selected_numbers=selected_numbers, trend_bins=trend_bins
        ),
        "distance_frequencies": _distance_frequency_figure(statistics),
        "space_frequencies": _space_frequency_figure(statistics),
        "space_box_plots": _space_box_figure(statistics),
        "space_extremes": _space_extremes_figure(statistics),
        "matching_pairs": _matching_pairs_figure(statistics),
    }
    figures.update(_distance_position_frequency_figures(statistics))
    figures.update(_correlation_figures(statistics, correlation_method))
    return figures


def build_export_zip(
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, go.Figure],
) -> bytes:
    """Return CSV tables and self-contained Plotly HTML files in one ZIP."""
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, table in tables.items():
            include_index = not isinstance(table.index, pd.RangeIndex)
            archive.writestr(
                f"tables/{name}.csv",
                table.to_csv(index=include_index, index_label="row"),
            )
        for name, figure in figures.items():
            archive.writestr(
                f"charts/{name}.html",
                figure.to_html(full_html=True, include_plotlyjs=True),
            )
    return archive_buffer.getvalue()
