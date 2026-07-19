"""Provide the Streamlit interface for trusted Draws statistics analysis."""

import hashlib
from typing import BinaryIO, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rand_ai.draws import Draws
from rand_ai.statistics import CorrelationMethod, DrawsStatistics
from rand_ai.visualizations import build_export_zip, build_figures

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_ANALYSIS_KEYS = (
    "dataset_hash",
    "statistics",
    "statistics_tables",
    "figures",
    "figure_key",
    "export_zip",
    "export_key",
)


def _file_digest(uploaded_file: UploadedFile) -> str:
    """Return a stable SHA-256 digest without copying the upload buffer."""
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()


def _clear_analysis() -> None:
    """Remove all cached analysis state from the current Streamlit session."""
    for key in _ANALYSIS_KEYS:
        st.session_state.pop(key, None)


def _analyze_upload(uploaded_file: UploadedFile) -> DrawsStatistics:
    """Load a trusted uploaded pickle and build its compact statistics snapshot."""
    if uploaded_file.size > _MAX_UPLOAD_BYTES:
        raise ValueError("Pickle file must not exceed 100 MiB")
    uploaded_file.seek(0)
    draws = Draws.load_trusted_pickle(cast(BinaryIO, uploaded_file))
    return DrawsStatistics(draws)


def _summary_value(summary: pd.DataFrame, metric: str) -> int | float:
    """Return one named summary value from the two-column summary table."""
    matching_rows = summary.loc[summary["metric"] == metric, "value"]
    if matching_rows.empty:
        raise KeyError(metric)
    return cast(int | float, matching_rows.iloc[0])


def _render_plot(figure: go.Figure, *, key: str) -> None:
    """Render one responsive Plotly figure with the Streamlit theme."""
    st.plotly_chart(figure, width="stretch", theme="streamlit", key=key)


def _render_overview(
    statistics: DrawsStatistics,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
) -> None:
    """Render dataset measures and broad distribution plots."""
    summary = tables["summary"]
    columns = st.columns(4)
    columns[0].metric("Draws", f"{statistics.draw_count:,}")
    columns[1].metric(
        "Unique combinations",
        f"{int(_summary_value(summary, 'unique_combinations')):,}",
    )
    columns[2].metric(
        "Repeated draws", f"{int(_summary_value(summary, 'repeated_draws')):,}"
    )
    columns[3].metric("Heavy-analysis sample", f"{statistics.sample_size:,}")
    _render_plot(figures["number_frequencies"], key="overview-number-frequencies")
    _render_plot(figures["draw_sum_distribution"], key="overview-draw-sums")
    _render_plot(figures["draw_composition"], key="overview-composition")
    with st.expander("Overview statistics table"):
        st.dataframe(summary, width="stretch", hide_index=True)


def _render_numbers(
    tables: dict[str, pd.DataFrame], figures: dict[str, go.Figure]
) -> None:
    """Render number-frequency, position, pair, and trend analysis."""
    _render_plot(figures["number_frequencies"], key="numbers-frequency")
    _render_plot(figures["position_frequencies"], key="numbers-position")
    _render_plot(figures["pair_cooccurrence"], key="numbers-pairs")
    _render_plot(figures["number_trends"], key="numbers-trends")
    with st.expander("Number statistics tables"):
        st.subheader("Overall frequency")
        st.dataframe(tables["number_frequencies"], width="stretch", hide_index=True)
        st.subheader("Descriptive statistics")
        st.dataframe(tables["number_descriptive"], width="stretch", hide_index=True)


def _render_spaces(
    statistics: DrawsStatistics,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
) -> None:
    """Render exact and sampled analysis of the six circular spaces."""
    st.caption(
        "Every draw has six circular spaces whose values sum to 43. "
        f"Box plots use a deterministic sample of {statistics.sample_size:,} draws."
    )
    _render_plot(figures["space_frequencies"], key="spaces-frequency")
    _render_plot(figures["space_box_plots"], key="spaces-box")
    _render_plot(figures["space_extremes"], key="spaces-extremes")
    st.dataframe(tables["space_descriptive"], width="stretch", hide_index=True)


def _render_relationships(
    statistics: DrawsStatistics,
    correlation_method: CorrelationMethod,
    figures: dict[str, go.Figure],
) -> None:
    """Render number, space, and cross-correlation heatmaps."""
    qualifier = (
        f"a deterministic sample of {statistics.sample_size:,} draws"
        if correlation_method == "spearman"
        else "the complete dataset"
    )
    st.caption(
        f"{correlation_method.title()} correlations use {qualifier}. "
        "Sorted positions naturally introduce structural correlation."
    )
    _render_plot(figures["number_correlations"], key="relationships-numbers")
    _render_plot(figures["space_correlations"], key="relationships-spaces")
    _render_plot(figures["number_space_correlations"], key="relationships-number-space")


def _render_randomness(
    tables: dict[str, pd.DataFrame], figures: dict[str, go.Figure]
) -> None:
    """Render randomness diagnostics with interpretation safeguards."""
    st.warning(
        "These diagnostics describe this dataset. They do not predict future "
        "draws or make any number more likely in a fair lottery."
    )
    _render_plot(figures["matching_pairs"], key="randomness-matching-pairs")
    st.dataframe(tables["randomness_diagnostics"], width="stretch", hide_index=True)
    st.caption(
        "A chi-square p-value is only marked reliable when every expected "
        "number count is at least five. Entropy near one indicates a balanced "
        "aggregate frequency distribution."
    )


def _render_export(
    tables: dict[str, pd.DataFrame],
    figures: dict[str, go.Figure],
    figure_key: tuple[object, ...],
) -> None:
    """Render a cached ZIP download containing CSV and interactive HTML."""
    if st.session_state.get("export_key") != figure_key:
        with st.spinner("Preparing CSV tables and self-contained HTML charts..."):
            st.session_state["export_zip"] = build_export_zip(tables, figures)
            st.session_state["export_key"] = figure_key
    st.download_button(
        "Download statistics and charts",
        data=st.session_state["export_zip"],
        file_name="draws-statistics.zip",
        mime="application/zip",
        width="stretch",
    )
    st.caption(
        "The ZIP contains compact CSV result tables and standalone interactive "
        "Plotly HTML files."
    )


def _render_dashboard(statistics: DrawsStatistics, dataset_hash: str) -> None:
    """Render analysis controls, figures, tables, and exports."""
    st.sidebar.header("Analysis controls")
    selected_numbers = st.sidebar.multiselect(
        "Numbers in trend chart",
        options=list(range(1, 50)),
        default=[1, 2, 3, 4, 5, 6],
    )
    if not selected_numbers:
        st.sidebar.warning("Select at least one number. Number 1 is shown for now.")
        selected_numbers = [1]
    maximum_bins = min(500, statistics.draw_count)
    default_bins = min(100, maximum_bins)
    trend_bins = st.sidebar.slider(
        "Trend bins", min_value=1, max_value=maximum_bins, value=default_bins
    )
    correlation_method = st.sidebar.selectbox(
        "Correlation method", options=("pearson", "spearman")
    )

    figure_key: tuple[object, ...] = (
        dataset_hash,
        tuple(selected_numbers),
        trend_bins,
        correlation_method,
    )
    if st.session_state.get("figure_key") != figure_key:
        with st.spinner("Building interactive charts..."):
            st.session_state["figures"] = build_figures(
                statistics,
                selected_numbers=selected_numbers,
                trend_bins=trend_bins,
                correlation_method=correlation_method,
            )
            st.session_state["figure_key"] = figure_key
    figures = cast(dict[str, go.Figure], st.session_state["figures"])
    tables = cast(dict[str, pd.DataFrame], st.session_state["statistics_tables"])

    overview, numbers, spaces, relationships, randomness, export = st.tabs(
        (
            "Overview",
            "Numbers",
            "Spaces",
            "Relationships",
            "Randomness",
            "Export",
        )
    )
    with overview:
        _render_overview(statistics, tables, figures)
    with numbers:
        _render_numbers(tables, figures)
    with spaces:
        _render_spaces(statistics, tables, figures)
    with relationships:
        _render_relationships(statistics, correlation_method, figures)
    with randomness:
        _render_randomness(tables, figures)
    with export:
        _render_export(tables, figures, figure_key)


def main() -> None:
    """Run the trusted-upload Draws statistics dashboard."""
    st.set_page_config(page_title="Draws Statistics", layout="wide")
    st.title("Draws statistics")
    st.write(
        "Explore number values and circular spaces from an existing Draws dataset."
    )

    st.sidebar.header("Trusted dataset")
    st.sidebar.warning(
        "Pickle loading can execute code. Continue only with a file you created "
        "or otherwise fully trust."
    )
    trusted = st.sidebar.checkbox("I trust this pickle file")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Draws pickle", type=("pkl", "pickle")
    )
    analyze = st.sidebar.button(
        "Analyze",
        type="primary",
        disabled=uploaded_file is None or not trusted,
        width="stretch",
    )

    current_hash = _file_digest(uploaded_file) if uploaded_file is not None else None
    if analyze and uploaded_file is not None:
        _clear_analysis()
        try:
            with st.spinner("Loading and analyzing trusted data..."):
                statistics = _analyze_upload(uploaded_file)
                st.session_state["statistics"] = statistics
                st.session_state["statistics_tables"] = statistics.export_tables()
                st.session_state["dataset_hash"] = current_hash
        except Exception as error:
            st.error(f"Analysis failed: {error}")

    analyzed_hash = st.session_state.get("dataset_hash")
    if uploaded_file is None:
        st.info(
            "Upload a trusted Draws pickle, confirm that you trust it, and select "
            "Analyze."
        )
        return
    if analyzed_hash != current_hash:
        st.info("Select Analyze to process the currently uploaded file.")
        return

    statistics = cast(DrawsStatistics, st.session_state["statistics"])
    _render_dashboard(statistics, cast(str, analyzed_hash))


if __name__ == "__main__":
    main()
