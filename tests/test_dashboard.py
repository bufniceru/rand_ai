"""Test the Streamlit trusted-upload statistics interface."""

import pickle
from pathlib import Path
from typing import cast

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import FileUploader

from rand_ai import Draw, Draws

_DASHBOARD_PATH = Path("src/rand_ai/dashboard.py")


def _dashboard() -> AppTest:
    """Return a newly executed dashboard test instance."""
    return AppTest.from_file(_DASHBOARD_PATH, default_timeout=30).run()


def _pickle_draws() -> bytes:
    """Return a small valid Draws pickle for interface tests."""
    draws = Draws()
    draws.add(Draw())
    draws.add(Draw(7, 8, 9, 10, 11, 12))
    draws.add(Draw(13, 14, 15, 16, 17, 18))
    return pickle.dumps(draws)


def _upload(dashboard: AppTest, filename: str, payload: bytes) -> None:
    """Upload one binary file through the dashboard test element."""
    uploader = cast(FileUploader, dashboard.get("file_uploader")[0])
    uploader.upload(filename, payload)


class TestDashboardUpload:
    """Test initial, trusted, valid, and invalid upload workflows."""

    def test_initial_view_requires_trusted_upload(self) -> None:
        """Verify the initial instructions and disabled analysis action."""
        dashboard = _dashboard()

        assert not dashboard.exception
        assert dashboard.checkbox[0].label == "I trust this pickle file"
        assert dashboard.button[0].label == "Analyze"
        assert dashboard.button[0].disabled
        assert "Upload a trusted Draws pickle" in dashboard.info[0].value

    def test_uploaded_file_still_requires_trust_confirmation(self) -> None:
        """Verify uploading alone does not enable unsafe pickle loading."""
        dashboard = _dashboard()
        _upload(dashboard, "draws.pkl", _pickle_draws())
        dashboard.run()

        assert dashboard.button[0].disabled
        assert "Select Analyze" in dashboard.info[0].value

    def test_valid_trusted_upload_renders_complete_dashboard(self) -> None:
        """Verify trusted analysis renders every tab, chart, metric, and export."""
        dashboard = _dashboard()
        dashboard.checkbox[0].check()
        _upload(dashboard, "draws.pkl", _pickle_draws())
        dashboard.run()
        dashboard.button[0].click().run(timeout=30)

        assert not dashboard.exception
        assert [tab.label for tab in dashboard.tabs] == [
            "Overview",
            "Numbers",
            "Spaces",
            "Relationships",
            "Randomness",
            "Export",
        ]
        assert len(dashboard.metric) == 4
        assert len(dashboard.get("plotly_chart")) == 21
        assert len(dashboard.get("download_button")) == 1
        assert any("do not predict" in warning.value for warning in dashboard.warning)

        dashboard.run(timeout=30)
        assert not dashboard.exception
        assert len(dashboard.get("download_button")) == 1

    @pytest.mark.parametrize(
        ("filename", "payload", "message"),
        (
            (
                "wrong.pkl",
                pickle.dumps({"not": "draws"}),
                "Pickle must contain a Draws instance",
            ),
            ("broken.pkl", b"not a pickle", "Analysis failed"),
        ),
    )
    def test_invalid_upload_displays_error(
        self, filename: str, payload: bytes, message: str
    ) -> None:
        """Verify wrong-type and corrupt trusted files fail visibly."""
        dashboard = _dashboard()
        dashboard.checkbox[0].check()
        _upload(dashboard, filename, payload)
        dashboard.run()
        dashboard.button[0].click().run(timeout=30)

        assert not dashboard.exception
        assert message in dashboard.error[0].value
