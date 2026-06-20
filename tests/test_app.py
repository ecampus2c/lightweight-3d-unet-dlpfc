"""Smoke tests for the Gradio web app. Skipped if Gradio is not installed.

These verify the interface builds and input validation works without requiring
TensorFlow (model inference is not exercised here).
"""
import pytest

pytest.importorskip("gradio", reason="gradio not installed")

from app.app import build_demo, segment   # noqa: E402


def test_demo_builds():
    demo = build_demo()
    assert demo is not None


def test_segment_rejects_missing_file():
    image, info, mask = segment(None, False)
    assert image is None
    assert mask is None
    assert "upload" in info.lower()
