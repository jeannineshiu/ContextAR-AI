"""
pytest configuration for ContextAR.

Custom markers:
    @pytest.mark.hardware  — tests that require a real camera or microphone.
                             Skipped by default; run with:
                                 python -m pytest --hardware
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run tests that require a real camera or microphone.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: mark test as requiring real camera/microphone hardware",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hardware"):
        return   # --hardware passed: run everything
    skip = pytest.mark.skip(reason="requires hardware — run with --hardware")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
