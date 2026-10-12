"""Opt-in helpers for unit tests that execute resource-guarded stages inline."""

import pytest


@pytest.fixture
def controlled_stage_rss(monkeypatch):
    """Install deterministic RSS readings without bypassing real stage guards.

    Call explicitly from semantic tests, whose long-lived pytest process is
    not a fresh benchmark worker. Supplied peaks apply to successive stages.
    Remaining stages use the default. Real RSS/process-isolation tests must
    not use this fixture, and its readings are never resource evidence.
    """
    from algorithm_evaluation import paper_pipeline_runner

    def install(*stage_peaks, default=64 * 1024**2):
        peaks = iter(stage_peaks)

        class ControlledStageRss:
            def __enter__(self):
                self.peak = next(peaks, default)
                return self

            def __exit__(self, *_args):
                pass

            def report(self):
                return {
                    "method": "unit_test_controlled_rss",
                    "peak_rss_bytes": self.peak,
                    "sample_count": 1,
                }

        monkeypatch.setattr(
            paper_pipeline_runner, "PeakProcessTreeRss", ControlledStageRss
        )

    return install
