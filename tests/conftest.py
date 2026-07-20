import pytest


@pytest.fixture(autouse=True)
def _no_run_archive(monkeypatch):
    """Keep tests hermetic: never write run archives into the repo tree.

    Tests that exercise archiving pass an explicit `runs_dir` (pointing at
    tmp_path), which takes precedence over this env default."""
    monkeypatch.setenv("RUNS_DIR", "")
