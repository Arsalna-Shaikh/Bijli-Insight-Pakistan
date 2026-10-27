import pandas as pd
import pytest
import requests

from backend.pipeline import flow


def test_fetch_failure_logs_failed_fetch_and_writes_no_data(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(flow.requests, "get", offline)
    with pytest.raises(requests.ConnectionError):
        flow.task_fetch_live_observation.fn(city="karachi")

    run_log = tmp_path / "prefect_runs.csv"
    flow.log_failed_fetch(run_log, requests.ConnectionError("offline"))
    rows = pd.read_csv(run_log)
    assert list(rows["state"]) == ["FAILED_FETCH"]
    assert rows["model_mae"].isna().all()
