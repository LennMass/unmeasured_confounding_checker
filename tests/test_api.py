"""Tests for the FastAPI endpoints."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)
SAMPLE = "data/sample_data.csv"


def test_health():
    assert client.get("/").json()["status"] == "ok"


def test_options():
    data = client.get("/options").json()
    assert "plr" in data["estimators"]
    assert "sklearn" in data["learners"]


def test_estimate():
    with open(SAMPLE, "rb") as f:
        resp = client.post(
            "/estimate",
            files={"file": ("sample.csv", f, "text/csv")},
            params={"learner": "sklearn"},
        )
    assert resp.status_code == 200
    assert "coefficient" in resp.json()


def test_sensitivity():
    with open(SAMPLE, "rb") as f:
        resp = client.post(
            "/sensitivity",
            files={"file": ("sample.csv", f, "text/csv")},
            params={"n_placebo": 10},
        )
    assert resp.status_code == 200
    report = resp.json()
    assert "estimation" in report
    assert "evalue" in report
    assert "placebo" in report
    assert "learner_robustness" in report