"""Tests for the core sensitivity modules."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import pytest

from src.core.pipeline import run_pipeline
from src.core.estimators import estimate
from src.core.sensitivity.evalue import compute_evalue, _evalue_from_rr
from src.core.sensitivity.placebo import run_placebo_test
from src.core.sensitivity.learner_compare import compare_learners
from src.core.schemas import CausalEstimate

SAMPLE = Path("data/sample_data.csv")


@pytest.fixture(scope="module")
def data():
    return run_pipeline(SAMPLE, "treatment", "outcome")


def test_pipeline_loads(data):
    assert data.n_obs == 800
    assert data.n_features >= 3


def test_estimate_recovers_effect(data):
    # True effect is 300; estimate should be in a sane range
    est = estimate(data, estimator="plr", learner="sklearn", log_mlflow=False)
    assert isinstance(est, CausalEstimate)
    assert 150 < est.coefficient < 450
    assert est.significant


def test_evalue_formula():
    # RR of 2 -> E-value = 2 + sqrt(2*1) = 3.414
    assert _evalue_from_rr(2.0) == pytest.approx(3.414, abs=0.01)
    # RR of 1 -> E-value = 1 (no robustness)
    assert _evalue_from_rr(1.0) == 1.0


def test_evalue_runs(data):
    est = estimate(data, estimator="plr", learner="sklearn", log_mlflow=False)
    ev = compute_evalue(est)
    assert ev.evalue_point >= 1.0
    assert "confounder" in ev.interpretation.lower()


def test_placebo_real_effect_in_tail(data):
    est = estimate(data, estimator="plr", learner="sklearn", log_mlflow=False)
    placebo = run_placebo_test(data, est, n_permutations=20)
    # A real effect should have a low placebo p-value
    assert placebo.placebo_p_value < 0.2
    assert placebo.n_permutations <= 20


def test_learner_comparison(data):
    results = compare_learners(data, learners=["sklearn", "xgboost"], estimator="plr")
    assert len(results) == 2
    learners = {r.learner for r in results}
    assert learners == {"sklearn", "xgboost"}