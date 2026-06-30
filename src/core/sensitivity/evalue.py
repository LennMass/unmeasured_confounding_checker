"""
E-value: how strong would an unmeasured confounder need to be to explain
away the observed effect?

Based on VanderWeele & Ding (2017). The E-value is the minimum strength of
association (on the risk ratio scale) that an unmeasured confounder would
need to have with both treatment and outcome to fully explain away the
observed treatment-outcome association.

For a continuous outcome we convert the standardized effect to an
approximate risk ratio following Ding & VanderWeele's recommended formula.
This is an approximation, adapt it to your outcome type for real research.
"""

import math

from src.core.schemas import CausalEstimate, EValueResult


def _rr_from_estimate(coefficient: float, std_error: float) -> tuple[float, float]:
    """Approximate risk ratio from a (standardized) effect estimate."""
    # Standardized effect size proxy, then Ding & VanderWeele transform
    d = coefficient / (std_error * math.sqrt(2)) if std_error > 0 else 0.0
    d = max(min(d, 5.0), -5.0)  # cap to avoid overflow
    rr_point = math.exp(0.91 * d)

    d_lower = (coefficient - 1.96 * std_error) / (std_error * math.sqrt(2)) if std_error > 0 else 0.0
    d_lower = max(min(d_lower, 5.0), -5.0)
    rr_ci = math.exp(0.91 * d_lower)

    return rr_point, rr_ci


def _evalue_from_rr(rr: float) -> float:
    """Compute the E-value from a risk ratio."""
    if rr < 1:
        rr = 1 / rr  # E-value is symmetric; work with RR >= 1
    if rr <= 1:
        return 1.0
    return rr + math.sqrt(rr * (rr - 1))


def compute_evalue(estimate: CausalEstimate) -> EValueResult:
    """Compute E-values for the point estimate and the CI bound."""

    rr_point, rr_ci = _rr_from_estimate(estimate.coefficient, estimate.std_error)

    evalue_point = _evalue_from_rr(rr_point)
    # If the CI crosses the null, the E-value for the CI is 1 (no robustness)
    evalue_ci = 1.0 if estimate.ci_lower <= 0 <= estimate.ci_upper else _evalue_from_rr(rr_ci)

    interpretation = (
        f"An unmeasured confounder would need to be associated with both treatment "
        f"and outcome by a risk ratio of at least {evalue_point:.2f} (above and beyond "
        f"the measured confounders) to fully explain away the observed effect. "
    )
    if evalue_ci > 1:
        interpretation += (
            f"To shift the confidence interval to include the null, a confounder of "
            f"strength {evalue_ci:.2f} would suffice."
        )
    else:
        interpretation += "The confidence interval already includes the null."

    return EValueResult(
        evalue_point=round(evalue_point, 4),
        evalue_ci=round(evalue_ci, 4),
        interpretation=interpretation,
    )