import numpy as np
import pytest

from del_simulator.utils.metrics import calculate_bedroc_score


def test_calculate_bedroc_score_matches_current_implementation():
    """
    Characterization test for the CURRENT calculate_bedroc_score, which is known to have
    an off-by-one ranking bug: m_rank uses 0-indexed rank positions
    (`(y_true[order] == 1).nonzero()[0]`) instead of the 1-indexed ranks the
    Truchon-Bayley formula requires. This inflates the score -- verified against the
    corrected formula (same computation with `+ 1` on m_rank) on the fixed input below,
    the current implementation returns ~49% higher than the correct value (0.950 vs
    0.637), consistent in magnitude with what an earlier review of this codebase found
    when checking against RDKit's reference BEDROC implementation.

    This bug is deliberately NOT fixed here. This test exists to give
    calculate_bedroc_score test coverage and lock in its current (buggy) behavior, so
    that if/when the bug is fixed, this test has to be deliberately and visibly updated
    rather than the behavior silently changing underneath an untested function. See
    del_simulator_review.pdf and del_simulator_coverage_report.pdf for the full
    diagnosis and the decision to defer the fix.
    """
    n_actives = 5
    n_total = 50
    y_true = np.array([1] * n_actives + [0] * (n_total - n_actives), dtype=float)
    y_pred = np.array(
        [
            1.1305,
            0.4895,
            0.5792,
            0.1458,
            1.0813,
            0.9128,
            0.6066,
            0.7295,
            0.5436,
            0.9351,
            0.8159,
            0.0027,
            0.8574,
            0.0336,
            0.7297,
            0.1757,
            0.8632,
            0.5415,
            0.2997,
            0.4227,
            0.0283,
            0.1243,
            0.6706,
            0.6472,
            0.6154,
            0.3837,
            0.9972,
            0.9808,
            0.6855,
            0.6505,
            0.6884,
            0.3889,
            0.1351,
            0.7215,
            0.5254,
            0.3102,
            0.4858,
            0.8895,
            0.9340,
            0.3578,
            0.5715,
            0.3219,
            0.5943,
            0.3379,
            0.3916,
            0.8903,
            0.2272,
            0.6232,
            0.0840,
            0.8326,
        ]
    )

    score = calculate_bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0)

    assert score == pytest.approx(0.9501139602137613)
