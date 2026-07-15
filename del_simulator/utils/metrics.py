from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import typing as T


def calculate_bedroc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decreasing: bool = True,
    alpha: float = 20.0,
) -> float:
    """BEDROC metric implemented according to Truchon and Bayley.

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true:
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred:
            Prediction values.
        decreasing:
            True if high values of ``y_pred`` correlates to positive class.
        alpha:
            Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
    """

    assert len(y_true) == len(
        y_pred
    ), "The number of scores must be equal to the number of labels"

    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = (
        r_a
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    )
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    bedroc = s * fac / rand_sum + cte

    return bedroc


def calculate_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the area under the receiver operating characteristic curve."""
    return roc_auc_score(y_true, y_pred)


def calculate_enrichment_factor(y_pred, y_true, top_percent=1) -> float:
    """
    Calculate the enrichment factor specified percent
    """
    top_percent = top_percent / 100
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]

    n_actives = np.sum(y_true)
    n_total = len(y_true)
    n_actives_in_x_percent = int(n_total * top_percent)
    n_actives_in_x_percent = max(1, n_actives_in_x_percent)
    enrichment_factor = (
        np.sum(y_true_sorted[:n_actives_in_x_percent])
        / n_actives_in_x_percent
        / (n_actives / n_total)
    )

    return enrichment_factor


def get_inference_metrics(y_pred_proba, y) -> T.Dict[str, float]:

    metrics = {}

    metrics["bedroc"] = calculate_bedroc_score(y_true=y, y_pred=y_pred_proba)
    metrics["enrichment_0.1pct"] = calculate_enrichment_factor(
        y_pred=y_pred_proba, y_true=y, top_percent=0.1
    )
    metrics["enrichment_1"] = calculate_enrichment_factor(
        y_pred=y_pred_proba, y_true=y, top_percent=1
    )
    metrics["enrichment_5"] = calculate_enrichment_factor(
        y_pred=y_pred_proba, y_true=y, top_percent=5
    )
    metrics["enrichment_10"] = calculate_enrichment_factor(
        y_pred=y_pred_proba, y_true=y, top_percent=10
    )
    metrics["auc"] = calculate_auc_roc(y_true=y, y_pred=y_pred_proba)
    metrics["f1"] = f1_score(y_true=y, y_pred=(y_pred_proba > 0.5).astype(int))

    return metrics
