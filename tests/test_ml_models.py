import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from del_simulator.ml_models import (
    DELSimulatorMLModelHarness,
    DELSimulatorRFModel,
    DELSimulatorChemPropModel,
)
from del_simulator.core import (
    ScaledPoissonRatioTestMethodParameters,
    NormalizedZScoreMethodParameters,
    RandomForestClassifierParameters,
    ChemPropGNNParameters,
)


@pytest.fixture(scope="module")
def data_df():
    return pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCF"],
            "target_counts": [50, 10, 5, 100],
            "ntc_counts": [5, 4, 3, 6],
        }
    )


@pytest.fixture(scope="module")
def harness():
    return DELSimulatorMLModelHarness(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )


def test_calculate_enrichment_scaled_poisson_ratio_test(harness, data_df):
    """
    scaled_poisson_ratio_test is the default enrichment_method. Before
    TrainingAndInferenceConfig.__post_init__ promoted enrichment_method_params to a typed
    dataclass, this call would raise AttributeError: 'dict' object has no attribute
    'library_diversity' whenever enrichment_method_params was still a plain dict.
    """
    enrichment = harness.calculate_enrichment(
        data_df,
        enrichment_method="scaled_poisson_ratio_test",
        enrichment_method_params=ScaledPoissonRatioTestMethodParameters(
            library_diversity=100_000, poisson_ratio=3.0
        ),
    )

    assert len(enrichment) == len(data_df)
    assert enrichment.notna().all()


def test_calculate_enrichment_normalized_zscore(harness, data_df):
    enrichment = harness.calculate_enrichment(
        data_df,
        enrichment_method="normalized_zscore",
        enrichment_method_params=NormalizedZScoreMethodParameters(
            library_diversity=100_000
        ),
    )

    assert len(enrichment) == len(data_df)
    assert enrichment.notna().all()


def test_calculate_enrichment_count_ratio(harness, data_df):
    """count_ratio never reads enrichment_method_params, so None is the correct input."""
    enrichment = harness.calculate_enrichment(
        data_df,
        enrichment_method="count_ratio",
        enrichment_method_params=None,
    )

    assert len(enrichment) == len(data_df)
    assert enrichment.notna().all()


class _EchoModel:
    """Fake model whose predict_proba just returns its input, so callers can verify
    exactly which rows a batching routine actually processed."""

    def predict_proba(self, X_batch):
        return X_batch


def test_batch_inference_sparse_fps_processes_all_rows_with_small_batch_size():
    """
    _batch_inference_sparse_fps used to loop with a hardcoded range(0, len(X), 50000)
    while slicing X[i:i+batch_size] -- for any len(X) < 50000 (i.e. every realistic test
    or small screening set), the outer loop only ran once, silently processing only the
    first batch_size rows and dropping the rest.
    """
    harness = DELSimulatorRFModel(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )
    X = [csr_matrix([[value]]) for value in range(5)]

    y_pred = harness._batch_inference_sparse_fps(_EchoModel(), X, batch_size=2)

    assert y_pred.flatten().tolist() == [0, 1, 2, 3, 4]


def test_rf_screening_subsample_is_reproducible(tmp_path):
    """
    load_inference_data's sample_size branch used np.random.choice with no seed, unlike
    the equivalent ChemProp path and everything else in this pipeline (seeded at 42).
    """
    harness = DELSimulatorRFModel(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )

    featurized_data_path = tmp_path / "features.npy"
    with open(featurized_data_path, "wb") as f:
        np.save(f, np.arange(20))

    X_first, _ = harness.load_inference_data(
        featurized_data_path=str(featurized_data_path), sample_size=5
    )
    X_second, _ = harness.load_inference_data(
        featurized_data_path=str(featurized_data_path), sample_size=5
    )

    np.testing.assert_array_equal(X_first, X_second)


def test_rf_model_default_method_parameters_support_attribute_access():
    """
    method_parameters used to default to a mutable {"n_estimators": 100} dict literal,
    but train() reads self.params.n_estimators via attribute access -- a plain dict would
    raise AttributeError.
    """
    harness = DELSimulatorRFModel(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )

    assert harness.params.n_estimators == 100


def test_chemprop_model_default_method_parameters_support_attribute_access():
    """
    Same issue as the RF model, for DELSimulatorChemPropModel's
    {"batch_size": 128, "num_workers": 8, "max_epochs": 10} default. max_epochs=10 is
    preserved explicitly here since ChemPropGNNParameters' own dataclass default is 30.
    """
    harness = DELSimulatorChemPropModel(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )

    assert harness.params.batch_size == 128
    assert harness.params.num_workers == 8
    assert harness.params.max_epochs == 10


@pytest.fixture
def chemprop_screening_files(tmp_path):
    smiles_path = tmp_path / "smiles.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"]}).to_csv(
        smiles_path, index=False
    )

    affinity_path = tmp_path / "affinities.csv"
    pd.DataFrame({"pKd": [4.0, 5.0, 6.0, 7.0]}).to_csv(affinity_path, index=False)

    return smiles_path, affinity_path


def test_chemprop_load_inference_data_cache_invalidated_by_changed_threshold(
    chemprop_screening_files, tmp_path
):
    """
    load_inference_data's cache-hit check used to be purely "does featurized_data_path
    exist", ignoring affinity_threshold and every other parameter that determines the
    cached labels. Re-running with the same cache path but a different affinity_threshold
    (a natural sweep of the hit-calling cutoff) used to silently return the first run's
    stale labels.
    """
    smiles_path, affinity_path = chemprop_screening_files
    featurized_data_path = tmp_path / "cache.pkl"
    harness = DELSimulatorChemPropModel(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )

    _, labels_low_threshold = harness.load_inference_data(
        featurized_data_path=str(featurized_data_path),
        smiles_path=str(smiles_path),
        affinity_path=str(affinity_path),
        affinity_threshold=4.5,
    )
    _, labels_high_threshold = harness.load_inference_data(
        featurized_data_path=str(featurized_data_path),
        smiles_path=str(smiles_path),
        affinity_path=str(affinity_path),
        affinity_threshold=6.5,
    )

    assert labels_low_threshold.tolist() == [0, 1, 1, 1]
    assert labels_high_threshold.tolist() == [0, 0, 0, 1]


def test_chemprop_load_inference_data_cache_hit_with_unchanged_parameters(
    chemprop_screening_files, tmp_path
):
    """The cache should still be reused (and return consistent results) when nothing
    about the call has changed."""
    smiles_path, affinity_path = chemprop_screening_files
    featurized_data_path = tmp_path / "cache.pkl"
    harness = DELSimulatorChemPropModel(
        input_data_path="unused",
        model_output_path="unused",
        metrics_output_path="unused",
    )

    kwargs = dict(
        featurized_data_path=str(featurized_data_path),
        smiles_path=str(smiles_path),
        affinity_path=str(affinity_path),
        affinity_threshold=6.5,
    )

    _, labels_first = harness.load_inference_data(**kwargs)
    _, labels_second = harness.load_inference_data(**kwargs)

    assert labels_first.tolist() == labels_second.tolist()


def test_rf_model_train_predict_run_inference_end_to_end(tmp_path):
    """
    End-to-end exercise of train/_predict/run_inference/save_model/compute_and_save_metrics
    together with a real RandomForestClassifier, none of which was covered by a real
    training run before (only individual helper functions were tested in isolation).
    """
    rng = np.random.default_rng(0)
    n_features = 16
    n_train = 20
    X_train = rng.integers(0, 2, size=(n_train, n_features)).astype(np.float64)
    y_train = np.array([0, 1] * (n_train // 2))
    data_df = pd.DataFrame(
        {
            "fingerprint": [csr_matrix(row.reshape(1, -1)) for row in X_train],
            "class": y_train,
        }
    )

    harness = DELSimulatorRFModel(
        input_data_path="unused",
        model_output_path=str(tmp_path / "models"),
        metrics_output_path=str(tmp_path / "metrics.jsonl"),
        method_parameters=RandomForestClassifierParameters(n_estimators=5),
    )

    model = harness.train(data_df, n_cv_splits=2)

    assert (tmp_path / "models" / "model_0.pkl").exists()
    assert (tmp_path / "metrics.jsonl").exists()

    screening_fps = [
        csr_matrix(row.reshape(1, -1))
        for row in rng.integers(0, 2, size=(6, n_features))
    ]
    featurized_data_path = tmp_path / "screening.npy"
    with open(featurized_data_path, "wb") as f:
        np.save(f, np.array(screening_fps, dtype=object))

    affinity_path = tmp_path / "affinities.csv"
    pd.DataFrame({"pKd": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}).to_csv(
        affinity_path, index=False
    )

    y_pred = harness.run_inference(
        model,
        featurized_data_path=str(featurized_data_path),
        affinity_path=str(affinity_path),
        affinity_threshold=6.0,
    )

    assert y_pred.shape == (6,)
    assert ((y_pred >= 0) & (y_pred <= 1)).all()

    with open(tmp_path / "metrics.jsonl") as f:
        metrics_lines = f.readlines()
    # one line from train()'s validation fold, one from run_inference's screening pass
    assert len(metrics_lines) == 2


def test_chemprop_model_train_predict_run_inference_end_to_end(tmp_path):
    """
    End-to-end train -> predict -> run_inference on a tiny dataset with a real MPNN model
    and lightning Trainer (~7s on CPU). num_workers=1 is required, not 0: train() builds
    its dataloaders with persistent_workers=True, which PyTorch rejects for num_workers=0.
    """
    smiles = ["CCO", "CCN", "CCC", "CCF", "CCCl", "CCBr", "CCI", "CO", "CN", "CC"]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    data_df = pd.DataFrame({"smiles": smiles, "class": labels})

    harness = DELSimulatorChemPropModel(
        input_data_path="unused",
        model_output_path=str(tmp_path / "models"),
        metrics_output_path=str(tmp_path / "metrics.jsonl"),
        method_parameters=ChemPropGNNParameters(
            batch_size=2, num_workers=1, max_epochs=1
        ),
    )

    model = harness.train(data_df, n_cv_splits=2)

    smiles_path = tmp_path / "screening_smiles.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"]}).to_csv(
        smiles_path, index=False
    )
    affinity_path = tmp_path / "screening_affinities.csv"
    pd.DataFrame({"pKd": [4.0, 5.0, 6.0, 7.0]}).to_csv(affinity_path, index=False)

    y_pred = harness.run_inference(
        model,
        smiles_path=str(smiles_path),
        affinity_path=str(affinity_path),
        affinity_threshold=5.5,
    )

    assert y_pred.shape == (4,)
    assert ((y_pred >= 0) & (y_pred <= 1)).all()
    assert (tmp_path / "metrics.jsonl").exists()
