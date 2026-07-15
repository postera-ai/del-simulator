import pytest
from omegaconf import OmegaConf

from del_simulator.core import (
    TrainingAndInferenceConfig,
    NormalizedZScoreMethodParameters,
    ScaledPoissonRatioTestMethodParameters,
    FingerPrintFeaturizerConfig,
    MorganFingerPrintGeneratorParameters,
    RDKitFingerPrintGeneratorParameters,
    AffinityGenConfig,
    AffinityGenSimilarityMethodParameters,
    AffinityGenRandomMethodParameters,
    Distribution,
    KernelParameters,
    TverskyParameters,
    TruncatedNormalParameters,
    LogNormalParameters,
)


def _load_run_config(run_overrides: dict):
    """
    Build a single per-run TrainingAndInferenceConfig the same way load_config()
    (del_simulator/utils/utils.py) does in production: OmegaConf.structured() for the
    schema, merged with a plain dict, then OmegaConf.to_object().

    Note: core.py defines TrainingAndInferenceConfig twice -- a per-run schema and an
    outer `runs: Dict[str, TrainingAndInferenceConfig]` wrapper that shadows it under the
    importable name. Going through the wrapper (as production code does) is the only way
    to reach the per-run class's __post_init__ from outside core.py.
    """
    base = {
        "input_data_path": "x",
        "model_output_path": "y",
        "metrics_output_path": "z",
        "ml_config": {"method": "random_forest"},
    }
    schema = OmegaConf.structured(TrainingAndInferenceConfig)
    conf = OmegaConf.create({"runs": {"run": {**base, **run_overrides}}})
    result = OmegaConf.to_object(OmegaConf.merge(schema, conf))
    return result.runs["run"]


def test_scaled_poisson_ratio_test_promotes_params():
    """
    This is the default enrichment_method and the one that used to crash
    calculate_enrichment with AttributeError, since enrichment_method_params stayed a
    plain dict after config loading.
    """
    run = _load_run_config(
        {
            "enrichment_method": "scaled_poisson_ratio_test",
            "enrichment_method_params": {
                "library_diversity": 100_000,
                "poisson_ratio": 3.0,
            },
        }
    )

    assert isinstance(
        run.enrichment_method_params, ScaledPoissonRatioTestMethodParameters
    )
    assert run.enrichment_method_params.library_diversity == 100_000
    assert run.enrichment_method_params.poisson_ratio == 3.0


def test_normalized_zscore_promotes_params():
    run = _load_run_config(
        {
            "enrichment_method": "normalized_zscore",
            "enrichment_method_params": {"library_diversity": 5_000},
        }
    )

    assert isinstance(run.enrichment_method_params, NormalizedZScoreMethodParameters)
    assert run.enrichment_method_params.library_diversity == 5_000


def test_count_ratio_leaves_params_untouched():
    """count_ratio never reads enrichment_method_params, so it should stay None."""
    run = _load_run_config({"enrichment_method": "count_ratio"})

    assert run.enrichment_method_params is None


def test_scaled_poisson_ratio_test_without_params_fails_fast():
    """
    scaled_poisson_ratio_test is the default enrichment_method. Omitting
    enrichment_method_params should fail immediately and clearly at config-load time,
    rather than silently producing a plain dict that later crashes deep inside
    calculate_enrichment with a confusing AttributeError.
    """
    with pytest.raises(Exception):
        _load_run_config({})


# FingerPrintFeaturizerConfig is typed with a plain `str` fp_generator_method (no OmegaConf
# enum coercion involved), so direct construction is a faithful reproduction of production use.


def test_fingerprint_featurizer_config_promotes_morgan_params():
    config = FingerPrintFeaturizerConfig(
        fp_generator_method="morgan",
        fp_generator_method_parameters={"radius": 2, "nBits": 1024},
    )

    assert isinstance(
        config.fp_generator_method_parameters, MorganFingerPrintGeneratorParameters
    )
    assert config.fp_generator_method_parameters.radius == 2
    assert config.fp_generator_method_parameters.nBits == 1024


def test_fingerprint_featurizer_config_promotes_rdkit_params():
    config = FingerPrintFeaturizerConfig(
        fp_generator_method="rdkit",
        fp_generator_method_parameters={"maxPath": 3},
    )

    assert isinstance(
        config.fp_generator_method_parameters, RDKitFingerPrintGeneratorParameters
    )
    assert config.fp_generator_method_parameters.maxPath == 3


def test_fingerprint_featurizer_config_maccs_has_no_params():
    config = FingerPrintFeaturizerConfig(
        fp_generator_method="maccs",
        fp_generator_method_parameters=None,
    )

    assert config.fp_generator_method_parameters is None


def test_fingerprint_featurizer_config_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown fp_generator_method"):
        FingerPrintFeaturizerConfig(
            fp_generator_method="ecfp",
            fp_generator_method_parameters=None,
        )


# AffinityGenSimilarityMethodParameters/AffinityGenRandomMethodParameters are only ever
# reached in production through AffinityGenConfig.runs (loaded via OmegaConf), which is what
# resolves AffinityGenRunConfig.method into a real AffinityMethod enum member before
# __post_init__ runs -- constructing AffinityGenRunConfig directly with a raw method string
# would not reflect how it's actually reached, so we go through the same OmegaConf path
# scripts/generate_affinities.py uses.


def _load_affinity_run_config(run_overrides: dict):
    schema = OmegaConf.structured(AffinityGenConfig)
    conf = OmegaConf.create(
        {
            "input_library_path": "x",
            "output_path": "y",
            "runs": {"run": run_overrides},
        }
    )
    result = OmegaConf.to_object(OmegaConf.merge(schema, conf))
    return result.runs["run"]


def test_affinity_gen_run_config_similarity_method_promotes_nested_dataclasses():
    run = _load_affinity_run_config(
        {
            "method": "similarity",
            "method_parameters": {
                "reference_affinity_path": "unused",
                "sampler": "truncated_normal",
                "kernel_params": {"a": 1.0, "c": 2.0},
                "tversky_params": {"a": 0.5, "b": 0.5},
                "fp_generator_method": "morgan",
                "fp_generator_method_parameters": {"radius": 2, "nBits": 1024},
                "sampler_params": {"loc": 0.0, "scale": 1.0},
            },
        }
    )

    params = run.method_parameters
    assert isinstance(params, AffinityGenSimilarityMethodParameters)
    assert params.sampler is Distribution.truncated_normal
    assert isinstance(params.kernel_params, KernelParameters)
    assert isinstance(params.tversky_params, TverskyParameters)
    assert isinstance(
        params.fp_generator_method_parameters, MorganFingerPrintGeneratorParameters
    )
    assert isinstance(params.sampler_params, TruncatedNormalParameters)
    assert params.sampler_params.loc == 0.0


def test_affinity_gen_run_config_random_method_promotes_lognormal_sampler_params():
    run = _load_affinity_run_config(
        {
            "method": "random",
            "method_parameters": {
                "sampler": "lognormal",
                "sampler_params": {"s": 1.0, "loc": 0.0, "scale": 1.0},
            },
        }
    )

    params = run.method_parameters
    assert isinstance(params, AffinityGenRandomMethodParameters)
    assert isinstance(params.sampler_params, LogNormalParameters)
    assert params.sampler_params.s == 1.0


def test_affinity_gen_random_method_parameters_invalid_sampler_raises():
    """
    sampler_params/sampler live under method_parameters, typed T.Any -- OmegaConf can't
    validate the sampler name at config-load time, so a typo'd sampler only surfaces here,
    as a plain ValueError from Distribution(...) construction.
    """
    with pytest.raises(ValueError, match="not a valid Distribution"):
        AffinityGenRandomMethodParameters(sampler="uniform")
