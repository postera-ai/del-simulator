from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from del_simulator.affinity_calculator import (
    SimilarityAffinityCalculator,
    RandomAffinityCalculator,
    TruncatedNormalSampler,
    LogNormalSampler,
    FingerprintGeneratorMethod,
)
from del_simulator.core import (
    Library,
    KernelParameters,
    TverskyParameters,
    MorganFingerPrintGeneratorParameters,
)


@pytest.fixture(scope="module")
def simple_data():
    query_lib_df = pd.read_csv(Path(__file__).parent / "resources/dag0_bbs0_output.csv")

    query_lib = Library(
        nsynthon_id=list(query_lib_df["nsynthon_id"]),
        smiles=list(query_lib_df["smiles"]),
        relative_fraction=np.array(query_lib_df["relative_fraction"]),
    )

    reference_data = pd.read_csv(Path(__file__).parent / "resources/affinities.csv")

    expected_df = pd.read_csv(
        Path(__file__).parent / "resources/dag0_bbs0_output_affinities.csv"
    )

    return query_lib, reference_data, expected_df


@pytest.mark.parametrize("test_data", ["simple_data"])
# this request / fixture name business is so we can parametrize across fixtures
# https://github.com/pytest-dev/pytest/issues/349
def test_affinity_calculator(test_data, request):
    query_lib, reference_data, expected_df = request.getfixturevalue(test_data)

    # a fixed seed makes this reproducible: the sampler and get_affinities derive their
    # per-molecule randomness from a hash of each SMILES string plus this seed
    sampler = TruncatedNormalSampler(loc=7.0, scale=0.5, seed=42)

    affinity_calculator = SimilarityAffinityCalculator(
        reference_affinities=list(reference_data["CNNaffinity"]),
        reference_smiles=list(reference_data["smiles"]),
        sampler=sampler,
        kernel_params=KernelParameters(a=0.35, c=30.0),
        tversky_params=TverskyParameters(a=1.0, b=0.0),
        fp_generator_method=FingerprintGeneratorMethod.morgan,
        fp_generator_method_parameters=MorganFingerPrintGeneratorParameters(
            radius=2, nBits=512
        ),
        parallelism=1,
    )

    affinities, _ = affinity_calculator.get_affinities(query_library=query_lib)

    result_df = pd.DataFrame(
        {
            "nsynthon_id": query_lib.nsynthon_id,
            "smiles": query_lib.smiles,
            "pKd": affinities,
        }
    )

    for df in (result_df, expected_df):
        df.sort_values(by=["nsynthon_id", "smiles"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(
        result_df[["nsynthon_id", "smiles", "pKd"]],
        expected_df[["nsynthon_id", "smiles", "pKd"]],
    )


def test_similarity_affinity_calculator_get_affinities_matches_naive_diagonal():
    """
    _get_affinities computes np.sum(sim_mx * affs.T, axis=1) instead of the mathematically
    equivalent but O(num_query^2) np.matmul(sim_mx, affs).diagonal(). This locks the
    vectorized computation against an independently-written naive reimplementation of the
    original diagonal-extraction formula, on a query set asymmetric in size to the
    reference set.
    """
    reference_data = pd.read_csv(Path(__file__).parent / "resources/affinities.csv")

    calc = SimilarityAffinityCalculator(
        reference_affinities=list(reference_data["CNNaffinity"]),
        reference_smiles=list(reference_data["smiles"]),
        sampler=TruncatedNormalSampler(loc=7.0, scale=0.5, seed=42),
        kernel_params=KernelParameters(a=0.35, c=30.0),
        tversky_params=TverskyParameters(a=1.0, b=0.0),
        fp_generator_method=FingerprintGeneratorMethod.morgan,
        fp_generator_method_parameters=MorganFingerPrintGeneratorParameters(
            radius=2, nBits=512
        ),
        parallelism=1,
    )

    query_smiles = [
        "CCCCCN",
        "CCCCCNC(=O)Cc1cc(Br)ccc1[N+](=O)[O-]",
        "c1ccccc1",
    ]
    baseline_affinities = np.array([6.1, 6.4, 5.9])

    vectorized = calc._get_affinities(query_smiles, baseline_affinities)

    # independent ground truth: the original np.matmul(...).diagonal() formula, reimplemented
    # inline rather than calling the (now-fixed) method under test
    query_fingerprints = calc.compute_fingerprints(query_smiles)
    sim_mx = calc.compute_tversky_similarity_mx(
        calc.reference_fingerprints,
        query_fingerprints,
        a=calc.tversky_params.a,
        b=calc.tversky_params.b,
    )
    sim_mx = calc.gaussian_kernel(sim_mx)
    sim_mx = np.hstack([sim_mx, (1.0 - np.max(sim_mx, axis=1)).T.reshape(-1, 1)])
    sim_mx = np.exp(calc.kernel_params.c * sim_mx) / np.sum(
        np.exp(calc.kernel_params.c * sim_mx), axis=1, keepdims=True
    )
    affs = np.vstack(
        [
            np.tile(calc.reference_affinities, (len(query_smiles), 1)).T,
            baseline_affinities,
        ]
    )
    naive = np.matmul(sim_mx, affs).diagonal()

    np.testing.assert_allclose(vectorized, naive)


def test_truncated_normal_sampler_seed_does_not_overflow():
    """
    sample() combines a per-SMILES seed (already clamped to [0, 2**32-1] by
    RandomAffinityCalculator._get_affinities) with self.seed via addition. Before the fix,
    a per-SMILES seed near the top of that range could push the sum past 2**32-1, and
    scipy/numpy's random_state requires a seed strictly in [0, 2**32-1].
    """
    sampler = TruncatedNormalSampler(seed=42)

    # the exact boundary case that used to raise
    # ValueError: Seed must be between 0 and 2**32 - 1
    value = sampler.sample(seed=2**32 - 1)
    assert np.isfinite(value)

    # deterministic: a fresh instance (bypassing the @lru_cache on the first one) with the
    # same seed must reproduce the same value
    other_sampler = TruncatedNormalSampler(seed=42)
    assert other_sampler.sample(seed=2**32 - 1) == value


def test_log_normal_sampler_seed_does_not_overflow():
    sampler = LogNormalSampler(seed=42)

    value = sampler.sample(seed=2**32 - 1)
    assert np.isfinite(value)

    other_sampler = LogNormalSampler(seed=42)
    assert other_sampler.sample(seed=2**32 - 1) == value


def test_random_affinity_calculator_unique_affinities_are_in_sorted_smiles_order():
    """
    get_affinities used plain list(set(smiles)) to build unique_smiles, whose iteration
    order is only stable for the lifetime of one process (it depends on PYTHONHASHSEED,
    randomized per-process by default). SimilarityAffinityCalculator relies on
    unique_affinities being index-aligned with its own separately-computed unique_smiles, so
    the order must be a documented, reproducible contract (sorted), not an implementation
    detail of set() that merely happens to agree within a single run.
    """
    smiles = [
        "CCC",
        "CCO",
        "CCN",
        "CCO",
        "CCC",
    ]  # duplicates, unsorted, unique count = 3
    library = Library(
        nsynthon_id=[str(i) for i in range(len(smiles))],
        smiles=smiles,
        relative_fraction=np.ones(len(smiles)),
    )
    sampler = TruncatedNormalSampler(loc=5.0, scale=2.0, seed=42)
    calculator = RandomAffinityCalculator(sampler=sampler, parallelism=1)

    _, unique_affinities = calculator.get_affinities(library)

    # use the calculator's own per-smiles hashing so this test doesn't duplicate/diverge
    # from _get_affinities's exact seed derivation
    expected_affinities = calculator._get_affinities(sorted(set(smiles)))

    np.testing.assert_array_equal(unique_affinities, expected_affinities)
