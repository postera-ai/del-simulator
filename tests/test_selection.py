import numpy as np
import pytest

from del_simulator.core import LibraryWithAffinities
from del_simulator.selection import SelectionExperiment


def test_binding_site_concentrations_mismatch_message():
    with pytest.raises(ValueError, match="binding site concentrations"):
        SelectionExperiment(
            binding_sites=["A", "B"],
            recovery_fractions=[0.5, 0.5],
            binding_site_concentrations_M=[1e-6],
            num_selection_rounds=3,
        )


def test_recovery_fractions_mismatch_message():
    """
    The recovery_fractions length check used to raise the same message as the
    binding_site_concentrations_M check, blaming the wrong parameter.
    """
    with pytest.raises(ValueError, match="recovery fractions"):
        SelectionExperiment(
            binding_sites=["A", "B"],
            recovery_fractions=[0.5],
            binding_site_concentrations_M=[1e-6, 1e-6],
            num_selection_rounds=3,
        )


def test_run_raises_when_binding_site_has_no_affinity_in_library():
    library = LibraryWithAffinities(
        nsynthon_id=["a"],
        smiles=["CCO"],
        relative_fraction=np.array([1.0]),
        Kd={"other": np.array([1e-6])},
    )
    experiment = SelectionExperiment(
        binding_sites=["target"],
        recovery_fractions=[1.0],
        binding_site_concentrations_M=[1e-6],
        num_selection_rounds=1,
    )

    with pytest.raises(ValueError, match="do not have corresponding affinities"):
        experiment.run(library, initial_library_concentration=1e-9)


def test_run_single_binding_site_enriches_tighter_binder():
    """
    Hand-computed expected values (verified independently via a standalone calculation, not by
    re-deriving the formula from the source): molecule "b" binds ~100x tighter (lower Kd) than
    "a", so despite equal input abundance it should end up proportionally enriched.
    """
    library = LibraryWithAffinities(
        nsynthon_id=["a", "b"],
        smiles=["CCO", "CCN"],
        relative_fraction=np.array([0.5, 0.5]),
        Kd={"target": np.array([1e-6, 1e-8])},
    )
    experiment = SelectionExperiment(
        binding_sites=["target"],
        recovery_fractions=[1.0],
        binding_site_concentrations_M=[1e-6],
        num_selection_rounds=1,
    )

    result = experiment.run(library, initial_library_concentration=1e-9)

    np.testing.assert_allclose(
        result.concentration, [1.25000000e-10, 2.47524752e-10], rtol=1e-6
    )
    # the tighter binder ("b") should end up enriched relative to its input abundance
    assert result.concentration[1] > result.concentration[0]


def test_run_two_binding_sites_multiple_rounds():
    """
    A second hand-computed case exercising the accumulation loop over multiple competing
    binding sites and num_selection_rounds > 1.
    """
    library = LibraryWithAffinities(
        nsynthon_id=["a", "b", "c"],
        smiles=["CCO", "CCN", "CCC"],
        relative_fraction=np.array([0.2, 0.3, 0.5]),
        Kd={
            "target1": np.array([1e-6, 1e-7, 1e-8]),
            "target2": np.array([1e-5, 1e-6, 1e-9]),
        },
    )
    experiment = SelectionExperiment(
        binding_sites=["target1", "target2"],
        recovery_fractions=[0.9, 0.5],
        binding_site_concentrations_M=[1e-6, 1e-7],
        num_selection_rounds=2,
    )

    result = experiment.run(library, initial_library_concentration=2e-9)

    np.testing.assert_allclose(
        result.concentration,
        [2.70298590e-11, 1.32947813e-10, 1.61712169e-10],
        rtol=1e-6,
    )
