import numpy as np
import pytest

from del_simulator.core import (
    SelectionExperimentResults,
    PCRGaussianEfficiencyParameters,
)
from del_simulator.readout import ReadoutExperiment


def test_run_no_pcr_noise_read_counts_proportional_to_concentration():
    selection_results = SelectionExperimentResults(
        smiles=["CCO", "CCN", "CCC"],
        nsynthon_id=["a", "b", "c"],
        concentration=np.array([0.1, 0.3, 0.6]),
    )
    experiment = ReadoutExperiment(num_reads=200_000, readout_seed=42, num_pcr_cycles=0)

    result = experiment.run(selection_results)
    counts = dict(zip(result.nsynthon_id, result.count))
    total = sum(counts.values())

    assert total == 200_000
    assert counts["a"] / total == pytest.approx(0.1, abs=0.01)
    assert counts["b"] / total == pytest.approx(0.3, abs=0.01)
    assert counts["c"] / total == pytest.approx(0.6, abs=0.01)


def test_run_is_deterministic_given_seed():
    selection_results = SelectionExperimentResults(
        smiles=["CCO", "CCN"],
        nsynthon_id=["a", "b"],
        concentration=np.array([0.4, 0.6]),
    )
    experiment = ReadoutExperiment(num_reads=1000, readout_seed=7, num_pcr_cycles=0)

    result1 = experiment.run(selection_results)
    result2 = experiment.run(selection_results)

    assert dict(zip(result1.nsynthon_id, result1.count)) == dict(
        zip(result2.nsynthon_id, result2.count)
    )


def test_run_with_pcr_noise_runs_and_is_deterministic():
    """
    ReadoutExperiment used to access pcr_efficiency_parameters.mean/.stdev, but
    PCRGaussianEfficiencyParameters only defines .loc/.scale -- this path would raise
    AttributeError for any num_pcr_cycles > 0, the entire point of this parameter.
    """
    selection_results = SelectionExperimentResults(
        smiles=["CCO", "CCN", "CCC"],
        nsynthon_id=["a", "b", "c"],
        concentration=np.array([0.2, 0.3, 0.5]),
    )
    pcr_params = PCRGaussianEfficiencyParameters(loc=0.9, scale=0.05, seed=42)
    experiment = ReadoutExperiment(
        num_reads=5000,
        readout_seed=1,
        num_pcr_cycles=3,
        pcr_efficiency_parameters=pcr_params,
    )

    result1 = experiment.run(selection_results)
    result2 = experiment.run(selection_results)

    assert sum(result1.count) == 5000
    assert dict(zip(result1.nsynthon_id, result1.count)) == dict(
        zip(result2.nsynthon_id, result2.count)
    )


def test_missing_pcr_efficiency_parameters_raises_immediately():
    """
    Omitting pcr_efficiency_parameters while num_pcr_cycles > 0 used to construct
    successfully and only raise AttributeError deep inside run() (on
    self.pcr_efficiency_parameters.seed). Assert it now fails fast, in __init__.
    """
    with pytest.raises(ValueError):
        ReadoutExperiment(
            num_reads=100,
            readout_seed=1,
            num_pcr_cycles=3,
            pcr_efficiency_parameters=None,
        )
