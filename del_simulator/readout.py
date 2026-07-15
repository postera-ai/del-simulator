import numpy as np
from del_simulator.core import (
    ReadoutExperimentResults,
    SelectionExperimentResults,
    PCRGaussianEfficiencyParameters,
)
from collections import Counter
from scipy import stats
import logging


class ReadoutExperiment:
    """
    class that models the running of the readout experiment
    """

    def __init__(
        self,
        num_reads: int,
        readout_seed: int,
        num_pcr_cycles: int = 0,
        pcr_efficiency_parameters: PCRGaussianEfficiencyParameters = None,
    ):
        if num_pcr_cycles > 0 and pcr_efficiency_parameters is None:
            raise ValueError(
                "pcr_efficiency_parameters must be provided when num_pcr_cycles > 0"
            )

        self.num_reads = num_reads
        self.num_pcr_cycles = num_pcr_cycles
        self.readout_seed = readout_seed
        self.pcr_efficiency_parameters = pcr_efficiency_parameters
        if self.pcr_efficiency_parameters is not None:
            self.lower_bound, self.upper_bound = (
                0 - self.pcr_efficiency_parameters.loc
            ) / self.pcr_efficiency_parameters.scale, (
                1.0 - self.pcr_efficiency_parameters.loc
            ) / self.pcr_efficiency_parameters.scale

    def run(
        self,
        selection_experiment_results: SelectionExperimentResults,
    ) -> ReadoutExperimentResults:

        # simulate the PCR enrichment experiment
        num_mols = len(selection_experiment_results.concentration)

        pcr_efficiencies = np.ones(num_mols)
        if self.num_pcr_cycles == 0:
            logging.info(
                "Number of PCR cycles is 0, disabling noise from PCR sampling."
            )
        else:

            np.random.seed(self.pcr_efficiency_parameters.seed)

            for pcr_cycle in range(0, self.num_pcr_cycles):
                pcr_efficiencies = pcr_efficiencies * (
                    np.ones(num_mols)
                    + stats.truncnorm.rvs(
                        a=self.lower_bound,
                        b=self.upper_bound,
                        loc=self.pcr_efficiency_parameters.loc,
                        scale=self.pcr_efficiency_parameters.scale,
                        size=num_mols,
                    )
                )

        p = (
            pcr_efficiencies  # this is noise.
            * selection_experiment_results.concentration
            / np.sum(pcr_efficiencies * selection_experiment_results.concentration)
        )  # the probability of extracting a given molecule
        np.random.seed(self.readout_seed)
        readouts = np.random.choice(
            selection_experiment_results.nsynthon_id,
            size=self.num_reads,
            replace=True,
            p=p,
        )
        # group readouts by count
        counts = Counter(readouts)  # speed? Mem?

        nsynthon_id, count = zip(*counts.items())
        return ReadoutExperimentResults(nsynthon_id=nsynthon_id, count=count)
