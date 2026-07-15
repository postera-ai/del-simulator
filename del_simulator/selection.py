import numpy as np
from del_simulator.core import (
    Library,
    LibraryWithAffinities,
    SelectionExperimentResults,
)
import typing as T


class SelectionExperiment:
    """
    class that models the running of the DEL selection experiment
    """

    def __init__(
        self,
        binding_sites: T.List[str],
        recovery_fractions: np.ndarray[float],
        binding_site_concentrations_M: np.ndarray[float],
        num_selection_rounds: int,
    ):
        self.num_selection_rounds = num_selection_rounds
        self.binding_sites = binding_sites
        self.recovery_fractions = recovery_fractions
        self.binding_site_concentrations_M = binding_site_concentrations_M

        if len(self.binding_site_concentrations_M) != len(self.binding_sites):
            raise ValueError(
                "number of binding sites and binding site concentrations must match"
            )

        if len(self.recovery_fractions) != len(self.binding_sites):
            raise ValueError(
                "number of binding sites and recovery fractions must match"
            )

    def run(
        self,
        library: LibraryWithAffinities,
        initial_library_concentration: float,
    ) -> SelectionExperimentResults:
        """run the selection experiment"""

        initial_ligand_concentration = initial_library_concentration / len(
            set(library.nsynthon_id)
        )

        if len(set(self.binding_sites) - set(library.Kd.keys())) > 0:
            raise ValueError(
                "There are binding sites defined in experiment that do not have corresponding affinities defined in the library"
            )

        # compute the Q term
        # FIXME is this numerically stable?
        q_numerator = 1.0
        for idx, binding_site in enumerate(self.binding_sites):
            q_numerator += (
                self.binding_site_concentrations_M[idx] / library.Kd[binding_site]
            )

        q = np.zeros(len(library.nsynthon_id))
        for idx, binding_site in enumerate(self.binding_sites):
            q_term = self.recovery_fractions[idx] * (
                self.binding_site_concentrations_M[idx] / library.Kd[binding_site]
            )

            q += q_term

        final_concentrations = (
            initial_ligand_concentration
            * library.relative_fraction
            * (q / q_numerator) ** self.num_selection_rounds
        )

        selection_results = SelectionExperimentResults(
            library.smiles, library.nsynthon_id, final_concentrations
        )

        return selection_results
