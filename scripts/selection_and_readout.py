import os
from del_simulator.selection import SelectionExperiment
from del_simulator.readout import ReadoutExperiment
from del_simulator.core import (
    AmplificationReadoutConfig,
    AmplificationRunConfig,
    LibraryWithAffinities,
    SelectionExperimentConfig,
)
from dataclasses import asdict

import logging
import pandas as pd
from del_simulator.utils.utils import load_config, parse_arguments
from pathlib import Path

if __name__ == "__main__":

    args = parse_arguments()

    config = load_config(
        config_file_path=args.config_file_path, cli_config_dotlist=args.config_attrs
    )
    logging.basicConfig(level=config.loglevel.upper())

    selection_expt_config = load_config(
        config_file_path=args.config_file_path,
        cli_config_dotlist=args.config_attrs,
        config_schema=SelectionExperimentConfig,
        config_section="selection",
    )

    amplification_readout_config = load_config(
        config_file_path=args.config_file_path,
        cli_config_dotlist=args.config_attrs,
        config_schema=AmplificationReadoutConfig,
        config_section="amplification_readout",
    )

    #   selection_expt = SelectionExperiment(**selection_experiment_config)
    logging.info(
        f"Loading DEL library library from {config.selection.input_library_path}"
    )
    lib_df = pd.read_csv(
        selection_expt_config.input_library_path,
        nrows=selection_expt_config.num_query_molecules,
    )

    logging.info(f"Loaded the DEL library. It has {len(lib_df)} molecules")
    del_library = LibraryWithAffinities(
        nsynthon_id=lib_df["nsynthon_id"].to_list(),
        smiles=lib_df["smiles"].to_list(),
        relative_fraction=lib_df["relative_fraction"].to_numpy(),
    )
    num_library_members = len(set(del_library.nsynthon_id))

    for key, val in selection_expt_config.input_affinity_paths.items():
        logging.info(f"Loading affinity data from {val}")

        affinity_data = (
            pd.read_csv(
                val,
                usecols=["pKd"],
                nrows=selection_expt_config.num_query_molecules,
            )
            .to_numpy()
            .flatten()
        )

        del_library.pKd[key] = affinity_data
        del_library.Kd[key] = 10 ** (-affinity_data)

    for (
        condition_name,
        experiment_params,
    ) in selection_expt_config.selection_experiments.items():
        # FIXME add in checking for prior results in the output path, and if they exist to not rerun -- proceed to readout
        logging.info(
            f"Simulating a DEL affinity selection experiment:\n Condition {condition_name} : {experiment_params}"
        )

        selection_expt = SelectionExperiment(**asdict(experiment_params))

        initial_library_concentration = (
            selection_expt_config.initial_library_amount_mol
            / selection_expt_config.experiment_volume_L
        )

        logging.info(
            f"Initial Library Concentration {initial_library_concentration:.2e}M =>  {initial_library_concentration/num_library_members:.2e}M per member"
        )

        selection_results = selection_expt.run(
            library=del_library,
            initial_library_concentration=initial_library_concentration,
        )

        if config.selection.get("output_path") is not None:
            output_path = config.selection.output_path
            Path(output_path).mkdir(parents=True, exist_ok=True)
            selection_results_df = pd.DataFrame(asdict(selection_results))
            selection_results_df.to_csv(
                f"{output_path}/{condition_name}.csv", index=False
            )

        output_path_prefix = amplification_readout_config.output_path_prefix

        for run_name, run_config in amplification_readout_config.readout_runs.items():

            amplification_run = AmplificationRunConfig(**asdict(run_config))
            logging.info(
                f"Simulating a DEL PCR/NGS readout experiment: \n Condition {condition_name} : {asdict(run_config)}"
            )

            readout_expt = ReadoutExperiment(
                num_pcr_cycles=amplification_run.num_pcr_amplification_cycles,
                pcr_efficiency_parameters=amplification_run.pcr_efficiency_parameters,
                num_reads=amplification_run.num_reads,
                readout_seed=amplification_readout_config.readout_seed,
            )

            del_readout = readout_expt.run(
                selection_experiment_results=selection_results,
            )

            if run_config.output_path is not None:

                output_path = os.path.join(output_path_prefix, run_config.output_path)

                Path(output_path).mkdir(parents=True, exist_ok=True)

                readout_df = pd.DataFrame(asdict(del_readout))
                readout_df.to_csv(f"{output_path}/{condition_name}.csv", index=False)
