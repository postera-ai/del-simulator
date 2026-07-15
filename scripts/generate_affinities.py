from del_simulator.core import (
    Library,
    AffinityGenConfig,
    AffinityGenRandomMethodParameters,
    AffinityGenSimilarityMethodParameters,
)

from del_simulator.affinity_calculator import (
    RandomAffinityCalculator,
    SimilarityAffinityCalculator,
    FingerprintGeneratorMethod,
    TruncatedNormalSampler,
    LogNormalSampler,
)

from dataclasses import dataclass, asdict

import logging
import pandas as pd
import numpy as np
import time
from del_simulator.utils.utils import load_config, parse_arguments
from pathlib import Path
import typing as T


def calculate_affinity(
    query_lib: Library,
    method: T.Literal["random", "similarity"] = "random",
    method_parameters: T.Union[
        AffinityGenSimilarityMethodParameters, AffinityGenRandomMethodParameters
    ] = None,
) -> np.array:

    params = method_parameters

    if (
        params.sampler == "truncated_normal"
    ):  # FIXME this is a bit convoluted -- the config objects are dataclasses; but the sampelrs expect kwargs...
        sampler = TruncatedNormalSampler(**asdict(params.sampler_params))
    elif params.sampler == "lognormal":
        sampler = LogNormalSampler(**asdict(params.sampler_params))
    else:
        raise NotImplementedError(f"Distribution {params.sampler} not implemented")

    if method == "similarity":
        logging.info(
            f"Loading reference affinities from {params.reference_affinity_path}"
        )
        # load the reference
        reference_data = pd.read_csv(
            params.reference_affinity_path,
            usecols=[
                params.reference_smiles_field_name,
                params.reference_affinity_field_name,
            ],
        )

        affinity_calculator = SimilarityAffinityCalculator(
            reference_affinities=list(
                reference_data[params.reference_affinity_field_name]
            ),
            reference_smiles=list(reference_data[params.reference_smiles_field_name]),
            sampler=sampler,
            kernel_params=params.kernel_params,
            tversky_params=params.tversky_params,
            fp_generator_method=FingerprintGeneratorMethod[
                params.fp_generator_method
            ],  # this is bogus -- when I replace omegaconf validation wiht pydantic, this will be improved
            fp_generator_method_parameters=params.fp_generator_method_parameters,
            parallelism=params.parallelism,
            progress_bar=params.progress_bar,
        )

        affinities, _ = affinity_calculator.get_affinities(
            query_library=query_lib,
            chunksize=params.chunksize,
        )

    elif method == "random":

        affinity_calculator = RandomAffinityCalculator(
            sampler=sampler,
            parallelism=params.parallelism,
        )

        affinities, _ = affinity_calculator.get_affinities(
            query_library=query_lib, chunksize=params.chunksize
        )
    else:
        raise NotImplementedError

    return affinities


if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(
        config_file_path=args.config_file_path, cli_config_dotlist=args.config_attrs
    )
    logging.basicConfig(level=config.loglevel.upper())

    affinity_gen_config = load_config(
        config_file_path=args.config_file_path,
        cli_config_dotlist=args.config_attrs,
        config_schema=AffinityGenConfig,
        config_section="affinity_generation",
    )

    logging.info(
        "Loading DEL library from %s (max_rows: %s)",
        affinity_gen_config.input_library_path,
        affinity_gen_config.num_query_molecules,
    )
    start_time = time.time()

    lib_df = pd.read_csv(
        affinity_gen_config.input_library_path,
        nrows=affinity_gen_config.num_query_molecules,
        sep=affinity_gen_config.input_library_sep,
    )

    if "relative_fraction" not in lib_df.columns:
        lib_df["relative_fraction"] = 1.0
        logging.warning(
            "relative_fraction column not found in the input library. Setting all fractions to 1.0"
        )
    if "nsynthon_id" not in lib_df.columns:
        lib_df["nsynthon_id"] = "0_0"
        logging.warning(
            "nsynthon_id column not found in the input library. Setting all nsynthon_ids to 0_0"
        )
    query_lib = Library(
        **lib_df[["nsynthon_id", "smiles", "relative_fraction"]].to_dict(orient="list")
    )

    for (
        condition_name,
        run_params,
    ) in affinity_gen_config.runs.items():

        logging.info(
            f"Calculating {condition_name} affinities for {len(query_lib.nsynthon_id)} mols using method {run_params.method}"
        )

        affinity_vec = calculate_affinity(
            query_lib=query_lib,
            method=run_params.method,
            method_parameters=run_params.method_parameters,
        )

        output_path = affinity_gen_config.output_path
        Path(output_path).mkdir(parents=True, exist_ok=True)

        logging.info(
            "Finished calculating affinities in %s s. Output saved to %s",
            round(time.time() - start_time, 2),
            f"{output_path}/{condition_name}.csv",
        )
        output_df = pd.DataFrame({"pKd": affinity_vec})  # RO

        output_df.round({"pKd": 6}).to_csv(
            f"{output_path}/{condition_name}.csv", index=False
        )
