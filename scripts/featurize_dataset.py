import argparse

from del_simulator.core import (
    FingerprintGeneratorMethod,
    MorganFingerPrintGeneratorParameters,
    FeaturizerRunsConfig,
)
from del_simulator.utils.utils import (
    load_config,
    FingerprintFeaturizer,
)

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import typing as T
from tqdm import tqdm
from pandarallel import pandarallel

import os
import time


def compute_and_save_fingerprints_in_chunks(
    smiles_df: pd.DataFrame,
    fp_output_path: str,
    fp_generator_method: FingerprintGeneratorMethod,
    fp_generator_method_parameters: T.Union[MorganFingerPrintGeneratorParameters],
    chunksize: int = 8192,
    parallelism: int = 4,
) -> None:
    pandarallel.initialize(progress_bar=False, nb_workers=parallelism)

    fpf = FingerprintFeaturizer(
        sparse=True,
        sanitize=True,
        to_numpy=True,
        fp_generator_method=fp_generator_method,
        fp_generator_method_parameters=fp_generator_method_parameters,
    )

    num_smiles = smiles_df.shape[0]
    num_chunks = (num_smiles - 1) // chunksize + 1

    logging.info(
        f"Computing fingerprints using {fp_generator_method} method. Breaking up the calculation into {num_chunks} chunks of {chunksize} smiles.",
    )
    # deleteing existing files
    if os.path.exists(fp_output_path):
        os.remove(fp_output_path)

    for chunk_idx in tqdm(range(num_chunks)):
        start_idx = chunk_idx * chunksize
        end_idx = (chunk_idx + 1) * chunksize
        subquery_df = smiles_df.iloc[start_idx:end_idx]

        fps = list(subquery_df.parallel_apply(lambda x: fpf.process_smiles(x)))

        with open(fp_output_path, "ba") as f:
            np.save(f, fps)

    logging.info(f"Saved fingerprints to {fp_output_path}")


def featurize(featurizer_config, dataset: pd.DataFrame) -> None:

    featurizer_parameters = featurizer_config.featurizer_parameters
    Path(featurizer_config.output_path).parent.mkdir(parents=True, exist_ok=True)

    if featurizer_config.featurizer_method == "fingerprint":

        compute_and_save_fingerprints_in_chunks(
            smiles_df=dataset[featurizer_parameters.smiles_column_name],
            fp_output_path=f"{featurizer_config.output_path}",
            chunksize=featurizer_parameters.chunksize,
            parallelism=featurizer_parameters.parallelism,
            fp_generator_method=FingerprintGeneratorMethod[
                featurizer_parameters.fp_generator_method
            ],
            fp_generator_method_parameters=featurizer_parameters.fp_generator_method_parameters,
        )

    else:

        raise NotImplementedError(
            f"Featurizer method {featurizer_config.featurizer_method} not implemented"
        )


if __name__ == "__main__":
    # this file splits the data, featurizes the train and test, and trains the model

    p = argparse.ArgumentParser()

    p.add_argument("config_file_path", type=str)
    p.add_argument("--config_attrs", nargs="+", default=[])

    args = p.parse_args()

    config = load_config(
        config_file_path=args.config_file_path, cli_config_dotlist=args.config_attrs
    )
    logging.basicConfig(level=config.loglevel.upper())

    featurizer_runs_config = load_config(
        config_file_path=args.config_file_path,
        cli_config_dotlist=args.config_attrs,
        config_schema=FeaturizerRunsConfig,
        config_section="featurizer_runs",
    )

    for featurizer_config in featurizer_runs_config.featurizer_runs:
        start_time = time.time()

        logging.info("Featurizing the Data")

        featurizer_parameters = featurizer_config.featurizer_parameters
        featurizer_method = featurizer_config.featurizer_method

        if featurizer_config.input_path is not None:
            filepath = os.path.join(featurizer_config.input_path)
            logging.info(f"Loading the smiles from {filepath} ")

            # load in the query library -->  assume we can fit the whole library in memory...
            dataset = pd.read_csv(
                filepath,
                usecols=[
                    featurizer_config.featurizer_parameters.smiles_column_name,
                ],
                sep="\t",  # FIXME expose as parameter
            )

            featurize(featurizer_config, dataset)

        logging.info("Featurization Complete!")
