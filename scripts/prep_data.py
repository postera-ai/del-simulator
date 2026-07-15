import os
import pandas as pd

import numpy as np

from tqdm import tqdm
from scipy import sparse
from del_simulator.utils.aggregation_utils import mean_aggregation, get_merged_data
from del_simulator.utils.utils import FingerprintFeaturizer
from del_simulator.core import DataPrepConfig, FingerprintGeneratorMethod
from pandarallel import pandarallel
from del_simulator.utils.utils import load_config, parse_arguments

pd.options.mode.chained_assignment = None  # suppress pandas warning
import logging

pandarallel.initialize(
    nb_workers=os.cpu_count(), progress_bar=False, verbose=2, use_memory_fs=False
)


if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(
        config_file_path=args.config_file_path, cli_config_dotlist=args.config_attrs
    )
    logging.basicConfig(level=config.loglevel.upper())

    runs_config = load_config(
        config_file_path=args.config_file_path,
        cli_config_dotlist=args.config_attrs,
        config_schema=DataPrepConfig,
        config_section="data_prep",
    )

    for run, config in runs_config.runs.items():

        logging.info(f"Processing run: {run}")
        assert (
            config.num_bbs_to_aggregate is None
            or config.num_bbs_to_aggregate <= config.bbs_per_nsynthon
        ), f"Number of building blocks to aggregate ({config.num_bbs_to_aggregate}) must be less than the number of building blocks per nsynthon ({config.bbs_per_nsynthon })"

        logging.info(
            f"Preparing data for {config.target_data_path}, {config.ntc_data_path}"
        )

        intended_product_df = pd.read_csv(config.intended_product_path)
        target_df = pd.read_csv(config.target_data_path)
        ntc_df = pd.read_csv(config.ntc_data_path)

        data_df = get_merged_data(
            target_df=target_df,
            ntc_df=ntc_df,
            intended_product_df=intended_product_df,
            bbs_per_nsynthon=config.bbs_per_nsynthon,
        )

        fpf = FingerprintFeaturizer(
            sparse=False,
            sanitize=True,
            to_numpy=True,
            fp_generator_method=FingerprintGeneratorMethod[config.fp_generator_method],
            fp_generator_method_parameters=config.fp_generator_method_parameters,
        )

        data_df["fingerprint"] = list(
            data_df["smiles"].parallel_apply(lambda x: fpf.process_smiles(x))
        )

        data_df["sparse_fingerprint"] = data_df["fingerprint"].parallel_apply(
            lambda x: sparse.csr_matrix(x)
        )

        out_df_path = os.path.join(
            config.output_path,
            f"processed_data_df.pkl",
        )
        os.makedirs(os.path.dirname(out_df_path), exist_ok=True)
        logging.info(f"Saving processed data to {out_df_path}")
        data_df.to_pickle(out_df_path)

        if config.num_bbs_to_aggregate is not None:

            logging.info(
                f"Performing {config.num_bbs_to_aggregate}-synthon aggregation!"
            )
            # aggregate nsynthons
            nsynthon_agg_df = mean_aggregation(
                data_df, config.bbs_per_nsynthon, config.num_bbs_to_aggregate
            )

            nsynthon_agg_df["fingerprint"] = nsynthon_agg_df[
                "fingerprint"
            ].parallel_apply(lambda x: sparse.csr_matrix(x))

            out_nsynthon_path = os.path.join(
                config.output_path,
                f"processed_data_nsynthon_agg_{config.num_bbs_to_aggregate}.pkl",
            )
            os.makedirs(os.path.dirname(out_nsynthon_path), exist_ok=True)
            logging.info(f"Saving processed data to {out_nsynthon_path}")

            nsynthon_agg_df.to_pickle(out_nsynthon_path)
