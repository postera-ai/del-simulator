import os

import pandas as pd

pd.options.mode.chained_assignment = None  # suppress pandas warning
import time

from pandarallel import pandarallel
from del_simulator.utils.utils import load_config, parse_arguments
from del_simulator.core import (
    TrainingAndInferenceConfig,
)
from del_simulator.ml_models import DELSimulatorRFModel, DELSimulatorChemPropModel

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
        config_schema=TrainingAndInferenceConfig,
        config_section="training_and_inference",
    )

    for run, run_config in runs_config.runs.items():

        start = time.time()

        # init up the model
        logging.info(f"Processing run: {run}")

        if run_config.ml_config.method == "random_forest":
            model_harness = DELSimulatorRFModel(
                input_data_path=run_config.input_data_path,
                model_output_path=run_config.model_output_path,
                metrics_output_path=run_config.metrics_output_path,
                method_parameters=run_config.ml_config.method_parameters,
            )
        elif run_config.ml_config.method == "chemprop":
            model_harness = DELSimulatorChemPropModel(
                input_data_path=run_config.input_data_path,
                model_output_path=run_config.model_output_path,
                metrics_output_path=run_config.metrics_output_path,
                method_parameters=run_config.ml_config.method_parameters,
            )
        else:
            raise ValueError(f"ML method {run_config.ml_config.method} not implemented")

        # load the data
        data_df = model_harness.load_train_data()

        enrichment = model_harness.calculate_enrichment(
            data_df=data_df,
            enrichment_method=run_config.enrichment_method,
            enrichment_method_params=run_config.enrichment_method_params,
        )

        data_df["class"] = (enrichment > run_config.enrichment_threshold).astype(int)
        logging.info(f"Positive class fraction: {data_df['class'].mean()}")
        model = model_harness.train(data_df=data_df, n_cv_splits=run_config.n_cv_splits)

        if run_config.inference is not None:

            y_inference_preds = model_harness.run_inference(
                model=model,
                smiles_path=run_config.inference.smiles_path,
                featurized_data_path=run_config.inference.featurized_data_path,
                affinity_path=run_config.inference.affinity_path,
                affinity_column=run_config.inference.affinity_column,
                affinity_threshold=run_config.inference.affinity_threshold,
                sample_size=run_config.inference.sample_size,
                separator=run_config.inference.separator,
            )

            # save the predictions
            if run_config.inference.output_predictions_path is not None:
                logging.info(
                    f"Saving inference predictions to {run_config.inference.output_predictions_path}"
                )
                # make the folder if it doesn't exist
                os.makedirs(
                    os.path.dirname(run_config.inference.output_predictions_path),
                    exist_ok=True,
                )

                pd.DataFrame(y_inference_preds, columns=["p_1"]).to_csv(
                    run_config.inference.output_predictions_path, index=False
                )  # pickle vs csv?
            #  with open(config.inference.output_predictions_path, "wb") as f:
            #    pickle.dump(y_inference_preds, f)

        end = time.time()
