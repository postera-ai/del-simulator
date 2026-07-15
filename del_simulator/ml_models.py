import logging
import pandas as pd
import typing as T
import torch

import numpy as np
import os
import time
import json
import pickle
import chemprop

from lightning.fabric.utilities.data import AttributeDict


from chemprop import data, featurizers
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from chemprop.nn.agg import AttentiveAggregation

from del_simulator.utils.utils import get_del_enrichment
from sklearn.model_selection import StratifiedKFold

from del_simulator.core import RatioTestProcessorParameters, ZScoreProcessorParameters

from del_simulator.utils.utils import load_npy_multiarray, get_del_enrichment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from del_simulator.core import (
    RatioTestProcessorParameters,
    ZScoreProcessorParameters,
    RandomForestClassifierParameters,
    ChemPropGNNParameters,
)
from del_simulator.utils.metrics import get_inference_metrics

torch.serialization.add_safe_globals([AttributeDict])
torch.serialization.add_safe_globals([chemprop.nn.utils.Activation])
pd.options.mode.chained_assignment = None  # suppress pandas warning


class DELSimulatorMLModelHarness:
    """
    Base class for DEL Simulator machine learning models.
    This class provides a common interface for training and inference.
    """

    def __init__(
        self,
        input_data_path: str,
        model_output_path: str,
        metrics_output_path: str,
    ):
        self.input_data_path = input_data_path
        self.model_output_path = model_output_path
        self.metrics_output_path = metrics_output_path

    def load_inference_data():
        raise NotImplementedError("Subclasses should implement this method.")

    def load_train_data(self):
        with open(self.input_data_path, "rb") as f:
            print(f"Loading training data from {self.input_data_path}")
            data_df = pickle.load(f)

        return data_df

    def calculate_enrichment(
        self, data_df, enrichment_method, enrichment_method_params
    ):
        # calulate enrichment
        if enrichment_method == "scaled_poisson_ratio_test":
            method_params = RatioTestProcessorParameters(
                total_number_of_reads_in_sample_1=data_df["target_counts"].sum(),
                total_number_of_reads_in_sample_2=data_df["ntc_counts"].sum(),
                library_size=enrichment_method_params.library_diversity,
                enrichment_threshold=enrichment_method_params.poisson_ratio,
            )
            input_columns = ["target_counts", "ntc_counts"]
        elif enrichment_method == "normalized_zscore":
            method_params = ZScoreProcessorParameters(
                library_size=enrichment_method_params.library_diversity,
                total_number_of_reads=data_df["target_counts"].sum(),
                unique_mols_observed=data_df["smiles"].nunique(),
            )
            input_columns = ["target_counts"]
        elif enrichment_method == "count_ratio":
            method_params = None
            input_columns = ["target_counts", "ntc_counts"]

        else:
            raise ValueError("Invalid enrichment method")

        enrichment = data_df.apply(
            lambda x: get_del_enrichment(
                method=enrichment_method,
                input_data=x[input_columns],
                method_params=method_params,
            ),
            axis=1,
        )
        return enrichment

    def save_model(self, model, model_output_path, suffix: str = "0"):
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        # if model exposes a save method, use it  otherwise pickle it and save it
        if hasattr(model, "save"):
            model.save(
                os.path.join(
                    model_output_path,
                    f"model_{suffix}.pt",
                )
            )
        else:
            with open(
                os.path.join(
                    model_output_path,
                    f"model_{suffix}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(model, f)

    def compute_and_save_metrics(
        self, y_pred, y_val, metadata: T.Dict[str, T.Any] = None
    ):
        metrics = get_inference_metrics(y_pred, y_val)

        metrics.update(**metadata)

        os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
        with open(self.metrics_output_path, "a") as f:
            json_string = json.dumps(metrics)
            f.write(json_string)
            f.write("\n")

    def train(self, X, y):
        """
        Train the model using the provided features and labels.
        :param X: Features for training.
        :param y: Labels for training.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _predict(self, X):
        """
        Predict labels for the provided features.
        :param X: Features for prediction.
        :return: Predicted labels.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def run_inference(self):
        """
        Predict labels for the provided features.
        :param X: Features for prediction.
        :return: Predicted labels.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class DELSimulatorRFModel(DELSimulatorMLModelHarness):
    """
    Random Forest model for DEL Simulator.
    This class implements the Random Forest algorithm for training and inference.
    """

    def __init__(
        self,
        input_data_path: str,
        model_output_path: str,
        metrics_output_path: str,
        method_parameters: T.Optional[RandomForestClassifierParameters] = None,
    ):
        super().__init__(input_data_path, model_output_path, metrics_output_path)
        self.params = method_parameters or RandomForestClassifierParameters()

    def load_inference_data(
        self,
        featurized_data_path: str,
        smiles_path: T.Optional[str] = None,
        affinity_path: T.Optional[str] = None,
        affinity_threshold: T.Optional[float] = None,
        affinity_column: T.Optional[str] = "pKd",
        sample_size: T.Optional[int] = None,
    ):
        """
        Load inference data from the specified path.
        :param input_data_path: Path to the input data file.
        :return: DataFrame containing the loaded data.
        """

        if featurized_data_path is not None:
            logging.info(f"Loading featurized data from {featurized_data_path}")
            X_screening_fps = load_npy_multiarray(
                featurized_data_path
            )  # Ideally, this should load the same way training data is loaded
            y_screening_affinities_label = None
        else:
            # FIXME add ability to featurize the smiles if the featurized data path is not provided
            raise ValueError("Featurized data path must be provided for RF model")

        if affinity_path is not None:
            y_screening_affinities = pd.read_csv(affinity_path)[affinity_column]
            y_screening_affinities_label = np.array(
                (y_screening_affinities > affinity_threshold).astype(int)
            )

        if sample_size is not None:
            logging.info(f"Sampling {sample_size} samples from the screening data")
            sample_ids = np.random.default_rng(42).choice(
                np.arange(len(X_screening_fps)), sample_size, replace=False
            )

            X_screening_fps = X_screening_fps[sample_ids]

            if y_screening_affinities_label is not None:
                y_screening_affinities_label = y_screening_affinities_label[sample_ids]

        return X_screening_fps, y_screening_affinities_label

    def train(self, data_df, n_cv_splits=5):
        logging.info(f"Training RF model using data from {self.input_data_path}")
        colname = "sparse_fingerprint"
        if colname not in data_df.columns:
            colname = "fingerprint"

        X = np.array([sparse_matrix.toarray() for sparse_matrix in data_df[colname]])[
            :, 0, :
        ]
        y = np.array(data_df["class"])

        # logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logging.info(
            f"Number of positive samples: {y.sum()}; Number of negative samples: {len(y) - y.sum()}",
        )

        skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            if fold > 0:
                break

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = RandomForestClassifier(
                n_estimators=self.params.n_estimators,
                # FIXME other parameters should be added here
                random_state=fold,
                n_jobs=-1,
                class_weight="balanced",
            )

            model.fit(X_train, y_train)

            logging.info(f"Fold {fold} - Training completed.")

            if self.model_output_path:
                self.save_model(
                    model,
                    self.model_output_path,
                    suffix=str(fold),
                )

            logging.info(f"Compyuting and saving metrics to {self.metrics_output_path}")
            y_pred = self._predict(model, X_val, batch_sparse_inference=False)
            self.compute_and_save_metrics(
                y_pred=y_pred,
                y_val=y_val,
                metadata={
                    "model_type": "random_forest",
                    "fold": str(fold),
                    "prediction_type": "validation",
                },
            )

        return model

    def _batch_inference_sparse_fps(
        self, model, X, batch_size: int = 50000
    ):  # fixme explose batch size?
        X = np.array(X)
        y_preds = []
        for i in range(0, len(X), batch_size):
            X_batch_sparse = X[i : i + batch_size]
            X_batch = np.array(
                [sparse_matrix.toarray() for sparse_matrix in X_batch_sparse]
            )[:, 0, :]
            y_pred = model.predict_proba(X_batch)
            y_preds.append(y_pred)

        return np.concatenate(y_preds)

    def _predict(self, model, X, batch_sparse_inference=False):
        if batch_sparse_inference:
            y_pred_proba = self._batch_inference_sparse_fps(model, X)
        else:
            y_pred_proba = model.predict_proba(X)

        try:
            y_pred_proba = y_pred_proba[:, 1]
        except:
            y_pred_proba = y_pred_proba

        return y_pred_proba

    def run_inference(
        self,
        model,
        featurized_data_path: T.Optional[str] = None,
        smiles_path: T.Optional[str] = None,
        affinity_path: T.Optional[str] = None,
        affinity_threshold: T.Optional[float] = None,
        affinity_column: T.Optional[str] = "pKd",
        sample_size: T.Optional[int] = None,
        separator: str = ",",
    ) -> np.ndarray:
        """
        Run inference on the provided features.
        :param X: Features for inference.
        :param batch_sparse_inference: Whether to use batch sparse inference.
        :param model: Pre-trained model for inference.
        :return: Predicted labels.
        """
        # separator is unused here -- RF reads pre-featurized fingerprints via
        # featurized_data_path, never smiles_path directly. Accepted so callers can pass the
        # same kwargs to either model harness.
        logging.info(f"Running inference using RF model")

        X_screening_fps, y_screening_affinities_label = self.load_inference_data(
            featurized_data_path=featurized_data_path,
            smiles_path=smiles_path,
            affinity_path=affinity_path,
            affinity_threshold=affinity_threshold,
            affinity_column=affinity_column,
            sample_size=sample_size,
        )

        y_pred = self._predict(model, X_screening_fps, batch_sparse_inference=True)

        if y_screening_affinities_label is not None:
            self.compute_and_save_metrics(
                y_pred=y_pred,
                y_val=y_screening_affinities_label,
                metadata={
                    "model_type": "random_forest",
                    "prediction_type": "screening",
                },
            )

        return y_pred


class DELSimulatorChemPropModel(DELSimulatorMLModelHarness):
    """
    ChemProp GNN model for DEL Simulator.
    """

    def __init__(
        self,
        input_data_path: str,
        model_output_path: str,
        metrics_output_path: str,
        method_parameters: T.Optional[ChemPropGNNParameters] = None,
    ):
        super().__init__(input_data_path, model_output_path, metrics_output_path)
        self.params = method_parameters or ChemPropGNNParameters(max_epochs=10)

    def load_inference_data(
        self,
        featurized_data_path: T.Optional[str] = None,
        smiles_path: T.Optional[str] = None,
        affinity_path: T.Optional[str] = None,
        affinity_threshold: T.Optional[float] = None,
        affinity_column: T.Optional[str] = "pKd",
        sample_size: T.Optional[int] = None,
        separator: str = ",",
    ):
        """
        Load inference data from the specified path.
        If featurized_data_path is provided and exists, loads cached dataloader and labels.
        If featurized_data_path is provided but does not exist, builds data and saves to that path.
        :param featurized_data_path: Path to a pickle file for caching the dataloader and labels.
        :return: Tuple of (screening_data_loader, labels array).
        """

        cache_key = {
            "smiles_path": smiles_path,
            "affinity_path": affinity_path,
            "affinity_threshold": affinity_threshold,
            "affinity_column": affinity_column,
            "sample_size": sample_size,
            "separator": separator,
        }

        if featurized_data_path is not None and os.path.exists(featurized_data_path):
            with open(featurized_data_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("cache_key") == cache_key:
                logging.info(
                    f"Loading cached inference data from {featurized_data_path}"
                )
                return cached["data_loader"], cached["labels"]
            logging.info(
                f"Cached inference data at {featurized_data_path} was built with "
                "different parameters -- recomputing and overwriting the cache."
            )

        screening_smiles = pd.read_csv(smiles_path, sep=separator)

        if affinity_path is not None:
            screening_affinities = pd.read_csv(affinity_path)
            screening_affinities["class"] = np.array(
                (screening_affinities[affinity_column] > affinity_threshold).astype(int)
            )

            screening_df = pd.concat([screening_smiles, screening_affinities], axis=1)
        else:
            screening_df = screening_smiles

        if sample_size is not None and sample_size < len(screening_df):
            screening_df = screening_df.sample(
                sample_size, random_state=42
            )  # FIXME init this

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        screening_data = [
            data.MoleculeDatapoint.from_smi(smi) for smi in screening_df["smiles"]
        ]
        screening_dset = data.MoleculeDataset(screening_data, featurizer=featurizer)

        screening_data_loader = data.build_dataloader(
            screening_dset,
            shuffle=False,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )

        labels = (
            np.array(screening_df["class"]) if "class" in screening_df.columns else None
        )

        if featurized_data_path is not None:
            logging.info(f"Saving inference data to {featurized_data_path}")
            os.makedirs(os.path.dirname(featurized_data_path), exist_ok=True)
            with open(featurized_data_path, "wb") as f:
                pickle.dump(
                    {
                        "data_loader": screening_data_loader,
                        "labels": labels,
                        "cache_key": cache_key,
                    },
                    f,
                )

        return screening_data_loader, labels

    def train(self, data_df, n_cv_splits=5):
        smiles = data_df.loc[:, "smiles"].values
        labels = np.expand_dims(data_df.loc[:, "class"].values, axis=-1)
        all_data = [
            data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, labels)
        ]

        # logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logging.info(
            f"Number of positive samples: {labels.sum()}; Number of negative samples: {len(labels) -labels.sum()}",
        )

        skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

        mols = [d.mol for d in all_data]
        for fold, (train_indices, val_indices) in enumerate(skf.split(mols, labels)):
            # FIXME -- expose the ability to skip folds logic to config
            if (
                fold > 0
            ):  # only doing a single fold because there are multiple selection replicas
                break

            train_data, val_data, _ = data.split_data_by_indices(
                all_data, [train_indices], [val_indices], None
            )

            featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

            train_dset = data.MoleculeDataset(train_data[0], featurizer)
            val_dset = data.MoleculeDataset(val_data[0], featurizer)

            train_loader = data.build_dataloader(
                train_dset,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers,
                shuffle=True,
                persistent_workers=True,
            )
            val_loader = data.build_dataloader(
                val_dset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                persistent_workers=True,
            )

            mpnn = models.MPNN(
                message_passing=nn.BondMessagePassing(),  # pass in other parameters from the config?
                agg=AttentiveAggregation(output_size=300),
                predictor=nn.BinaryClassificationFFN(n_tasks=1),
                batch_norm=True,
            )

            saved_model_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1)

            trainer = pl.Trainer(
                logger=False,
                enable_progress_bar=True,
                accelerator="auto",
                devices="auto",
                max_epochs=self.params.max_epochs,
                enable_checkpointing=True,
                callbacks=[saved_model_callback],
            )

            logging.info(f"Number of GPUs: {torch.cuda.device_count()}")

            trainer.fit(mpnn, train_loader, val_loader)
            best_model_path = saved_model_callback.best_model_path
            model = mpnn.load_from_checkpoint(
                best_model_path, weights_only=False
            )  # fixme in torch 2.6 this changed to default True and borked downstream

        return model

    def _predict(self, model, data_loader):
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=True,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="1",
            )

            preds = trainer.predict(model, data_loader)

        preds = torch.cat(preds, dim=0).detach().cpu().numpy()[:, 0]

        return preds

    def run_inference(
        self,
        model,
        featurized_data_path: T.Optional[str] = None,
        smiles_path: T.Optional[str] = None,
        affinity_path: T.Optional[str] = None,
        affinity_threshold: T.Optional[float] = None,
        affinity_column: T.Optional[str] = "pKd",
        sample_size: T.Optional[int] = None,
        separator: str = ",",
    ):
        """
        Run inference on the provided features.
        :param X: Features for inference.
        :param batch_sparse_inference: Whether to use batch sparse inference.
        :param model: Pre-trained model for inference.
        :return: Predicted labels.
        """
        logging.info(f"Running inference using Chemprop")

        start = time.time()

        logging.info(
            f"Loading inference data from {featurized_data_path} and {smiles_path} -- subsetting to {sample_size} samples"
        )

        data_loader, y_screening_affinities_label = self.load_inference_data(
            featurized_data_path=featurized_data_path,
            smiles_path=smiles_path,
            affinity_path=affinity_path,
            affinity_threshold=affinity_threshold,
            affinity_column=affinity_column,
            sample_size=sample_size,
            separator=separator,
        )

        logging.info(f"Data loaded in {time.time()-start}s")

        y_pred = self._predict(model, data_loader)

        end = time.time()

        logging.info(f"Inference Complete in {end-start}s.")

        if y_screening_affinities_label is not None:
            logging.info(f"Computing and saving metrics to {self.metrics_output_path}")
            self.compute_and_save_metrics(
                y_pred=y_pred,
                y_val=y_screening_affinities_label,
                metadata={
                    "model_type": "chemprop",
                    "prediction_type": "screening",
                    "input_data": self.input_data_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                },
            )

        return y_pred
