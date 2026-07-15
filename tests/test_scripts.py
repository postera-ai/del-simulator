import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.sparse import csr_matrix

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
RESOURCES_DIR = Path(__file__).parent / "resources"


def _run_script(script_name, config_path):
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), str(config_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _write_config(tmp_path, config_dict):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_dict, f)
    return config_path


def test_generate_library_smoke(tmp_path):
    output_path = tmp_path / "out"
    config_path = _write_config(
        tmp_path,
        {
            "loglevel": "INFO",
            "library_generation": {
                "building_block_path": str(RESOURCES_DIR / "bbs0.json"),
                "reaction_scheme_path": str(RESOURCES_DIR / "dag0.json"),
                "output_path": str(output_path),
                "num_cpu": 1,
                "chunksize": 10,
                "return_all_products": True,
            },
        },
    )

    result = _run_script("generate_library.py", config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    library_csv = output_path / "library.csv"
    assert library_csv.exists()
    assert len(pd.read_csv(library_csv)) > 0


def test_generate_affinities_smoke(tmp_path):
    library_csv = tmp_path / "library.csv"
    pd.DataFrame(
        {
            "nsynthon_id": ["0_0", "0_1", "1_0"],
            "smiles": ["CCO", "CCN", "CCC"],
            "relative_fraction": [0.3, 0.3, 0.4],
        }
    ).to_csv(library_csv, index=False)

    output_path = tmp_path / "out"
    config_path = _write_config(
        tmp_path,
        {
            "loglevel": "INFO",
            "affinity_generation": {
                "input_library_path": str(library_csv),
                "output_path": str(output_path),
                "runs": {
                    "random_run": {
                        "method": "random",
                        "method_parameters": {
                            "sampler": "truncated_normal",
                            "sampler_params": {"loc": 6.0, "scale": 1.0},
                        },
                    }
                },
            },
        },
    )

    result = _run_script("generate_affinities.py", config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    affinity_csv = output_path / "random_run.csv"
    assert affinity_csv.exists()
    affinities = pd.read_csv(affinity_csv)
    assert "pKd" in affinities.columns
    assert len(affinities) == 3


def test_selection_and_readout_smoke(tmp_path):
    library_csv = tmp_path / "library.csv"
    pd.DataFrame(
        {
            "nsynthon_id": ["0_0", "0_1", "1_0"],
            "smiles": ["CCO", "CCN", "CCC"],
            "relative_fraction": [0.3, 0.3, 0.4],
        }
    ).to_csv(library_csv, index=False)

    affinity_csv = tmp_path / "affinity_target.csv"
    pd.DataFrame({"pKd": [6.0, 7.0, 8.0]}).to_csv(affinity_csv, index=False)

    selection_output_path = tmp_path / "selection_out"
    readout_output_path_prefix = tmp_path / "readout_out"
    config_path = _write_config(
        tmp_path,
        {
            "loglevel": "INFO",
            "selection": {
                "input_library_path": str(library_csv),
                "input_affinity_paths": {"target": str(affinity_csv)},
                "initial_library_amount_mol": 2.0e-11,
                "experiment_volume_L": 200.0e-6,
                "output_path": str(selection_output_path),
                "selection_experiments": {
                    "target_experiment": {
                        "binding_sites": ["target"],
                        "recovery_fractions": [0.8],
                        "binding_site_concentrations_M": [1.0e-6],
                        "num_selection_rounds": 1,
                    }
                },
            },
            "amplification_readout": {
                "readout_seed": 0,
                "output_path_prefix": str(readout_output_path_prefix),
                "readout_runs": {
                    "reads_1k": {"num_reads": 1000, "output_path": "reads_1k"}
                },
            },
        },
    )

    result = _run_script("selection_and_readout.py", config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    selection_csv = selection_output_path / "target_experiment.csv"
    assert selection_csv.exists()
    assert len(pd.read_csv(selection_csv)) == 3

    readout_csv = readout_output_path_prefix / "reads_1k" / "target_experiment.csv"
    assert readout_csv.exists()
    readout_df = pd.read_csv(readout_csv)
    assert {"nsynthon_id", "count"} <= set(readout_df.columns)
    assert readout_df["count"].sum() == 1000


def test_prep_data_smoke(tmp_path):
    """
    Also exercises del_simulator/utils/aggregation_utils.py (get_merged_data,
    mean_aggregation, get_centroid_fp_smiles), which is never imported by any other test in
    this suite (its only production caller is this same script).
    """
    target_csv = tmp_path / "target.csv"
    pd.DataFrame({"nsynthon_id": ["0_0_0", "0_0_1"], "count": [50, 10]}).to_csv(
        target_csv, index=False
    )
    ntc_csv = tmp_path / "ntc.csv"
    pd.DataFrame({"nsynthon_id": ["0_0_0", "0_0_1"], "count": [5, 4]}).to_csv(
        ntc_csv, index=False
    )

    output_path = tmp_path / "out"
    config_path = _write_config(
        tmp_path,
        {
            "loglevel": "INFO",
            "data_prep": {
                "runs": {
                    "test_run": {
                        "intended_product_path": str(
                            RESOURCES_DIR / "dag0_bbs0_output.csv"
                        ),
                        "target_data_path": str(target_csv),
                        "ntc_data_path": str(ntc_csv),
                        "output_path": str(output_path),
                        "fp_generator_method": "morgan",
                        "fp_generator_method_parameters": {"nBits": 64, "radius": 2},
                        "bbs_per_nsynthon": 3,
                        "num_bbs_to_aggregate": 2,
                    }
                }
            },
        },
    )

    result = _run_script("prep_data.py", config_path)

    assert result.returncode == 0, result.stdout + result.stderr

    processed_df = pd.read_pickle(output_path / "processed_data_df.pkl")
    assert len(processed_df) > 0

    agg_df = pd.read_pickle(output_path / "processed_data_nsynthon_agg_2.pkl")
    assert len(agg_df) > 0


def test_featurize_dataset_smoke(tmp_path):
    smiles_tsv = tmp_path / "smiles.tsv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"]}).to_csv(
        smiles_tsv, sep="\t", index=False
    )

    output_fp_path = tmp_path / "out" / "fingerprints.npy"
    config_path = _write_config(
        tmp_path,
        {
            "loglevel": "INFO",
            "featurizer_runs": {
                "featurizer_runs": [
                    {
                        "input_path": str(smiles_tsv),
                        "output_path": str(output_fp_path),
                        "featurizer_method": "fingerprint",
                        "featurizer_parameters": {
                            "fp_generator_method": "morgan",
                            "fp_generator_method_parameters": {
                                "nBits": 64,
                                "radius": 2,
                            },
                            "chunksize": 2,
                            "parallelism": 1,
                            "smiles_column_name": "smiles",
                        },
                    }
                ]
            },
        },
    )

    result = _run_script("featurize_dataset.py", config_path)

    assert result.returncode == 0, result.stdout + result.stderr

    from del_simulator.utils.utils import load_npy_multiarray

    fps = load_npy_multiarray(str(output_fp_path))
    assert len(fps) == 4


def test_training_and_inference_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n_rows = 10
    # count_ratio enrichment = target_counts / max(ntc_counts, 1); half the rows land above
    # the threshold=2 cutoff and half below, giving a class-balanced dataset for
    # StratifiedKFold(n_splits=2).
    target_counts = [10] * (n_rows // 2) + [1] * (n_rows // 2)
    ntc_counts = [1] * (n_rows // 2) + [10] * (n_rows // 2)
    fingerprints = [
        csr_matrix(row.reshape(1, -1)) for row in rng.integers(0, 2, size=(n_rows, 16))
    ]
    data_df = pd.DataFrame(
        {
            "smiles": ["CCO"] * n_rows,
            "target_counts": target_counts,
            "ntc_counts": ntc_counts,
            "fingerprint": fingerprints,
        }
    )
    input_data_path = tmp_path / "data_df.pkl"
    data_df.to_pickle(input_data_path)

    model_output_path = tmp_path / "model_out"
    metrics_output_path = tmp_path / "metrics.jsonl"
    config_path = _write_config(
        tmp_path,
        {
            "loglevel": "INFO",
            "training_and_inference": {
                "runs": {
                    "test_run": {
                        "input_data_path": str(input_data_path),
                        "model_output_path": str(model_output_path),
                        "metrics_output_path": str(metrics_output_path),
                        "enrichment_method": "count_ratio",
                        "enrichment_threshold": 2,
                        "n_cv_splits": 2,
                        "ml_config": {
                            "method": "random_forest",
                            "method_parameters": {"n_estimators": 5},
                        },
                    }
                }
            },
        },
    )

    result = _run_script("training_and_inference.py", config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (model_output_path / "model_0.pkl").exists()
    assert metrics_output_path.exists()
