# A Computational Simulator for DNA-Encoded Libraries (DEL)

## Background

An in-silico simulator of DNA-encoded library (DEL) screening and analysis. It models a DEL
experiment end to end: combinatorial library generation from building blocks and a reaction
scheme, per-molecule affinity simulation, multi-round selection against one or more targets, PCR
amplification and NGS readout simulation, and data prep/ML training and inference (Random Forest
or ChemProp) for predicting hits from the resulting sequencing counts. This is the reference
implementation behind the paper below; see `notebooks/paper_figures.ipynb` for the code that
generates its figures.

## References
Please cite our paper: [Understanding Machine Learning Models Trained on
DNA-Encoded Libraries for Virtual Screening](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-8rw8j) if you find this project
helpful:

```
@article{menzeleev2025simulator,
  title={DEL Simulator: A Digital Twin for Understanding Machine Learning on DNA-Encoded Libraries},
  author={Menzeleev, Artur and Chitturi, Sathya and Davies, Geraint and Schroeder, Tony and Lee, Alpha},
  year={2025},
  publisher={ChemRxiv}
}
```

## Installation

Requires Python 3.11 (pinned in `pyproject.toml`) and [Poetry](https://python-poetry.org/).

### Using Poetry (recommended)

```
poetry install
```

If your default Python isn't 3.11, point Poetry at one explicitly first with
`poetry env use python3.11` (recent Poetry versions will also auto-detect and fall back to a
compatible interpreter on their own).

Run everything below with `poetry run <command>` (e.g. `poetry run python scripts/...`,
`poetry run pytest tests/`) -- verified with a fresh `poetry install` (Poetry 2.4.1). Note that
`poetry shell` is **not** a good alternative here: as of Poetry 2.0 it's no longer built in (needs
`poetry self add poetry-plugin-shell` first), and `poetry env activate`, its newer replacement,
didn't work out of the box in our testing either (missing bash activator). `poetry run` avoids
all of this and works the same way across Poetry versions.

### Running the tests

```
poetry run pytest tests/
```

### Using Docker (optional)

The package ships its own `Dockerfile`:

```
docker build -t del-sim:latest .
docker run -it --rm -v "$(pwd)":/app/del-simulator del-sim:latest bash
```

The image is built on `nvidia/cuda`, but a GPU is only used opportunistically for ChemProp
training -- CPU is fine for the walkthrough below.

A few things worth knowing about running this way (verified by actually running the full
walkthrough below in a container):

- Dependencies live in a Poetry-managed virtualenv, separate from the container's system Python.
  Once you're in the container's shell, run `poetry shell` (or prefix every command with
  `poetry run`) -- plain `python scripts/...` won't see rdkit/chemprop/etc.
- ChemProp training spins up multiple PyTorch DataLoader workers
  (`ml_config.method_parameters.num_workers`, 8 by default in the example config), which need
  more `/dev/shm` than Docker's small default (64MB). If you see
  `RuntimeError: unable to allocate shared memory`, add `--shm-size=2g` to `docker run` (training
  still completes without it, just noisily, since failed workers are retried).
- The container runs as root, so files written back through the bind mount (including the
  example experiment's output) end up owned by root on the host. Add
  `--user "$(id -u):$(id -g)"` to `docker run` if you'd rather they be owned by you, or just
  `sudo rm -rf` them later.

<details>
<summary><h2 style="display: inline;">Usage (Quickstart)</h2></summary>

Set the following environment variable:

- `DEL_SIMULATOR_EXPERIMENT_PATH` -- the path under which a given experiment's input and output
  data lives. Referenced from the example configs as `${oc.env:DEL_SIMULATOR_EXPERIMENT_PATH}`.

The simulator is driven by modular YAML configs (OmegaConf) -- each section corresponds to one
step below. A single config can drive an entire run end to end, or, more commonly, library
generation and affinity calculation are run once at the start of a study from a shared config,
with later steps iterated on via their own smaller configs.

The example below walks through an entire DEL experiment end to end from one config,
`data/experiments/example/config.yaml`.

0. **Preparation**

    ```
    export DEL_SIMULATOR_EXPERIMENT_PATH=$(pwd)/data/experiments/example
    ```

1. **Library generation**

    ```
    poetry run python scripts/generate_library.py data/experiments/example/config.yaml
    ```

    Also run this a second time against `config_no_yield_noise.yaml`, which regenerates the same
    library at 100% yield (no reaction noise). This produces the "intended product" ground-truth
    library under `intended_product/library.csv`, which step 4 (data prep) needs alongside the
    noisy library produced above:

    ```
    poetry run python scripts/generate_library.py data/experiments/example/config_no_yield_noise.yaml
    ```

2. **Affinity calculation**

    ```
    poetry run python scripts/generate_affinities.py data/experiments/example/config.yaml
    ```

3. **Selection and readout**

    ```
    poetry run python scripts/selection_and_readout.py data/experiments/example/config.yaml
    ```

4. **Data prep**

    ```
    poetry run python scripts/prep_data.py data/experiments/example/config.yaml
    ```

5. **Featurize the example screening set**

    A standalone utility for pre-featurizing a large screening set ahead of Random Forest
    inference (ChemProp featurizes SMILES on the fly, so it doesn't need this step -- only RF's
    `inference` block in step 6 does). The checked-in
    `inference/2024.01_Enamine_REAL_10k_smiles.csv` (a 10,000-SMILES sample) is what step 6's
    inference blocks screen against, so run this before step 6 or RF's inference will
    fail on a missing file:

    ```
    poetry run python scripts/featurize_dataset.py data/experiments/example/inference/featurize_screening_set.yaml
    ```

    Its companion `inference/affinity_seh_512_10k.csv` (also checked in, used for the optional
    screening-accuracy metrics in step 6) is already generated -- you don't need to reproduce it
    to run the walkthrough. For reference, it was made the same way as step 2's
    `generate_affinities.py`, just pointed at this screening set instead of the full library and
    renamed to match the filename step 6 expects.

    To use your own screening set instead of the checked-in one, point step 6's `inference`
    config at your own SMILES/affinity files (and re-run this step's `featurize_dataset.py`
    command against them first).

6. **ML training and inference**

    ```
    poetry run python scripts/training_and_inference.py data/experiments/example/config.yaml
    ```

    Trains a Random Forest and a ChemProp model on the prepped data from step 4, then runs both
    against the screening set from step 5 (RF from its pre-featurized `.npy`, ChemProp straight
    from the SMILES CSV -- see the `separator` comment in `config.yaml` if you swap in your own
    screening set with a different format).

    `DELSimulatorChemPropModel` (unlike the Random Forest harness) doesn't persist its trained
    model to `model_output_path` -- Lightning's checkpoint ends up under a `checkpoints/`
    directory relative to wherever you ran the script from instead. Only relevant if you need to
    reload the ChemProp model later outside of this same run.

    If you adapt this config for your own data and don't have a screening set yet, disable both
    runs' `inference` sections via `--config_attrs` rather than deleting them from the YAML:

    ```
    poetry run python scripts/training_and_inference.py data/experiments/example/config.yaml \
      --config_attrs \
        training_and_inference.runs.replica_0_100k_chemprop.inference=null \
        training_and_inference.runs.replica_0_100k_rf.inference=null
    ```

Every script also accepts `--config_attrs key.path=value ...` to override individual config
values from the command line (an OmegaConf dotlist merge) without editing the YAML file, e.g.:

```
poetry run python scripts/generate_library.py data/experiments/example/config.yaml --config_attrs library_generation.num_cpu=4
```

</details>

See `notebooks/` for example downstream analyses: `paper_figures.ipynb` reproduces the figures
from the paper above.
