import re
from typing import Union

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, Mol, MolToSmiles
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem, DataStructs

import numpy as np
from scipy.sparse import csr_matrix
import logging

from functools import cache, partial
from chembl_structure_pipeline import standardizer
import logging
import sys, argparse
import logging
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigAttributeError
import typing as T
import os, sys, argparse
from dataclasses import dataclass, asdict
from scipy import sparse
from del_simulator.core import (
    MorganFingerPrintGeneratorParameters,
    FingerprintGeneratorMethod,
    RDKitFingerPrintGeneratorParameters,
    ZScoreProcessorParameters,
    RatioTestProcessorParameters,
)

from del_simulator.core import (
    BuildingBlocksSmiles,
    BuildingBlocksMols,
    BLANK_BUILDING_BLOCK_SMILES,
    SKIPPED_NSYTHON_BIT_ENCODING,
)


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return np.array(fp)


def extract_numbers(entry):
    return [int(num) for num in re.findall(r"\d+", entry)]


def parse_arguments():

    p = argparse.ArgumentParser()

    p.add_argument("config_file_path", type=str)
    p.add_argument("--config_attrs", nargs="+", default=[])
    args = p.parse_args()

    return args


def load_yaml(file_path: str, config_file_path: str) -> DictConfig:

    # Resolve the absolute path of the file
    base_dir = os.path.dirname(os.path.abspath(config_file_path))
    abs_path = os.path.join(base_dir, file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"YAML file '{abs_path}' not found")

    return OmegaConf.load(abs_path)


def load_config(
    config_file_path: str,
    cli_config_dotlist: T.List[str] = [],
    config_schema=None,
    config_section: str = None,
) -> DictConfig:

    # register a resolver to load yaml files relative to the main config file
    load_yaml_relative_path = partial(load_yaml, config_file_path=config_file_path)

    OmegaConf.register_new_resolver(
        "load_yaml", load_yaml_relative_path, replace=True, use_cache=False
    )

    file_conf = OmegaConf.load(config_file_path)
    cli_conf = OmegaConf.from_dotlist(cli_config_dotlist)
    conf = OmegaConf.merge(file_conf, cli_conf)

    if config_section is not None:
        for section in config_section.split("."):
            try:
                conf = OmegaConf.select(conf, section)
            except ConfigAttributeError as e:
                logging.error(
                    f"Could not find section {config_section} in config file. Error: {e}"
                )
                sys.exit(1)

    OmegaConf.resolve(conf)

    if config_schema is not None:
        schema = OmegaConf.structured(config_schema)
        unresolved_conf = OmegaConf.merge(schema, conf)
        conf = OmegaConf.to_object(unresolved_conf)

    return conf


def setup_logger(name, log_file, level=logging.INFO):

    # add a file logger in addition to the stream logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logging.getLogger().handlers[0].setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def get_del_enrichment(
    method: str,
    method_params: ZScoreProcessorParameters | RatioTestProcessorParameters,
    input_data: np.ndarray,
):  # fixme this is

    if (
        method == "zscore" or method == "normalized_zscore"
    ):  # ACS Comb. Sci. 2019, 21, 75−82
        assert len(input_data) == 1

        p = 1.0 / method_params.library_size
        expected_count = method_params.total_number_of_reads * p
        count = input_data.iloc[0]  # a single column

        enrichment = (count - expected_count) / np.sqrt(expected_count * (1.0 - p))

        if method == "normalized_zscore":
            enrichment = enrichment / np.sqrt(method_params.total_number_of_reads)
    elif (
        method == "poisson_ratio_test" or method == "scaled_poisson_ratio_test"
    ):  # J. Chem. Inf. Model. 2022, 62, 2316−2331

        assert len(input_data) == 2

        read_ratio = (
            method_params.total_number_of_reads_in_sample_1
            / method_params.total_number_of_reads_in_sample_2
            * method_params.enrichment_threshold
        )

        enrichment = (
            2
            * (
                np.sqrt(input_data.iloc[0] + 0.375)
                - np.sqrt(read_ratio * (input_data.iloc[1] + 0.375))
            )
            / np.sqrt(1.0 + (read_ratio))
        )

        if method == "scaled_poisson_ratio_test":
            enrichment = enrichment * np.sqrt(
                method_params.library_size
                / method_params.total_number_of_reads_in_sample_1
            )  # fixme figure out why this works?
    elif method == "count_ratio":
        enrichment = input_data.iloc[0] / max(input_data.iloc[1], 1)
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    return enrichment


def load_npy_multiarray(filepath: str) -> np.ndarray:
    npdata = []
    with open(filepath, "rb") as f:
        while True:
            try:
                npdata.append(np.load(f, allow_pickle=True))
            except EOFError as e:
                break
    return np.hstack(npdata)


SKIPPED_BB_ID_SENTINEL = -1


def extract_bb_ids(nsynthon_id: str) -> list[int]:
    # Split (not regex-extract digits) so a skipped-bit token like SKIPPED_NSYTHON_BIT_ENCODING
    # ("X") yields a sentinel entry instead of being silently dropped, which would otherwise
    # shorten the id list and misalign it against bbs_per_nsynthon for branched synthons.
    return [
        SKIPPED_BB_ID_SENTINEL if bb_id == SKIPPED_NSYTHON_BIT_ENCODING else int(bb_id)
        for bb_id in nsynthon_id.split("_")
    ]


def clean_building_blocks(raw_bbs: BuildingBlocksSmiles) -> BuildingBlocksMols:
    """
    Perform a single pass through the building blocks, clean the smiles and conver to mols.

    Args:
        raw_building_blocks (BuildingBlocks): The raw building blocks.

    Returns:
        BuildingBlocks: The cleaned building blocks.

    Raises:
        ValueError: If an invalid smiles is encountered.

    This function performs the following steps:
    1. Checks that smiles can be converted to/from mol.
    2. Removes the salt and alerts the user.
    3. Checks if there is enhanced stereochemistry and alerts the user.

    Returns a cleaned (desalted) BuildingBlocks.

    """
    logging.info("Cleaning Building Blocks")
    clean_bbs = BuildingBlocksMols({})

    for step_name, smiles_list in raw_bbs.items():
        clean_mols_list = []
        num_canonicalized = 0
        for smiles in smiles_list:
            if smiles == BLANK_BUILDING_BLOCK_SMILES:  # accept blank building blocks
                clean_mols_list.append(None)
                continue

            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                raise ValueError(f"Invalid smiles: {smiles}")

            canonical_smiles = Chem.MolToCXSmiles(mol)

            # Check if the mol can be converted back to smiles
            if canonical_smiles != smiles:
                num_canonicalized += 1
                logging.debug(
                    f"Converting {smiles} did not yield identical canonicalized smiles {canonical_smiles}. Proceeding with the converted, canonical CXSMILES."
                )

            # Remove salt and alert the user
            desalted_mol = standardizer.get_parent_mol(mol, neutralize=False)[0]

            if Chem.MolToCXSmiles(desalted_mol) != canonical_smiles:
                logging.warning(f"Salt removed from smiles: {canonical_smiles}")
                mol = desalted_mol

            # Check for enhanced stereochemistry and alert the user
            # TODO  #944: add in test that enhanced sterochemistry is detected and works through the library generator
            if len(mol.GetStereoGroups()) > 0:
                logging.warning(
                    f"Enhanced stereochemistry detected: {canonical_smiles}"
                )
            mol = Chem.AddHs(mol)
            mol.SetProp("smiles", canonical_smiles)
            clean_mols_list.append(mol)

        if num_canonicalized > 0:
            logging.warning(
                f"{step_name}: {num_canonicalized} of {len(smiles_list)} SMILES were not in canonical form!"
            )

        logging.info(f"{step_name}: {len(clean_mols_list)} SMILES added")

        clean_bbs[step_name] = clean_mols_list

    return clean_bbs


def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


class FingerprintFeaturizer:
    """
    quick utility class to compute fingerprints

    """

    def __init__(
        self,
        fp_generator_method: FingerprintGeneratorMethod,
        fp_generator_method_parameters: T.Union[
            MorganFingerPrintGeneratorParameters,
            RDKitFingerPrintGeneratorParameters,
            None,
        ],
        sparse: bool = True,
        sanitize: bool = True,
        to_numpy: bool = True,
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.sanitize = sanitize
        self.to_numpy = to_numpy
        self.fp_generator_method = fp_generator_method
        if (
            self.fp_generator_method == FingerprintGeneratorMethod.morgan
            or self.fp_generator_method == FingerprintGeneratorMethod.rdkit
        ):
            self.fp_generator_method_parameters = asdict(fp_generator_method_parameters)
        else:  # MACCS takes no parameters
            self.fp_generator_method_parameters = None

    def process_smiles(self, smiles):
        return self.process_mol(
            MolFromSmiles(
                MolToSmiles(MolFromSmiles(smiles, sanitize=self.sanitize)),
                sanitize=self.sanitize,
            )
        )

    @cache
    def process_mol(self, mol: Mol) -> Union[np.ndarray, csr_matrix]:
        if self.fp_generator_method == FingerprintGeneratorMethod.morgan:
            _fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, **self.fp_generator_method_parameters
            )

        elif self.fp_generator_method == FingerprintGeneratorMethod.rdkit:
            _fp = Chem.RDKFingerprint(mol, **self.fp_generator_method_parameters)

        elif self.fp_generator_method == FingerprintGeneratorMethod.maccs:
            _fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            raise NotImplementedError(
                f"Fingerprint generator method {self.fp_generator_method} not implemented"
            )

        if self.to_numpy:
            fp = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(_fp, fp)
        else:
            fp = _fp

        if not self.sparse:
            return fp
        else:
            fps = csr_matrix(fp)
            return fps
