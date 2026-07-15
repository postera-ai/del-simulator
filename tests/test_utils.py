from scipy.sparse import csr_matrix

from del_simulator.core import (
    FingerprintGeneratorMethod,
    RDKitFingerPrintGeneratorParameters,
)
from del_simulator.utils.utils import (
    FingerprintFeaturizer,
    extract_bb_ids,
    SKIPPED_BB_ID_SENTINEL,
)


def test_process_smiles_rdkit_fingerprint():
    """
    RDKitFingerPrintGeneratorParameters used to define a "radius" field (copy-pasted from
    MorganFingerPrintGeneratorParameters), but Chem.RDKFingerprint has no such kwarg -- this
    method crashed unconditionally until the field was renamed to maxPath.
    """
    featurizer = FingerprintFeaturizer(
        fp_generator_method=FingerprintGeneratorMethod.rdkit,
        fp_generator_method_parameters=RDKitFingerPrintGeneratorParameters(maxPath=2),
    )

    fp = featurizer.process_smiles("CCO")

    assert isinstance(fp, csr_matrix)
    assert fp.nnz > 0


def test_process_smiles_maccs_fingerprint():
    """MACCS takes no generator parameters -- fp_generator_method_parameters is None."""
    featurizer = FingerprintFeaturizer(
        fp_generator_method=FingerprintGeneratorMethod.maccs,
        fp_generator_method_parameters=None,
    )

    fp = featurizer.process_smiles("CCO")

    assert isinstance(fp, csr_matrix)
    assert fp.shape[1] == 167  # MACCS keys length
    assert fp.nnz > 0


def test_extract_bb_ids_plain_digits():
    assert extract_bb_ids("3_7_12") == [3, 7, 12]


def test_extract_bb_ids_maps_skipped_bit_token_to_sentinel():
    """
    extract_bb_ids used to regex-extract only digit runs, silently dropping the "X"
    skipped-bit token used for branched-synthon nsynthon_ids (e.g. "0_0_X"). That shortened
    the returned id list, misaligning downstream `np.array(extracted_bb_ids)[:, i]` indexing
    in aggregation_utils.get_merged_data.
    """
    assert extract_bb_ids("0_0_X") == [0, 0, SKIPPED_BB_ID_SENTINEL]
    assert extract_bb_ids("0_X_0") == [0, SKIPPED_BB_ID_SENTINEL, 0]
