import pandas as pd

from del_simulator.utils.utils import SKIPPED_BB_ID_SENTINEL
from del_simulator.utils.aggregation_utils import get_merged_data


def test_get_merged_data_with_branched_synthons():
    """
    get_merged_data used extract_bb_ids to build a B0/B1/.../B{bbs_per_nsynthon-1} array via
    np.array(extracted_bb_ids)[:, i]. Branched nsynthon_ids containing the skipped-bit token
    ("0_0_X") used to produce a shorter id list than plain-digit ids, making the array ragged
    and raising/misaligning at this indexing step.
    """
    target_df = pd.DataFrame({"nsynthon_id": ["0_0_X", "0_X_0"], "count": [5, 3]})
    ntc_df = pd.DataFrame({"nsynthon_id": ["0_0_X", "0_X_0"], "count": [1, 1]})
    intended_product_df = pd.DataFrame(
        {"nsynthon_id": ["0_0_X", "0_X_0"], "smiles": ["CCO", "CCN"]}
    )

    merged = get_merged_data(target_df, ntc_df, intended_product_df, bbs_per_nsynthon=3)

    assert list(merged["B0"]) == [0, 0]
    assert list(merged["B1"]) == [0, SKIPPED_BB_ID_SENTINEL]
    assert list(merged["B2"]) == [SKIPPED_BB_ID_SENTINEL, 0]
