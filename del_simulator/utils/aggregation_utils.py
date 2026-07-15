from itertools import combinations
import numpy as np
import pandas as pd

from del_simulator.utils.utils import extract_bb_ids


def n_choose_k(n, k):
    return list(combinations(n, k))


def get_centroid_fp_smiles(
    fps: list | np.ndarray, smiles: list
) -> tuple[np.ndarray, str]:
    fps_array = np.array(fps)
    centroid = np.mean(fps_array, axis=0)
    distances = np.sum((fps_array - centroid[np.newaxis, :]) ** 2, axis=1)
    closest_idx = np.argmin(distances)
    return fps[closest_idx], smiles[closest_idx]


def mean_aggregation(
    df: pd.DataFrame, bbs_per_nsynthon: int, num_bbs_to_aggregate: int
) -> pd.DataFrame:

    possible_list = [f"B{i}" for i in range(bbs_per_nsynthon)]
    # choose type of aggregation; e.g. if possible list has three elements, then nsynthon_aggregation = 2 is disynthon aggregation
    group_by_list = n_choose_k(possible_list, num_bbs_to_aggregate)

    df_list = []

    for bb_subset in group_by_list:
        grouped_df = (
            df.groupby(list(bb_subset))
            .agg(
                {
                    "target_counts": "sum",
                    "ntc_counts": "sum",
                    "nsynthon_id": list,
                    "fingerprint": list,
                    "smiles": list,
                }
            )
            .reset_index()
        )

        grouped_df[["fingerprint", "smiles"]] = grouped_df.apply(
            lambda row: pd.Series(
                get_centroid_fp_smiles(row["fingerprint"], row["smiles"])
            ),
            axis=1,
        )
        df_list.append(grouped_df)

    result_df = pd.concat(df_list, axis=0)
    return result_df


def get_merged_data(
    target_df: pd.DataFrame,
    ntc_df: pd.DataFrame,
    intended_product_df: pd.DataFrame,
    bbs_per_nsynthon: int,
) -> pd.DataFrame:
    merged_data = pd.merge(target_df, ntc_df, how="outer", on="nsynthon_id")
    merged_data.fillna(0, inplace=True)
    merged_data.columns = ["nsynthon_id", "target_counts", "ntc_counts"]
    data_df = pd.merge(merged_data, intended_product_df, how="inner", on="nsynthon_id")

    extracted_bb_ids = [extract_bb_ids(entry) for entry in data_df["nsynthon_id"]]

    # extract_bb_ids maps a skipped-bit token ("X") to SKIPPED_BB_ID_SENTINEL rather than
    # dropping it, so branched-synthon ids (e.g. "0_0_X") still produce a rectangular array.
    for i in range(bbs_per_nsynthon):
        data_df[f"B{i}"] = np.array(extracted_bb_ids)[:, i]
    return data_df
