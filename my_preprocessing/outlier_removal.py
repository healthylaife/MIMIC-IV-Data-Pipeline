import pandas as pd
import numpy as np


def compute_outlier_imputation(
    arr: np.ndarray, cut_off: float, left_thresh: float, impute: bool
) -> np.ndarray:
    """Imputes or flags outliers in an array based on percentile thresholds.

    Args:
        arr (np.ndarray): The input array.
        cut_off (float): The upper percentile threshold to define outliers.
        left_thresh (float): The lower percentile threshold to define outliers.
        impute (bool): If True, outliers are imputed with threshold values. If False, they are replaced with NaN.

    Returns:
        np.ndarray: The array with outliers imputed or flagged.
    """
    lower_bound = np.percentile(arr, left_thresh)
    upper_bound = np.percentile(arr, cut_off)

    if impute:
        np.clip(arr, lower_bound, upper_bound, out=arr)
    else:
        arr = np.where((arr < lower_bound) | (arr > upper_bound), np.nan, arr)


def outlier_imputation(
    data: pd.DataFrame,
    id_attribute: str,
    value_attribute: str,
    cut_off: float,
    left_thresh: float,
    impute: bool,
) -> pd.DataFrame:
    """
    Applies outlier imputation or removal to a specific attribute of a DataFrame, grouped by another attribute.

    Args:
    data (pd.DataFrame): The input DataFrame.
    id_attribute (str): The attribute to group by.
    value_attribute (str): The attribute to apply outlier processing.
    cut_off (float): Upper percentile threshold for defining outliers.
    left_thresh (float): Lower percentile threshold for defining outliers.
    impute (bool): If True, imputes outliers with threshold values; if False, replaces them with NaN.

    Returns:
    pd.DataFrame: The DataFrame with outlier processing applied.
    """

    def impute_group(group):
        return compute_outlier_imputation(group, cut_off, left_thresh, impute)

    # Apply the outlier imputation or removal to each group
    data[value_attribute] = data.groupby(id_attribute)[value_attribute].transform(
        impute_group
    )

    # Optionally drop rows with NaN values in the value_attribute column
    if not impute:
        data = data.dropna(subset=[value_attribute])

    return data
