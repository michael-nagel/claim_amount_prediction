#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd

# Function


def enc_categ_vars(df: pd.DataFrame, col: str, rm_first: bool) -> pd.DataFrame:
    """
    Encode a categorical variable.

    This function encodes a categorical variable to a one-hot integer
    array and concatenates it to the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variables to be encoded.
    col : str
        Variables to be encoded.
    rm_first : bool
        Whether to get k-1 dummies out of k categorical levels by
        removing the first level.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded variables.

    Examples
    --------
    for ele in df.select_dtypes(include=["category"]).columns:
        df = enc_categ_var(df, ele, True)
    """
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[col],
                prefix=col,
                drop_first=rm_first,
                prefix_sep="",
                dtype="uint8",
            ),
        ],
        axis=1,
    ).drop([col], axis=1)

    return df
