#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd

# Function


def create_data_overview(df: pd.DataFrame):
    """
    Create Data Overview.

    Display basic information about the DataFrame.

    Input:
        df: (pd.DataFrame)
        Input DataFrame

    Returns:
        None
    """
    print("Shape of the DataFrame:", df.shape)
    print("\nVariable Information:")
    print(df.info())
    print("\nSummary of Numerical Variables:")
    print(df.describe())
    print("\nFirst Few Rows:")
    print(df.head())
