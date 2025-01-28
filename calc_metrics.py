#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

# Function


def calc_metrics(
    y_test: pd.Series, y_pred: np.ndarray, metric_name: str
) -> float:
    """
    Calculate performance metrics.

    Input:
        y_test: pd.Series
            Test labels.
        y_pred: np.ndarray
            Predicted labels.
        metric_name: str
            Metric to be calculated.

    Return:


    """
    if metric_name == "rmse":
        metric = root_mean_squared_error(y_test, y_pred)
    elif metric_name == "r2":
        metric = r2_score(y_test, y_pred)
    elif metric_name == "f1":
        metric = f1_score(y_test, y_pred)
    elif metric_name == "acc":
        metric = accuracy_score(y_test, y_pred)
    elif metric_name == "rocauc":
        metric = roc_auc_score(y_test, y_pred)
    elif metric_name == "prec":
        metric = precision_score(y_test, y_pred)
    else:
        metric = np.nan
    return metric
