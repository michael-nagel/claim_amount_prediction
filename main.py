#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Imports

from collections import defaultdict

import arff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2

from calc_metrics import calc_metrics
from create_data_overview import create_data_overview
from enc_categ_vars import enc_categ_vars
from save_fig import save_fig

# %% External Files

data_freq = arff.load("data/freMTPL2freq.arff")

df_freq = pd.DataFrame(
    data_freq,
    columns=[
        "IDpol",
        "ClaimNb",
        "Exposure",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
    ],
)

data_sev = arff.load("data/freMTPL2sev.arff")
df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])

# %% Cockpit

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

plt.close("all")
sns.set_theme(style="whitegrid")

# %% Create data overview for df_freq
create_data_overview(df=df_freq)

# %% Create data overview for df_sev
create_data_overview(df=df_sev)

# %% Shaping

# Aggregate claim amount if there are multiple damages for the same ID
df_sev = df_sev.groupby("IDpol")["ClaimAmount"].sum()

# Merge
df = pd.merge(left=df_freq, right=df_sev, how="left", on="IDpol")

# Missing claim amounts correspont to 0
df.loc[df["ClaimAmount"].isna(), "ClaimAmount"] = 0

df["ClaimNb"].value_counts()

# Create dummy variable for claim
df["HadClaim"] = 0
df.loc[df["ClaimAmount"] > 0, "HadClaim"] = 1

# Drop ID
df = df.drop("IDpol", axis=1)

# %% Descriptives for the numerical variables
df.describe()

# %% Descriptives for the categorical variables
df.describe(include="object")

# %% Distribution of HadClaim

_, ax = plt.subplots()
sns.countplot(
    data=df,
    x="HadClaim",
    order=df["HadClaim"].value_counts().index,
    ax=ax,
    stat="probability",
)
ax.set(title="Count Plot of HadClaim", xlabel="HadClaim", ylabel="Share")
plt.yticks(np.arange(0, 1.01, 0.1))
save_fig("countplot_hadclaim")
plt.show(block=False)

# %% Distribution of numerical variables
num_vars = df.select_dtypes(include=[float]).columns

fig, ax = plt.subplots(4, 2, figsize=(12, 10))
ax = ax.flatten()
for i, col in enumerate(num_vars):
    sns.violinplot(data=df, x=col, ax=ax[i])
    ax[i].set(title=col, xlabel="", ylabel="")
fig.subplots_adjust(hspace=0.4, wspace=0.2)
fig.suptitle("Distribution of Numerical Variables", fontsize=16, y=0.94)
save_fig("dist_num_vars")
plt.show(block=False)

# %% Distribution of categorical variables
categ_vars = df.select_dtypes(include=["object"]).columns

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()
for i, col in enumerate(categ_vars):
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax[i])
    ax[i].set(title=col, xlabel="", ylabel="")
    if i == 3:
        ax[i].tick_params(axis="x", rotation=90)
fig.suptitle("Frequency Plot of Categorical Variables", fontsize=16, y=0.94)
save_fig("freq_plot_categ_vars")
plt.show(block=False)

# %% Correlation Analysis

# Correlations between numerical variables - use Spearman rather than Pearson!
correlation_matrix = df[num_vars].corr(method="spearman")

_, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
)
plt.title("Correlation Matrix", fontsize=16)
save_fig("heatmap_num_vars")
plt.show(block=False)

# %% Eyeballing Potential Relationships Between Target and Features

fig, ax = plt.subplots(4, 2, figsize=(12, 10))
ax = ax.flatten()
for i, col in enumerate(num_vars):
    if col != "ClaimAmount":
        sns.scatterplot(
            data=df,
            x=col,
            y="ClaimAmount",
            ax=ax[i],
        )
        ax[i].set(title="", xlabel=col, ylabel="Claim Amount")
fig.subplots_adjust(hspace=0.5, wspace=0.25)
fig.suptitle(
    "Scatterplot Claim Amount Against Numerical Variables", fontsize=16, y=0.94
)
plt.show(block=False)

# %% Encoding

# Onehot encode categorical variables
for col in categ_vars:
    if col != "Region":
        df = enc_categ_vars(df=df, col=col, rm_first=False)

# Label encode Region for computational efficiency (>20 categories)
encoder = LabelEncoder()
df["Region"] = encoder.fit_transform(df["Region"])

# %% Transformations

X = df.drop(["HadClaim", "ClaimAmount"], axis=1)
y = df["ClaimAmount"].copy()

# Log-transform the target to handle skewness
y = np.log1p(y)  # (add a small constant to avoid log(0))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

# %% Standardize the non-dummy features: fit scaler to train and apply on test

scale_vars = X_train.columns[X_train.dtypes != "uint8"]
scaler = StandardScaler()
X_train[scale_vars] = scaler.fit_transform(X_train[scale_vars])
X_test[scale_vars] = scaler.transform(X_test[scale_vars])

# %% Create a container for storing metrics later

metrics = defaultdict(lambda: defaultdict())

# %% Zero Inflated Model (Tweedie Family):

# Initialize Tweedie Regressor
tw = TweedieRegressor()

# Define the parameter grid
tw_param_dist = {
    "power": [0, 1, 1.5],
    "alpha": [0.1, 1, 2],
    "fit_intercept": [True, False],
    "link": ["auto", "identity", "log"],
}

# Initialize RandomizedSearchCV
tw_random_search = RandomizedSearchCV(
    estimator=tw,
    param_distributions=tw_param_dist,
    n_iter=10,  # Number of random combinations to try
    scoring="neg_mean_squared_error",  # Metric to evaluate
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Print updates
    random_state=SEED,  # Ensure reproducibility
    n_jobs=-1,  # Use all available CPU cores
)

# Fit the RandomizedSearchCV
tw_random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters:", tw_random_search.best_params_)
print("Best CV Score (MSE):", tw_random_search.best_score_)

# Use the best model for predictions
best_tw = tw_random_search.best_estimator_
y_pred = best_tw.predict(X_test)

# %% Calculate Performance Metrics

metrics["RMSE"]["Tweedie"] = calc_metrics(y_test, y_pred, "rmse")
metrics["R-Squared"]["Tweedie"] = calc_metrics(y_test, y_pred, "r2")
print(f"RMSE Tweedie: {metrics["RMSE"]["Tweedie"]:.4f}")
print(f"R-Squared Tweedie: {metrics["R-Squared"]["Tweedie"]:.4f}")

# %% Feature Importance

feat_import_tw = pd.Series(np.abs(best_tw.coef_), index=X.columns).sort_values(
    ascending=False
)

_, ax = plt.subplots()
feat_import_tw.nlargest(15).plot(kind="barh", ax=ax)
ax.set(
    title="Feature Importance Tweedie",
    xlabel="Estimated Coefficient",
    ylabel="Feature",
)
save_fig("feat_import_tweedie")
plt.show(block=False)

# %% Neural Network

# Build model
nn = Sequential(
    [
        InputLayer(shape=(X_train.shape[1],)),
        Dense(
            units=128,
            activation="relu",
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
        ),
        Dropout(0.3),
        Dense(
            units=64,
            activation="tanh",
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
        ),
        Dropout(0.3),
        Dense(
            units=32,
            activation="relu",
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
        ),
        Dense(units=1),
    ]
)

nn.compile(
    optimizer="adam",
    loss="mse",  # Mean Squared Error for regression
    metrics=["mse"],
)

# Train the regression model
history = nn.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=1024,
    verbose=1,
)

# %% Plot Training and Validation Loss

_, ax = plt.subplots()
ax.plot(history.history["loss"], label="Training Loss")
ax.plot(history.history["val_loss"], label="Validation Loss")
ax.set(xlabel="Epochs", ylabel="Loss", title="Training and Validation Loss")
plt.xticks(np.arange(0, 25, 5))
plt.legend()
save_fig("nn_train_val_loss")
plt.show(block=False)

# %% Predict Claim Amounts

y_pred = nn.predict(X_test)

# %% Evaluate Performance

metrics["RMSE"]["NeuralNet"] = calc_metrics(y_test, y_pred, "rmse")
metrics["R-Squared"]["NeuralNet"] = calc_metrics(y_test, y_pred, "r2")
print(f"RMSE Neural Network: {metrics["RMSE"]["NeuralNet"]:.4f}")
print(f"R-Squared Neural Network: {metrics["R-Squared"]["NeuralNet"]:.4f}")

# %% Feature Importance

feat_import_nn = permutation_importance(
    nn,
    X_test,
    y_test,
    scoring="neg_mean_squared_error",
    n_repeats=2,
    random_state=42,
)

feat_import_nn = pd.Series(
    np.abs(feat_import_nn.importances_mean), index=X.columns
).sort_values(ascending=False)

_, ax = plt.subplots()
feat_import_nn.nlargest(15).plot(kind="barh", ax=ax)
ax.set(
    title="Feature Importance Neural Network",
    xlabel="Estimated Coefficient",
    ylabel="Feature",
)
save_fig("feat_import_neural_net")
plt.show(block=False)

# %% Random Forest Estimator With Oversampled Non-Zero Claims:

# Oversample non-zero claims to achieve a balanced dataset
X_train_boot, y_train_boot = resample(
    X_train[y_train > 0],
    y_train[y_train > 0],
    n_samples=y_train[y_train == 0].shape[0],
    replace=True,
    random_state=SEED,
)

X_train_boot = pd.concat([X_train_boot, X_train[y_train == 0]], axis=0)
y_train_boot = pd.concat([y_train_boot, y_train[y_train == 0]], axis=0)

# %% Train Random Forest

rf = RandomForestRegressor(n_estimators=10, random_state=SEED)
rf.fit(X_train_boot, y_train_boot)
y_pred = rf.predict(X_test)

# %% Evaluate Performance

metrics["RMSE"]["RandomForest"] = calc_metrics(y_test, y_pred, "rmse")
metrics["R-Squared"]["RandomForest"] = calc_metrics(y_test, y_pred, "r2")
print(f"RMSE Random Forest: {metrics["RMSE"]["RandomForest"]:.4f}")
print(f"R-Squared Random Forest: {metrics["R-Squared"]["RandomForest"]:.4f}")

# Feature Importance
feat_import_rf = pd.Series(
    rf.feature_importances_, index=X_train.columns
).sort_values(ascending=False)

_, ax = plt.subplots()
feat_import_rf.nlargest(15).plot(kind="barh", ax=ax)
ax.set(
    title="Feature Importance Random Forest",
    xlabel="Importance",
    ylabel="Feature",
)
save_fig("feat_import_random_forest")
plt.show(block=False)

# %% Two-Stage Approach With XGBoost

# 1. Classification

# Define class variable
y_train_class = (y_train > 0).astype(int)
y_test_class = (y_test > 0).astype(int)

train_counts = y_train_class.value_counts(normalize=True)
test_counts = y_test_class.value_counts(normalize=True)
df_counts = pd.DataFrame(
    {"Train set": train_counts, "Test set": test_counts}
).sort_index()

# %% Demonstrate Equal Share of Positives in Train and Test Set

_, ax = plt.subplots()
df_counts.plot(kind="bar", ax=ax)
ax.set(
    ylabel="Share",
    xlabel="HadClaim",
    title="Comparison of Class Share in Train and Test Set",
)
plt.show(block=False)

# %% Determine Hyperparameters for XGBoost Classifier

# Hyperparameters grid for RandomizedSearchCV
xgb_param_dist = {
    "n_estimators": [100, 200, 500],
    "max_depth": [2, 5, 8],
    "min_child_weight": [2, 5, 8],
    "subsample": [0.75, 0.9],
    "colsample_bytree": [0.25, 0.5, 0.75],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}

# Sample weight to tackle imbalanced classes
scaling_weigth = train_counts[0] / train_counts[1]

# %% Initialize and Train XGBoost Classifier

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scaling_weigth,
    verbosity=1,
    random_state=SEED,
)

# Set up RandomizedSearchCV
xgb_clf_random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=xgb_param_dist,
    n_iter=10,
    scoring="roc_auc",
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=SEED,
)

xgb_clf_random_search.fit(X_train, y_train_class)

# %% Get the Best Model and Use for Prediction

best_xgb_clf = xgb_clf_random_search.best_estimator_

y_pred_prob = best_xgb_clf.predict_proba(X_test)[:, 1]
y_pred_class = best_xgb_clf.predict(X_test)

# %% Check Calibration

print(pd.Series(y_pred_class).value_counts(True))

# %% Plot Confusion Matrix Using Seaborn Heatmap

conf_matrix = confusion_matrix(y_test_class, y_pred_class)
_, ax = plt.subplots()
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=[0, 1],
    yticklabels=[0, 1],
    ax=ax,
)
ax.set(
    xlabel="Predicted Labels", ylabel="True Labels", title="Confusion Matrix"
)
save_fig("confusion_matrix")
plt.show(block=False)

# %% Evaluate Performance

metrics["ROCAUC"]["XGBoostClf"] = calc_metrics(
    y_test_class, y_pred_prob, "rocauc"
)
metrics["F1"]["XGBoostClf"] = calc_metrics(y_test_class, y_pred_class, "f1")
print(f"ROC-AUC XGBoost Classification: {metrics["ROCAUC"]["XGBoostClf"]:.4f}")
print(f"F1-Score XGBoost Classification: {metrics["F1"]["XGBoostClf"]:.4f}")

# %% Plot feature importance

_, ax = plt.subplots()
xgb.plot_importance(best_xgb_clf, max_num_features=15, ax=ax)
ax.set(title="Feature Importance XGBoost Classifier")
save_fig("feat_import_xgboost_clf")
plt.show(block=False)

# %% 2. Regression

# Only use non-zero claims for train and test set
X_train_reg = X_train[y_train > 0]
y_train_reg = y_train[y_train > 0]
X_test_reg = X_test[y_test > 0]
y_test_reg = y_test[y_test > 0]

# %% Initialize XGBoost Regressor

xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    scale_pos_weight=scaling_weigth,
    verbosity=1,
    random_state=SEED,
)

# Set up RandomizedSearchCV
xgb_reg_random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=xgb_param_dist,
    n_iter=25,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=SEED,
)

xgb_reg_random_search.fit(X_train_reg, y_train_reg)

# %% Get the Best Model for Prediction

best_xgb_reg = xgb_reg_random_search.best_estimator_

y_pred_reg = best_xgb_reg.predict(X_test_reg)

# %% Evaluate Performance

metrics["RMSE"]["XGBoostReg"] = calc_metrics(y_test_reg, y_pred_reg, "rmse")
metrics["R-Squared"]["XGBoostReg"] = calc_metrics(y_test_reg, y_pred_reg, "r2")
print(f"RMSE XGBoost Regression: {metrics["RMSE"]["XGBoostReg"]:.4f}")
print(
    f"R-Squared XGBoost Regression: {metrics["R-Squared"]["XGBoostReg"]:.4f}"
)

# %% Plot Feature Importance

_, ax = plt.subplots()
xgb.plot_importance(best_xgb_reg, max_num_features=15, ax=ax)
ax.set(title="Feature Importance XGBoost Regressor")
save_fig("feat_import_xgboost_reg")
plt.show(block=False)

# %% 3. Final predictions

y_pred = np.zeros_like(y_test, dtype=float)
y_pred[y_pred_class == 1] = best_xgb_reg.predict(X_test[y_pred_class == 1])

# %% Evaluate Combined Performance

metrics["RMSE"]["XGBoost"] = calc_metrics(y_test, y_pred, "rmse")
metrics["R-Squared"]["XGBoost"] = calc_metrics(y_test, y_pred, "r2")
print(f"RMSE XGBoost: {metrics["RMSE"]["XGBoost"]:.4f}")
print(f"R-Squared XGBoost: {metrics["R-Squared"]["XGBoost"]:.4f}")

# %% Comparison of Performance Metrics

metrics_df = pd.DataFrame(
    {"RMSE": metrics["RMSE"], "R-Squared": metrics["R-Squared"]}
)
metrics_df = metrics_df.drop("XGBoostReg")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.barplot(metrics_df["RMSE"].sort_values(), ax=ax[0])
sns.barplot(metrics_df["R-Squared"].sort_values(ascending=False), ax=ax[1])
fig.suptitle("Comparison of Performance Metrics")
save_fig("comparison_metrics")
plt.show(block=False)

# %% Predict Expected Claim Amount Per Year for Each Customer

df_pred = df.drop(["HadClaim", "ClaimAmount"], axis=1)

# Apply trained scaler
df_pred[scale_vars] = scaler.transform(df_pred[scale_vars])

# Predict claim amount using trained network
pred_claim_amount = nn.predict(df_pred)

# Apply exponential function to get back initial scale
pred_claim_amount = np.expm1(pred_claim_amount)
