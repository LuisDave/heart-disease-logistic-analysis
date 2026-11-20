from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_PREDICTORS,
    CATEGORICAL_PREDICTORS,
    CLASSIFICATION_THRESHOLD,
    CONTINUOUS_OR_COUNT_PREDICTORS,
    PREDICTORS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def formula_term(predictor: str) -> str:
    return f"C({predictor})" if predictor in CATEGORICAL_PREDICTORS else predictor


def make_formula(predictors: list[str] | tuple[str, ...]) -> str:
    rhs = " + ".join(formula_term(p) for p in predictors)
    return f"{TARGET} ~ {rhs}"


def fit_logit(df: pd.DataFrame, predictors=PREDICTORS):
    """Fit an interpretable statsmodels logistic regression model."""
    formula = make_formula(list(predictors))
    return smf.logit(formula, data=df).fit(disp=False, maxiter=500)


def coefficient_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "z": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_95_low": ci.iloc[:, 0].values,
            "ci_95_high": ci.iloc[:, 1].values,
        }
    )
    table["odds_ratio"] = np.exp(table["coef"])
    table["or_ci_95_low"] = np.exp(table["ci_95_low"])
    table["or_ci_95_high"] = np.exp(table["ci_95_high"])
    table["significant_0_05"] = table["p_value"] < 0.05
    return table


def model_metrics(model) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "nobs",
                "df_model",
                "df_resid",
                "pseudo_r2_mcfadden",
                "log_likelihood",
                "ll_null",
                "llr",
                "llr_p_value",
                "aic",
                "bic",
            ],
            "value": [
                model.nobs,
                model.df_model,
                model.df_resid,
                model.prsquared,
                model.llf,
                model.llnull,
                model.llr,
                model.llr_pvalue,
                model.aic,
                model.bic,
            ],
        }
    )


def likelihood_ratio_block_tests(df: pd.DataFrame, predictors=PREDICTORS) -> pd.DataFrame:
    """
    Test each original source predictor as a block by comparing the full model
    against a nested model with that predictor removed.
    """
    full = fit_logit(df, predictors)
    rows = []
    for predictor in predictors:
        reduced_predictors = [p for p in predictors if p != predictor]
        reduced = fit_logit(df, reduced_predictors)
        lr_stat = 2 * (full.llf - reduced.llf)
        df_diff = int(full.df_model - reduced.df_model)
        p_value = chi2.sf(lr_stat, df_diff)
        rows.append(
            {
                "predictor": predictor,
                "lr_stat": lr_stat,
                "df": df_diff,
                "p_value": p_value,
                "significant_0_05": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows).sort_values(["p_value", "lr_stat"], ascending=[True, False])


def build_sklearn_pipeline(predictors=PREDICTORS) -> Pipeline:
    predictors = list(predictors)
    categorical = [p for p in predictors if p in CATEGORICAL_PREDICTORS]
    binary = [p for p in predictors if p in BINARY_PREDICTORS]
    continuous = [p for p in predictors if p in CONTINUOUS_OR_COUNT_PREDICTORS]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical,
            ),
            ("continuous", StandardScaler(), continuous),
            ("binary", "passthrough", binary),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(max_iter=2000, solver="lbfgs")
    return Pipeline([("preprocess", transformer), ("model", classifier)])


def evaluate_holdout(
    df: pd.DataFrame,
    predictors=PREDICTORS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    threshold: float = CLASSIFICATION_THRESHOLD,
):
    """Evaluate predictive performance on a stratified holdout set."""
    predictors = list(predictors)
    X = df[predictors]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    pipe = build_sklearn_pipeline(predictors)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "threshold": threshold,
    }
    return pipe, cm, metrics, (X_train, X_test, y_train, y_test, proba, pred)
