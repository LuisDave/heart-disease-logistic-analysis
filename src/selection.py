from __future__ import annotations

import itertools
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .config import PREDICTORS
from .modeling import fit_logit


def exhaustive_model_search(
    df: pd.DataFrame,
    predictors=PREDICTORS,
    max_predictors: int = 4,
) -> pd.DataFrame:
    """
    Fit every combination of 1..max_predictors original source variables.

    Ranking is performed at the original-variable level. Categorical source
    variables may expand into several dummy coefficients internally.
    """
    rows: list[dict] = []
    warnings.simplefilter("ignore", ConvergenceWarning)

    for size in range(1, max_predictors + 1):
        for combo in itertools.combinations(predictors, size):
            try:
                model = fit_logit(df, list(combo))
                converged = bool(model.mle_retvals.get("converged", True))
                if not converged:
                    continue
                rows.append(
                    {
                        "predictors": " + ".join(combo),
                        "n_source_predictors": size,
                        "n_parameters": len(model.params),
                        "pseudo_r2_mcfadden": model.prsquared,
                        "log_likelihood": model.llf,
                        "ll_null": model.llnull,
                        "llr": model.llr,
                        "llr_p_value": model.llr_pvalue,
                        "aic": model.aic,
                        "bic": model.bic,
                    }
                )
            except Exception:
                # A failed candidate is skipped rather than stopping the complete search.
                continue

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No candidate logistic models could be fitted.")
    return result


def rank_by_llr_pvalue(results: pd.DataFrame) -> pd.DataFrame:
    """Minimum LLR p-value; ties are broken by larger LLR then smaller AIC."""
    return results.sort_values(
        ["llr_p_value", "llr", "aic"], ascending=[True, False, True]
    ).reset_index(drop=True)


def rank_by_pseudo_r2(results: pd.DataFrame) -> pd.DataFrame:
    """Maximum McFadden pseudo-R²; ties are broken by smaller AIC."""
    return results.sort_values(
        ["pseudo_r2_mcfadden", "aic"], ascending=[False, True]
    ).reset_index(drop=True)


def parse_predictor_string(value: str) -> list[str]:
    return [part.strip() for part in value.split("+")]
