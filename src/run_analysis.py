from __future__ import annotations

import json

import pandas as pd

from .config import OUTPUT_FIGURES, OUTPUT_TABLES, PREDICTORS
from .data import audit_data, clean_data, descriptive_summary, load_raw_data, save_clean_data
from .modeling import (
    coefficient_table,
    evaluate_holdout,
    fit_logit,
    likelihood_ratio_block_tests,
    model_metrics,
)
from .selection import (
    exhaustive_model_search,
    parse_predictor_string,
    rank_by_llr_pvalue,
    rank_by_pseudo_r2,
)
from .visualization import (
    save_confusion_matrix,
    save_correlation_matrix,
    save_logistic_curve_oldpeak,
    save_model_selection_top10,
    save_roc_curve,
    save_target_balance,
)


def main() -> dict:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    raw_audit = audit_data(raw)
    clean, cleaning_audit = clean_data(raw)
    save_clean_data(clean)

    raw_audit.to_csv(OUTPUT_TABLES / "raw_data_audit.csv", index=False)
    cleaning_audit.to_csv(OUTPUT_TABLES / "cleaning_audit.csv", index=False)
    descriptive_summary(clean).to_csv(OUTPUT_TABLES / "descriptive_statistics.csv")

    save_target_balance(clean, OUTPUT_FIGURES / "target_balance.png")
    save_correlation_matrix(clean, OUTPUT_FIGURES / "correlation_matrix.png")
    save_logistic_curve_oldpeak(clean, OUTPUT_FIGURES / "logistic_curve_oldpeak.png")

    full_model = fit_logit(clean, PREDICTORS)
    full_coef = coefficient_table(full_model)
    full_metrics = model_metrics(full_model)
    block_tests = likelihood_ratio_block_tests(clean, PREDICTORS)

    full_coef.to_csv(OUTPUT_TABLES / "full_model_coefficients.csv", index=False)
    full_metrics.to_csv(OUTPUT_TABLES / "full_model_metrics.csv", index=False)
    block_tests.to_csv(OUTPUT_TABLES / "predictor_block_lr_tests.csv", index=False)
    (OUTPUT_TABLES / "full_model_summary.txt").write_text(
        full_model.summary().as_text(), encoding="utf-8"
    )

    _, cm_full, metrics_full, holdout_full = evaluate_holdout(clean, PREDICTORS)
    pd.DataFrame(cm_full, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        OUTPUT_TABLES / "confusion_matrix_full.csv"
    )
    pd.DataFrame([metrics_full]).to_csv(
        OUTPUT_TABLES / "holdout_metrics_full.csv", index=False
    )
    save_confusion_matrix(cm_full, OUTPUT_FIGURES / "confusion_matrix_full.png", "Matriz de confusión · modelo completo")
    save_roc_curve(
        holdout_full[3],
        holdout_full[4],
        OUTPUT_FIGURES / "roc_curve_full.png",
        "Curva ROC · modelo completo",
    )

    search = exhaustive_model_search(clean, PREDICTORS, max_predictors=4)
    by_llr = rank_by_llr_pvalue(search)
    by_r2 = rank_by_pseudo_r2(search)
    search.to_csv(OUTPUT_TABLES / "model_search_all.csv", index=False)
    by_llr.head(20).to_csv(OUTPUT_TABLES / "top_models_by_llr_pvalue.csv", index=False)
    by_r2.head(20).to_csv(OUTPUT_TABLES / "top_models_by_pseudo_r2.csv", index=False)
    save_model_selection_top10(by_r2, OUTPUT_FIGURES / "top_reduced_models.png")

    best_llr_predictors = parse_predictor_string(by_llr.iloc[0]["predictors"])
    best_r2_predictors = parse_predictor_string(by_r2.iloc[0]["predictors"])

    best_llr_model = fit_logit(clean, best_llr_predictors)
    best_r2_model = fit_logit(clean, best_r2_predictors)
    coefficient_table(best_llr_model).to_csv(
        OUTPUT_TABLES / "best_llr_model_coefficients.csv", index=False
    )
    coefficient_table(best_r2_model).to_csv(
        OUTPUT_TABLES / "best_pseudo_r2_model_coefficients.csv", index=False
    )
    model_metrics(best_llr_model).to_csv(
        OUTPUT_TABLES / "best_llr_model_metrics.csv", index=False
    )
    model_metrics(best_r2_model).to_csv(
        OUTPUT_TABLES / "best_pseudo_r2_model_metrics.csv", index=False
    )

    _, cm_reduced, metrics_reduced, holdout_reduced = evaluate_holdout(clean, best_r2_predictors)
    pd.DataFrame(cm_reduced, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        OUTPUT_TABLES / "confusion_matrix_reduced.csv"
    )
    pd.DataFrame([metrics_reduced]).to_csv(
        OUTPUT_TABLES / "holdout_metrics_reduced.csv", index=False
    )
    save_confusion_matrix(
        cm_reduced,
        OUTPUT_FIGURES / "confusion_matrix_reduced.png",
        "Matriz de confusión · mejor modelo reducido",
    )
    save_roc_curve(
        holdout_reduced[3],
        holdout_reduced[4],
        OUTPUT_FIGURES / "roc_curve_reduced.png",
        "Curva ROC · mejor modelo reducido",
    )

    summary = {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "duplicates_removed": int(len(raw) - len(clean)),
        "full_model_pseudo_r2": float(full_model.prsquared),
        "full_model_log_likelihood": float(full_model.llf),
        "full_model_ll_null": float(full_model.llnull),
        "full_model_llr_p_value": float(full_model.llr_pvalue),
        "full_holdout_accuracy": float(metrics_full["accuracy"]),
        "full_holdout_roc_auc": float(metrics_full["roc_auc"]),
        "best_by_llr_pvalue": best_llr_predictors,
        "best_by_pseudo_r2": best_r2_predictors,
        "reduced_holdout_accuracy": float(metrics_reduced["accuracy"]),
        "reduced_holdout_roc_auc": float(metrics_reduced["roc_auc"]),
        "candidate_models_fitted": int(len(search)),
    }
    (OUTPUT_TABLES / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, ensure_ascii=False))
