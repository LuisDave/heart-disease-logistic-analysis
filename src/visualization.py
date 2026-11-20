from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import TARGET


def save_target_balance(df: pd.DataFrame, path) -> None:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["0", "1"], counts.values)
    ax.set_title("Distribución de la variable objetivo")
    ax.set_xlabel("target")
    ax.set_ylabel("Observaciones")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 2, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_correlation_matrix(df: pd.DataFrame, path) -> None:
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", TARGET]
    corr = df[continuous].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlación entre variables cuantitativas seleccionadas")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(cm, path, title="Matriz de confusión") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_roc_curve(y_test, proba, path, title="Curva ROC") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_logistic_curve_oldpeak(df: pd.DataFrame, path) -> None:
    """
    Univariate logistic curve for visualization.

    The multivariable model cannot be represented as a single 2D sigmoid without
    conditioning on the remaining variables. This plot deliberately uses the
    continuous predictor oldpeak to show the observed binary outcomes and its
    marginal logistic relationship with target.
    """
    model = smf.logit("target ~ oldpeak", data=df).fit(disp=False)
    x_grid = np.linspace(df["oldpeak"].min(), df["oldpeak"].max(), 250)
    pred = model.predict(pd.DataFrame({"oldpeak": x_grid}))

    rng = np.random.default_rng(2026)
    jitter = rng.normal(0, 0.018, size=len(df))
    y_jittered = np.clip(df[TARGET].to_numpy() + jitter, -0.05, 1.05)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(df["oldpeak"], y_jittered, alpha=0.35, s=20, label="Observaciones (jitter)")
    ax.plot(x_grid, pred, linewidth=2.5, label="Probabilidad logística estimada")
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("oldpeak")
    ax.set_ylabel("target / P(target=1)")
    ax.set_title("Relación logística univariada entre oldpeak y target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_selection_top10(results: pd.DataFrame, path) -> None:
    top = results.head(10).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(top["predictors"], top["pseudo_r2_mcfadden"])
    ax.set_xlabel("Pseudo R² de McFadden")
    ax.set_ylabel("Predictores")
    ax.set_title("Mejores modelos reducidos (máximo 4 variables)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
