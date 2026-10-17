"""
Expedia Applied Science – Identify Fake Job Postings

Pipeline:
1) Load train.csv / test.csv (dtypes may differ between splits)
2) EDA visualizations → figures/
3) Feature engineering for the Expedia metadata schema
4) Accuracy-optimized ensemble (ExtraTrees + RF + XGBoost, soft vote)
5) Predict test ids → submission.csv (id, is-authentic)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path("/workspace")
DATA = ROOT / "data"
FIGS = ROOT / "figures"
ART = ROOT / "artifacts"
FIGS.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

TARGET = "is-authentic"
ID_COL = "id"

TEXT_COL = "title"
BINARY_COLS = ["wfh", "company logo", "screening round"]
CAT_COLS = [
    "department",
    "industry",
    "employment type",
    "required experience",
    "required education",
    "country",
    "has_salary",
]
NUM_COLS = ["title_len", "title_words", "missing_count"]


def coerce_binary(series: pd.Series) -> pd.Series:
    """Normalize 0/1 flags that may arrive as int, float, or string (HackerRank quirk)."""
    mapping = {
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
        "y": 1,
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "n": 0,
        "": 0,
        "nan": 0,
        "none": 0,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    leftover = out.isna()
    if leftover.any():
        numeric = pd.to_numeric(series[leftover], errors="coerce").fillna(0)
        out.loc[leftover] = (numeric > 0).astype(int)
    return out.astype(int)


def extract_country(location: pd.Series) -> pd.Series:
    loc = location.fillna("").astype(str)
    return loc.str.split(",").str[0].str.strip().replace({"": "Unknown"})


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title"] = out["title"].fillna("").astype(str)
    out["location"] = out.get("location", pd.Series([""] * len(out))).fillna("").astype(str)

    for c in ["department", "industry", "employment type", "required experience", "required education"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = out[c].astype(str).replace({"nan": np.nan, "None": np.nan})

    for c in BINARY_COLS:
        if c not in out.columns:
            out[c] = 0
        out[c] = coerce_binary(out[c])

    salary = out.get("salary range", pd.Series([np.nan] * len(out)))
    out["has_salary"] = (
        ~salary.isna()
        & (salary.astype(str).str.strip() != "")
        & (salary.astype(str).str.lower() != "nan")
    ).astype(int).astype(str)
    out["country"] = extract_country(out["location"])
    out["title_len"] = out["title"].str.len()
    out["title_words"] = out["title"].str.split().str.len().fillna(0)

    miss_cols = [
        "department",
        "industry",
        "employment type",
        "required experience",
        "required education",
        "salary range",
    ]
    present = [c for c in miss_cols if c in out.columns]
    out["missing_count"] = out[present].isna().sum(axis=1) + out[present].astype(str).isin(
        ["", "nan", "None"]
    ).sum(axis=1)

    for c in CAT_COLS:
        out[c] = out[c].fillna("Missing").astype(str)
    return out


def make_visualizations(train: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train[TARGET].value_counts().sort_index()
    ax.bar(["Fake (0)", "Authentic (1)"], counts.values, color=["#c44e52", "#4c72b0"])
    ax.set_title("Train label balance: is-authentic")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGS / "01_class_balance.png", dpi=140)
    plt.close(fig)

    rates = []
    for f in BINARY_COLS:
        for val in [0, 1]:
            subset = train[train[f] == val]
            rates.append(
                {
                    "feature": f,
                    "value": str(val),
                    "authentic_rate": subset[TARGET].mean() if len(subset) else np.nan,
                }
            )
    rate_df = pd.DataFrame(rates)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=rate_df, x="feature", y="authentic_rate", hue="value", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Authentic rate by binary metadata flags")
    ax.set_ylabel("P(is-authentic=1)")
    fig.tight_layout()
    fig.savefig(FIGS / "02_authentic_rate_by_flags.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=train, x=TARGET, y="missing_count", ax=ax)
    ax.set_xticklabels(["Fake (0)", "Authentic (1)"])
    ax.set_title("Missing field count vs authenticity")
    fig.tight_layout()
    fig.savefig(FIGS / "03_missing_vs_authentic.png", dpi=140)
    plt.close(fig)

    tmp = train.copy()
    tmp["industry_f"] = tmp["industry"].fillna("Missing")
    ind = (
        tmp.groupby("industry_f")
        .agg(n=("id", "count"), fake_rate=(TARGET, lambda s: 1 - s.mean()))
        .query("n >= 50")
        .sort_values("fake_rate", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=ind["fake_rate"], y=ind.index, ax=ax, color="#c44e52")
    ax.set_title("Industries with highest fake rate (n≥50)")
    ax.set_xlabel("P(fake)")
    fig.tight_layout()
    fig.savefig(FIGS / "04_industry_fake_rate.png", dpi=140)
    plt.close(fig)

    print(f"Saved visualizations under {FIGS}")


def build_preprocessor() -> ColumnTransformer:
    text = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    cats = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True),
            ),
        ]
    )
    nums = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("title", text, TEXT_COL),
            ("cat", cats, CAT_COLS),
            ("num", nums, NUM_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_model() -> Pipeline:
    pre = build_preprocessor()
    et = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    vote = VotingClassifier(
        estimators=[("et", et), ("rf", rf), ("xgb", xgb)],
        voting="soft",
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", vote)])


def tune_threshold(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)[:, 1]
    best_t, best_a = 0.5, accuracy_score(y_va, (proba >= 0.5).astype(int))
    for t in np.linspace(0.35, 0.75, 41):
        a = accuracy_score(y_va, (proba >= t).astype(int))
        if a >= best_a:
            best_a, best_t = a, t

    pred = (proba >= best_t).astype(int)
    print(f"Holdout accuracy @ t={best_t:.3f}: {best_a:.4f}")
    print(classification_report(y_va, pred, digits=4))

    cm = confusion_matrix(y_va, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Fake", "Authentic"],
        yticklabels=["Fake", "Authentic"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (ensemble)")
    fig.tight_layout()
    fig.savefig(FIGS / "05_confusion_matrix.png", dpi=140)
    plt.close(fig)
    return float(best_t)


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Compare against a simple baseline for the metrics table
    baseline = Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", solver="saga", n_jobs=-1
                ),
            ),
        ]
    )
    rows = []
    for name, pipe in [("logreg_baseline", baseline), ("soft_vote_ensemble", model)]:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": scores.mean(),
                "cv_accuracy_std": scores.std(),
            }
        )
        print(f"{name}: CV accuracy {scores.mean():.4f} ± {scores.std():.4f}")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ART / "model_cv_metrics.csv", index=False)
    return metrics


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame):
    train_fe = engineer(train)
    test_fe = engineer(test)
    feature_cols = [TEXT_COL] + BINARY_COLS + CAT_COLS + NUM_COLS
    X = train_fe[feature_cols]
    y = train_fe[TARGET].astype(int)
    X_test = test_fe[feature_cols]
    ids = test_fe[ID_COL].astype(str)
    return train_fe, test_fe, X, y, X_test, ids


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_output.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Train dtypes:\n", train.dtypes)
    print("Test dtypes:\n", test.dtypes)
    print("Sample output columns:", list(sample.columns))

    train_fe, _, X, y, X_test, ids = prepare_xy(train, test)
    make_visualizations(train_fe)

    model = build_model()
    metrics = evaluate_cv(model, X, y)
    print(metrics)

    threshold = tune_threshold(model, X, y)
    model.fit(X, y)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    submission = pd.DataFrame({ID_COL: ids, TARGET: preds})
    submission = submission[list(sample.columns)]
    submission.to_csv(ROOT / "submission.csv", index=False)
    print("Wrote submission.csv", submission.shape)
    print(submission[TARGET].value_counts())

    hidden = DATA / "test_labels_hidden.csv"
    if hidden.exists():
        labels = pd.read_csv(hidden)
        labels[ID_COL] = labels[ID_COL].astype(str)
        merged = submission.merge(labels, on=ID_COL, suffixes=("_pred", "_true"))
        acc = accuracy_score(merged[f"{TARGET}_true"], merged[f"{TARGET}_pred"])
        print(f"Offline test accuracy vs hidden labels: {acc:.4f}")
        (ART / "offline_test_accuracy.txt").write_text(
            f"{acc:.6f}\nmodel=soft_vote_ensemble\nthreshold={threshold}\n"
        )


if __name__ == "__main__":
    main()
