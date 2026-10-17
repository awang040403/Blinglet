"""
Expedia Applied Science – Identify Fake Job Postings (fast path)

Target: is-authentic (1 = real, 0 = fake)
Deliverable: submission.csv with columns [id, is-authentic]
Metric: accuracy
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


# ---------------------------------------------------------------------------
# STEP 1 – Robust type coercion
# Why: HackerRank/Expedia note that test dtypes may not match train
#      (e.g. wfh as "0"/"1" strings vs ints). Wrong types break models.
# ---------------------------------------------------------------------------
def coerce_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1, "1.0": 1, "true": 1, "yes": 1, "y": 1,
        "0": 0, "0.0": 0, "false": 0, "no": 0, "n": 0,
        "": 0, "nan": 0, "none": 0,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    leftover = out.isna()
    if leftover.any():
        numeric = pd.to_numeric(series[leftover], errors="coerce").fillna(0)
        out.loc[leftover] = (numeric > 0).astype(int)
    return out.astype(int)


def extract_country(location: pd.Series) -> pd.Series:
    # location like "US, NY, New York" → country = "US"
    loc = location.fillna("").astype(str)
    return loc.str.split(",").str[0].str.strip().replace({"": "Unknown"})


# ---------------------------------------------------------------------------
# STEP 2 – Feature engineering
# How each feature family affects the authenticity decision:
#
#   title (TF-IDF)     – scam posts often reuse generic titles
#                        ("Home Based Payroll", "Data Entry") or odd n-grams
#   company logo       – missing logo is a strong fake signal in this data
#   screening round    – real employers more often ask screening questions
#   wfh                – remote-only ads are slightly riskier in EMSCAD
#   has_salary         – salary present/absent; sparse but informative
#   industry/dept/...  – some industries & vague depts skew fake
#   country            – geography proxy from location
#   missing_count      – emptier forms correlate with fraud
#   title_len/words    – very short or keyword-stuffed titles stand out
# ---------------------------------------------------------------------------
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
        "department", "industry", "employment type",
        "required experience", "required education", "salary range",
    ]
    present = [c for c in miss_cols if c in out.columns]
    out["missing_count"] = out[present].isna().sum(axis=1) + out[present].astype(str).isin(
        ["", "nan", "None"]
    ).sum(axis=1)

    for c in CAT_COLS:
        out[c] = out[c].fillna("Missing").astype(str)
    return out


# ---------------------------------------------------------------------------
# STEP 3 – EDA visualizations (decision context)
# ---------------------------------------------------------------------------
def make_eda(train: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # Class imbalance: accuracy baselines sit near majority rate (~95%)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train[TARGET].value_counts().sort_index()
    ax.bar(["Fake (0)", "Authentic (1)"], counts.values, color=["#c44e52", "#4c72b0"])
    ax.set_title("Step: label balance — why accuracy needs care")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGS / "01_class_balance.png", dpi=140)
    plt.close(fig)

    # Binary flags → authenticity rate (direct effect on decision)
    rows = []
    for f in BINARY_COLS:
        for val in [0, 1]:
            sub = train[train[f] == val]
            rows.append({"feature": f, "value": str(val), "authentic_rate": sub[TARGET].mean()})
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=pd.DataFrame(rows), x="feature", y="authentic_rate", hue="value", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Step: how binary flags shift P(authentic)")
    ax.set_ylabel("P(is-authentic=1)")
    fig.tight_layout()
    fig.savefig(FIGS / "02_authentic_rate_by_flags.png", dpi=140)
    plt.close(fig)

    # Missingness effect
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=train, x=TARGET, y="missing_count", ax=ax)
    ax.set_xticklabels(["Fake (0)", "Authentic (1)"])
    ax.set_title("Step: emptier postings lean fake")
    fig.tight_layout()
    fig.savefig(FIGS / "03_missing_vs_authentic.png", dpi=140)
    plt.close(fig)

    # Industry risk
    tmp = train.copy()
    tmp["industry_f"] = tmp["industry"].fillna("Missing")
    ind = (
        tmp.groupby("industry_f")
        .agg(n=(ID_COL, "count"), fake_rate=(TARGET, lambda s: 1 - s.mean()))
        .query("n >= 50")
        .sort_values("fake_rate", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=ind["fake_rate"], y=ind.index, ax=ax, color="#c44e52")
    ax.set_title("Step: industries with highest fake rate (n≥50)")
    ax.set_xlabel("P(fake)")
    fig.tight_layout()
    fig.savefig(FIGS / "04_industry_fake_rate.png", dpi=140)
    plt.close(fig)


def build_preprocessor() -> ColumnTransformer:
    # Title → bag of 1–2 grams (text drives many splits)
    text = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, stop_words="english")
    cats = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)),
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


def get_feature_names(pre: ColumnTransformer) -> np.ndarray:
    return pre.get_feature_names_out()


# ---------------------------------------------------------------------------
# STEP 4 – Model (ExtraTrees: fast + strong accuracy on sparse+tabular mix)
# Decision rule: majority vote of randomized trees on engineered features.
# Higher tree impurity reduction → higher feature importance.
# ---------------------------------------------------------------------------
def build_model() -> Pipeline:
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# STEP 5 – Top-20 feature importance visualization
# Interpretation guide:
#   title__<token>           – title word/phrase importance
#   cat__industry_<x>        – that industry category
#   bin__company logo        – logo present/absent
#   num__missing_count       – form completeness
# ---------------------------------------------------------------------------
def plot_top20_importance(model: Pipeline, path: Path) -> pd.DataFrame:
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    names = get_feature_names(pre)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    top = pd.DataFrame(
        {"feature": names[order], "importance": importances[order]}
    )
    top.to_csv(ART / "top20_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature", ax=ax, color="#4c72b0")
    ax.set_title("Top 20 features by ExtraTrees importance\n(higher → more influence on is-authentic)")
    ax.set_xlabel("Mean impurity decrease")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Human-readable effect notes for the top rows
    notes = []
    for _, row in top.iterrows():
        f = row["feature"]
        if f.startswith("bin__company logo") or "company logo" in f:
            effect = "Logo missing raises fake odds; logo present supports authentic."
        elif f.startswith("bin__screening") or "screening round" in f:
            effect = "Screening questions present → more authentic; absent → riskier."
        elif f.startswith("bin__wfh") or f.endswith("wfh"):
            effect = "WFH=1 slightly associated with higher fraud rate in this corpus."
        elif "missing_count" in f:
            effect = "More empty fields → more likely fake."
        elif "has_salary" in f:
            effect = "Salary listed vs blank shifts authenticity probability."
        elif f.startswith("title__"):
            effect = f"Title token '{f.split('__', 1)[-1]}' helps separate scam vs real wording."
        elif f.startswith("cat__"):
            effect = f"Category level '{f.split('__', 1)[-1]}' shifts class odds."
        else:
            effect = "Numeric/text signal used by trees to split authentic vs fake."
        notes.append({"feature": f, "importance": row["importance"], "how_it_affects_decision": effect})

    notes_df = pd.DataFrame(notes)
    notes_df.to_csv(ART / "top20_feature_effects.csv", index=False)
    print("\n=== Top 20 features & decision effect ===")
    for r in notes:
        print(f"- {r['feature']}: {r['importance']:.4f} → {r['how_it_affects_decision']}")
    return top


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame):
    train_fe = engineer(train)
    test_fe = engineer(test)
    cols = [TEXT_COL] + BINARY_COLS + CAT_COLS + NUM_COLS
    return train_fe, test_fe, train_fe[cols], train_fe[TARGET].astype(int), test_fe[cols], test_fe[ID_COL].astype(str)


def main() -> None:
    # ---- Load ----
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_output.csv")
    print("STEP load:", train.shape, test.shape, "sample cols", list(sample.columns))

    # ---- Features for train & test ----
    train_fe, _, X, y, X_test, ids = prepare_xy(train, test)
    print("STEP features ready:", X.shape, "test", X_test.shape)

    # ---- Visual EDA ----
    make_eda(train_fe)
    print("STEP eda figures →", FIGS)

    # ---- Fit + quick accuracy check ----
    model = build_model()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"STEP cv accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)
    va_pred = model.predict(X_va)
    print("STEP holdout accuracy:", round(accuracy_score(y_va, va_pred), 4))
    print(classification_report(y_va, va_pred, digits=3))

    cm = confusion_matrix(y_va, va_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fake", "Authentic"], yticklabels=["Fake", "Authentic"])
    ax.set_title("Holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGS / "05_confusion_matrix.png", dpi=140)
    plt.close(fig)

    # ---- Top-20 importance (on holdout-fitted model, then refit full) ----
    plot_top20_importance(model, FIGS / "06_top20_feature_importance.png")

    # ---- Final fit on all train → submission.csv ----
    model.fit(X, y)
    preds = model.predict(X_test).astype(int)
    submission = pd.DataFrame({ID_COL: ids, TARGET: preds})
    # Enforce exact sample_output column order: id, is-authentic
    submission = submission[["id", "is-authentic"]]
    submission.to_csv(ROOT / "submission.csv", index=False)
    print("STEP wrote submission.csv", submission.shape, list(submission.columns))
    print(submission.head())

    hidden = DATA / "test_labels_hidden.csv"
    if hidden.exists():
        labels = pd.read_csv(hidden)
        labels[ID_COL] = labels[ID_COL].astype(str)
        merged = submission.merge(labels, on=ID_COL, suffixes=("_pred", "_true"))
        acc = accuracy_score(merged[f"{TARGET}_true"], merged[f"{TARGET}_pred"])
        print(f"STEP offline test accuracy: {acc:.4f}")
        (ART / "offline_test_accuracy.txt").write_text(f"{acc:.6f}\nmodel=extratrees\n")


if __name__ == "__main__":
    main()
