"""Prepare EMSCAD fake-job data in Expedia Applied Science assessment schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("/tmp/fake_job_postings.csv")
OUT = Path("/workspace/data")
OUT.mkdir(parents=True, exist_ok=True)

RENAME = {
    "job_id": "id",
    "salary_range": "salary range",
    "telecommuting": "wfh",
    "has_company_logo": "company logo",
    "has_questions": "screening round",
    "employment_type": "employment type",
    "required_experience": "required experience",
    "required_education": "required education",
}

FEATURE_COLS = [
    "id",
    "title",
    "location",
    "department",
    "industry",
    "salary range",
    "wfh",
    "company logo",
    "screening round",
    "employment type",
    "required experience",
    "required education",
]


def main() -> None:
    df = pd.read_csv(RAW)
    df = df.rename(columns=RENAME)
    # Expedia target: 1 = authentic, 0 = fake
    df["is-authentic"] = (1 - df["fraudulent"]).astype(int)
    df = df[FEATURE_COLS + ["is-authentic"]].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["is-authentic"],
    )

    train_out = train_df.reset_index(drop=True)
    test_labels = test_df[["id", "is-authentic"]].reset_index(drop=True)
    test_out = test_df.drop(columns=["is-authentic"]).reset_index(drop=True)

    # Mimic HackerRank warning: test column dtypes may differ from train
    for col in ["wfh", "company logo", "screening round"]:
        test_out[col] = test_out[col].astype(str)
    test_out["id"] = test_out["id"].astype(str)

    train_out.to_csv(OUT / "train.csv", index=False)
    test_out.to_csv(OUT / "test.csv", index=False)
    test_labels.to_csv(OUT / "test_labels_hidden.csv", index=False)

    sample = test_out[["id"]].copy()
    sample["is-authentic"] = 1
    sample.to_csv(OUT / "sample_output.csv", index=False)

    print("Wrote", OUT / "train.csv", train_out.shape)
    print("Wrote", OUT / "test.csv", test_out.shape)
    print("Class balance (train):")
    print(train_out["is-authentic"].value_counts(normalize=True).round(4))
    print("sample_output columns:", list(sample.columns))


if __name__ == "__main__":
    main()
