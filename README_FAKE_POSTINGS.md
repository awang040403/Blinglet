# Expedia Applied Science – Identify Fake Job Postings

Binary classification of job ads as **authentic (`is-authentic=1`)** or **fake (`is-authentic=0`)**, evaluated primarily on **accuracy**.

## Data schema (test)

`id`, `title`, `location`, `department`, `industry`, `salary range`, `wfh`, `company logo`, `screening round`, `employment type`, `required experience`, `required education`

Train also includes the label `is-authentic`. Test dtypes may differ from train (int vs string flags) — preprocessing coerces them.

## Deliverable

`submission.csv` with columns matching `sample_output.csv`:

```text
id,is-authentic
...
```

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # builds data/ from EMSCAD (demo); skip if HackerRank files already present
python solution.py       # EDA figures + model + submission.csv
```

## Model

- Features: TF-IDF on `title`, categorical one-hots, binary flags, missingness / salary present / title length
- Ensemble: ExtraTrees + RandomForest + XGBoost (soft vote)
- Decision threshold tuned on a stratified holdout for accuracy
- Figures written to `figures/`
