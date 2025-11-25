# Assignment 3: Online Recruitment Fraud Detection Labeler

## Group Information
- Group Members: 
Grace Myers
Eva Huang
Xinyi Huang
Jiayu Zhang
- Policy Proposal: Online Recruitment Fraud Detection

## Overview
This project provides a Bluesky labeler that detects online recruitment fraud by labeling posts as:
- `fraudulent-recruitment`
- `suspicious-recruitment`
- no label (legit)

It includes:
- A training script that learns from labeled text (`cleaned_training_data.csv`)
- A labeler that fetches real posts via the Bluesky API and predicts labels
- An evaluator that reports accuracy, precision, recall, and F1

## Setup
1) Python and env
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Recommended: Python 3.12 (some libs are not 3.14-ready).

2) Environment variables
Create `.env` in `bluesky-assign3/`:
```
USERNAME=your_bluesky_username
PW=your_bluesky_password
```

## Training (Step 1)
Input file: `cleaned_training_data.csv` with columns:
- `text`: post text
- `label`: 0=legit, 1=suspicious, 2=fraudulent

Script: `train_model.py`

Flags:
- `--data`: Path to training CSV (default: `cleaned_training_data.csv`)
- `--out_dir`: Directory to save artifacts (default: `.`)
- `--data_used_for_training`: Fraction in (0,1). If >0, randomly sample that fraction of rows for training (stratified when possible); the rest form the test set. Overrides `--test_count`.
- `--test_count`: If `--data_used_for_training` is 0, hold out this many rows for test (stratified when possible). Default 100.
- `--binary`: Collapse labels 1 and 2 into a single positive class (1=recruitment-risk, 0=legit).

Artifacts saved:
- `vectorizer.pkl`
- `fraud_model.pkl`
- `train_test_split_indices.csv`
- `train_set.csv`, `test_set.csv`

Run (examples):
Binary, train on 75% of data (random stratified), test on the rest:
```bash
python train_model.py --data cleaned_training_data.csv --out_dir . --data_used_for_training 0.75 --binary
```

Console output includes validation accuracy and a classification report (precision/recall/F1).

## Testing (Step 2)
You can test the trained model in two separate ways. Choose one or run both:

1) Offline evaluation on held-out CSV (fast iteration)

Use the artifacts from training to evaluate a CSV that has `text,label`:

```bash
# Evaluate on the held-out test split produced by training
```
python test_model.py --data test_set.csv --binary

Accuracy: 0.8293
Precision: 0.8182
Recall: 0.6429
F1: 0.7200
```

Outputs:
- Accuracy
- Precision/recall/F1 (binary or 3-class, matching your choice)

2) Live evaluation on real posts (recommended for final reporting)

Script: `evaluate_labeler.py`

Input CSV format:
- `URL`: Bluesky post URL
- `Labels`: JSON array of expected labels, e.g., `[]`, `["suspicious-recruitment"]`, `["fraudulent-recruitment"]`

Run:
```bash
python evaluate_labeler.py data_real_only_v2_prepared.csv
```

What it does:
- Logs into Bluesky with `USERNAME`/`PW`
- Fetches each post by URL
- Runs the labeler and compares predicted vs expected
- Prints:
  - Overall accuracy
  - Precision/recall/F1 for `fraudulent-recruitment` and `suspicious-recruitment`
  - Mismatched rows (for debugging)

Optional:
```bash
python evaluate_labeler.py data_real_only_v2_prepared.csv --emit_labels
```
This will emit labels via the labeler account (use only if you intend to publish labels).

2) Offline sanity-check on the held-out test split

After training, review `test_set.csv` (created by `train_model.py`) for the exact examples used in validation. The training script already prints validation accuracy and a classification report for this split. Use this for quick iteration; use the live evaluation above for end-to-end results.

## Notes and Tips
- If some posts cannot be fetched (deleted/private), predictions may be empty for those rows.
- For better recall, train with more “hard negatives” (legit posts that look scammy) and clear positives (fees/off-platform/unrealistic pay).
- You can switch between 3-class and binary training via `--binary`, then evaluate with the same evaluator.

## Ethical Considerations
- False positives can harm legitimate posters; false negatives can expose users to scams.
- We use “suspicious” vs “fraudulent” to convey confidence, not certainty.
- Adjust thresholds and training data to balance safety and precision.

