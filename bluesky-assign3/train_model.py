#!/usr/bin/env python3
"""
Train recruitment-fraud labeler on cleaned_training_data.csv and save model artifacts.

Inputs:
  - cleaned_training_data.csv (default) with columns:
      text: post text
      label: integer class {0=legit, 1=suspicious, 2=fraudulent}
Outputs:
  - vectorizer.pkl
  - fraud_model.pkl
"""

import argparse
import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score


DEFAULT_DATA = "cleaned_training_data.csv"
VECTORIZER_OUT = "vectorizer.pkl"
MODEL_OUT = "fraud_model.pkl"


def load_data(path: str) -> Tuple[pd.Series, pd.Series]:
	"""Load training CSV and return (texts, labels)."""
	# Ensure the training file exists where the user expects it
	if not os.path.exists(path):
		raise FileNotFoundError(f"Training data not found at {path}")
	# Read the CSV; we expect exactly two columns: text (str) and label (int)
	df = pd.read_csv(path)
	required = {"text", "label"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	# Cast text to string and labels to int for downstream model compatibility
	texts = df["text"].astype(str)
	labels = df["label"].astype(int)
	return texts, labels


def stratified_fixed_train(texts: pd.Series, labels: pd.Series, train_count: int, random_state: int = 42):
	"""
	Create a training set of size data_used_for_training via random sampling (stratified when possible).
	The remainder is used as test set.
	"""
	n_samples = len(texts)
	if train_count <= 0 or train_count >= n_samples:
		raise ValueError(f"train_count must be in (0, {n_samples}), got {train_count}")
	try:
		# Stratified random selection of a fixed number of training samples
		split = StratifiedShuffleSplit(n_splits=1, train_size=train_count, random_state=random_state)
		for train_idx, test_idx in split.split(texts, labels):
			return train_idx, test_idx
	except Exception:
		# Fallback: non-stratified selection if stratification fails
		rs = ShuffleSplit(n_splits=1, train_size=train_count, random_state=random_state)
		for train_idx, test_idx in rs.split(texts):
			return train_idx, test_idx
	raise RuntimeError("Failed to create a train/test split")


def train(texts: pd.Series, labels: pd.Series, data_used_for_training: int, binary: bool = False):
	"""Train TF-IDF + Logistic Regression with a fixed-size test set.
	If binary=True, collapse labels {1,2} -> 1 and 0 -> 0.
	"""
	# Optionally collapse to a binary problem: 0 = legit, 1 = (suspicious OR fraudulent)
	if binary:
		labels = labels.map(lambda v: 1 if int(v) in (1, 2) else 0)

	# Choose split strategy:
	# - Randomly sample N rows for training (stratified when possible), remaining used for test
	train_idx, test_idx = stratified_fixed_train(texts, labels, train_count=data_used_for_training, random_state=42)
	# Partition text and labels into train and validation folds
	x_train, x_val = texts.iloc[train_idx], texts.iloc[test_idx]
	y_train, y_val = labels.iloc[train_idx], labels.iloc[test_idx]

	# Vectorize text using word n-grams (1-2), remove very rare terms (min_df=2)
	# and very frequent terms (max_df=0.98). Stop words are removed to reduce noise.
	vectorizer = TfidfVectorizer(
		lowercase=True,
		stop_words="english",
		ngram_range=(1, 2),
		min_df=2,
		max_df=0.98,
	)
	# Fit the vectorizer on training text and transform both splits
	x_train_vec = vectorizer.fit_transform(x_train)
	x_val_vec = vectorizer.transform(x_val)

	# Train a logistic regression classifier:
	# - max_iter increased for convergence on sparse, high-dimensional features
	# - class_weight='balanced' to mitigate class imbalance
	model = LogisticRegression(
		max_iter=2000,
		multi_class="auto",
		class_weight="balanced",
	)
	model.fit(x_train_vec, y_train)

	# Training-set summary ONLY (as requested)
	# Evaluate the model against the TRAINING split and report metrics
	y_pred_train = model.predict(x_train_vec)
	print("\n" + "=" * 70)
	print("TRAINING SET PERFORMANCE (evaluated on training split)")
	print("This is the accuracy, precision, recall, and f1 score used against the training set")
	print("=" * 70)
	if binary:
		from sklearn.metrics import classification_report
		acc_train = accuracy_score(y_train, y_pred_train)
		print(f"Accuracy: {acc_train:.4f}")
		print("\nClassification report (binary: 0=legit, 1=recruitment-risk):")
		print(classification_report(y_train, y_pred_train, target_names=["legit", "recruitment-risk"], digits=3))
	else:
		from sklearn.metrics import classification_report
		acc_train = accuracy_score(y_train, y_pred_train)
		print(f"Accuracy: {acc_train:.4f}")
		print("\nClassification report (labels: 0=legit, 1=suspicious, 2=fraudulent):")
		print(classification_report(y_train, y_pred_train, digits=3))

	return vectorizer, model, train_idx, test_idx


def save_artifacts(vectorizer, model, out_dir: str):
	"""Persist model artifacts to the specified directory."""
	# Ensure the output directory exists
	os.makedirs(out_dir, exist_ok=True)
	# Save vectorizer and model as pickles for later inference in the labeler
	vec_path = os.path.join(out_dir, VECTORIZER_OUT)
	mod_path = os.path.join(out_dir, MODEL_OUT)
	with open(vec_path, "wb") as f:
		pickle.dump(vectorizer, f)
	with open(mod_path, "wb") as f:
		pickle.dump(model, f)
	print(f"Saved vectorizer to {vec_path}")
	print(f"Saved model to {mod_path}")


def main():
	parser = argparse.ArgumentParser(description="Train recruitment-fraud model")
	# Path to the input training CSV (defaults to cleaned_training_data.csv at repo root)
	parser.add_argument(
		"--data",
		type=str,
		default=DEFAULT_DATA,
		help="Path to training CSV (default: cleaned_training_data.csv)",
	)
	# Where to write model artifacts (vectorizer.pkl, fraud_model.pkl) and split CSVs
	parser.add_argument(
		"--out_dir",
		type=str,
		default=".",
		help="Directory to write model artifacts (default: project root)",
	)
	# Specify fraction of rows to use for training in [0,1] (randomly sampled, stratified when possible).
	# If > 0, overrides --test_count.
	parser.add_argument(
		"--data_used_for_training",
		type=float,
		default=0,
		help="If >0, fraction in [0,1] of rows to use for training (random, stratified when possible); rest used for test. Overrides --test_count.",
	)
	# Train a binary classifier instead of 3-class:
	# labels 1 and 2 collapse into a single positive class ("recruitment-risk")
	parser.add_argument(
		"--binary",
		action="store_true",
		help="Train a binary classifier by collapsing labels 1 and 2 into a single positive class.",
	)
	args = parser.parse_args()

	# 1) Load raw text and labels from CSV
	texts, labels = load_data(args.data)
	# Compute absolute train size from fraction if provided
	n_samples = len(texts)
	fraction = args.data_used_for_training if (args.data_used_for_training and args.data_used_for_training > 0) else 0.75
	if not (0 < fraction < 1):
		raise ValueError("--data_used_for_training must be a fraction in (0,1)")
	train_count_from_frac = int(round(n_samples * fraction))
	# Ensure at least 1 in train and at least 1 in test
	train_count_from_frac = max(1, min(train_count_from_frac, n_samples - 1))
	# 2) Create the requested split, vectorize text, and train the classifier
	vectorizer, model, train_idx, test_idx = train(
		texts, labels, data_used_for_training=train_count_from_frac, binary=args.binary
	)
	# 3) Persist model artifacts for downstream labeler inference
	save_artifacts(vectorizer, model, args.out_dir)

	# Save splits for reproducibility
	# (a) Save index membership for train vs. test
	split_path = os.path.join(args.out_dir, "train_test_split_indices.csv")
	pd.DataFrame({
		"index": list(train_idx) + list(test_idx),
		"split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
	}).to_csv(split_path, index=False)
	print(f"Saved split indices to {split_path}")

	# Also save explicit CSVs
	# (b) Save the actual records for train and test so users can inspect or reuse them
	df_all = pd.DataFrame({"text": texts, "label": labels})
	train_csv = os.path.join(args.out_dir, "train_set.csv")
	test_csv = os.path.join(args.out_dir, "test_set.csv")
	df_all.iloc[train_idx].to_csv(train_csv, index=False)
	df_all.iloc[test_idx].to_csv(test_csv, index=False)
	print(f"Saved train set to {train_csv}")
	print(f"Saved test set to {test_csv}")


if __name__ == "__main__":
	main()


