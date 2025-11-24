#!/usr/bin/env python3
"""
Offline evaluation script for the recruitment-fraud model.

Usage examples (from bluesky-assign3/):
  - Evaluate on the held-out set produced by training:
      python test_model.py --data test_set.csv --binary
  - Evaluate on any CSV with columns text,label (0=legit, 1=suspicious, 2=fraudulent):
      python test_model.py --data cleaned_training_data.csv
"""

import argparse
import os
import pickle
from typing import Tuple, Set

import pandas as pd
from sklearn.metrics import accuracy_score


DEFAULT_DATA = "test_set.csv"
DEFAULT_VECTORIZER = "vectorizer.pkl"
DEFAULT_MODEL = "fraud_model.pkl"
DEFAULT_SPLIT_INDICES = "train_test_split_indices.csv"


def load_data(path: str, binary: bool) -> Tuple[pd.Series, pd.Series]:
	"""
	Load CSV with columns: text, label.
	Optionally collapse to binary labels: 0=legit, 1=recruitment-risk (suspicious or fraudulent).
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Data not found at {path}")
	df = pd.read_csv(path)
	required = {"text", "label"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	texts = df["text"].astype(str)
	labels = df["label"].astype(int)
	if binary:
		labels = labels.map(lambda v: 1 if int(v) in (1, 2) else 0)
	return texts, labels


def load_test_indices(split_path: str) -> Set[int]:
	"""Load indices marked as 'test' from a split indices CSV."""
	if not os.path.exists(split_path):
		raise FileNotFoundError(f"Split indices file not found at {split_path}")
	df = pd.read_csv(split_path)
	if "index" not in df.columns or "split" not in df.columns:
		raise ValueError("Split indices file must contain 'index' and 'split' columns.")
	return set(df[df["split"] == "test"]["index"].astype(int).tolist())


def load_artifacts(vectorizer_path: str, model_path: str):
	"""Load saved vectorizer and model pickles."""
	if not os.path.exists(vectorizer_path):
		raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model not found at {model_path}")
	with open(vectorizer_path, "rb") as f:
		vectorizer = pickle.load(f)
	with open(model_path, "rb") as f:
		model = pickle.load(f)
	return vectorizer, model


def main():
	parser = argparse.ArgumentParser(description="Offline test for recruitment-fraud model")
	parser.add_argument(
		"--data",
		type=str,
		default=DEFAULT_DATA,
		help=f"Path to CSV with text,label columns (default: {DEFAULT_DATA})",
	)
	parser.add_argument(
		"--vectorizer",
		type=str,
		default=DEFAULT_VECTORIZER,
		help=f"Path to saved vectorizer pickle (default: {DEFAULT_VECTORIZER})",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=DEFAULT_MODEL,
		help=f"Path to saved model pickle (default: {DEFAULT_MODEL})",
	)
	parser.add_argument(
		"--binary",
		action="store_true",
		help="Evaluate as binary classification (0=legit, 1=recruitment-risk).",
	)
	parser.add_argument(
		"--split_indices",
		type=str,
		default=DEFAULT_SPLIT_INDICES,
		help=f"Path to train/test split indices CSV (default: {DEFAULT_SPLIT_INDICES})",
	)
	args = parser.parse_args()

	# Load data, ALWAYS using non-training rows for evaluation:
	# - If evaluating the original training CSV (cleaned_training_data.csv), filter to 'test' indices.
	# - Otherwise (e.g., test_set.csv), assume the CSV already contains only test rows.
	base = os.path.basename(args.data)
	if base == "cleaned_training_data.csv":
		if not os.path.exists(args.data):
			raise FileNotFoundError(f"Data not found at {args.data}")
		df = pd.read_csv(args.data)
		if "text" not in df.columns or "label" not in df.columns:
			raise ValueError("Data file must contain 'text' and 'label' columns.")
		test_indices = load_test_indices(args.split_indices)
		df = df.loc[df.index.isin(test_indices)]
		if df.empty:
			raise ValueError("After enforcing test split, no rows remain. Check split file and data alignment.")
		texts = df["text"].astype(str)
		labels = df["label"].astype(int)
		if args.binary:
			labels = labels.map(lambda v: 1 if int(v) in (1, 2) else 0)
	else:
		texts, labels = load_data(args.data, binary=args.binary)

	vectorizer, model = load_artifacts(args.vectorizer, args.model)

	# Vectorize and predict
	x = vectorizer.transform(texts)
	y_pred = model.predict(x)

	# Collapse to binary for overall metrics (positive = recruitment-risk)
	if args.binary:
		y_true = labels.values
		y_hat = y_pred
	else:
		y_true = (labels.values != 0).astype(int)
		y_hat = (y_pred != 0).astype(int)

	# Compute overall metrics across the test set
	tp = int(((y_hat == 1) & (y_true == 1)).sum())
	tn = int(((y_hat == 0) & (y_true == 0)).sum())
	fp = int(((y_hat == 1) & (y_true == 0)).sum())
	fn = int(((y_hat == 0) & (y_true == 1)).sum())

	total = tp + tn + fp + fn
	accuracy = (tp + tn) / total if total > 0 else 0.0
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1: {f1:.4f}")


if __name__ == "__main__":
	main()


