#!/usr/bin/env python3
"""
Evaluation script for policy proposal labeler
Calculates accuracy, precision, recall, and F1 score
"""

import argparse
import json
import os
import pandas as pd
from atproto import Client
from dotenv import load_dotenv
from pylabel import PolicyProposalLabeler, label_post, did_from_handle

load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")


def calculate_metrics(predictions, ground_truth):
    """
    Calculate overall accuracy, precision, recall, and F1 score
    treating positive as ANY risk label ('fraudulent-recruitment' OR 'suspicious-recruitment').
    """
    tp = fp = fn = tn = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_set = set(pred)
        truth_set = set(truth)
        pred_pos = ("fraudulent-recruitment" in pred_set) or ("suspicious-recruitment" in pred_set)
        truth_pos = ("fraudulent-recruitment" in truth_set) or ("suspicious-recruitment" in truth_set)
        if pred_pos and truth_pos:
            tp += 1
        elif pred_pos and not truth_pos:
            fp += 1
        elif (not pred_pos) and truth_pos:
            fn += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total,
    }


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate policy proposal labeler')
    parser.add_argument('test_data', type=str, help='Path to test data CSV file')
    parser.add_argument('--emit_labels', action='store_true', help='Actually emit labels to Bluesky')
    args = parser.parse_args()
    
    # Initialize client and labeler
    print("Initializing Bluesky client...")
    client = Client()
    client.login(USERNAME, PW)
    
    labeler = PolicyProposalLabeler(client)
    
    # Set up labeler client if emitting labels
    labeler_client = None
    if args.emit_labels:
        try:
            did = did_from_handle(USERNAME)
            labeler_client = client.with_proxy("atproto_labeler", did)
            print("✓ Labeler client initialized for label emission")
        except Exception as e:
            print(f"⚠️  Warning: Could not set up labeler client: {e}")
            print("   Labels will not be emitted. Continuing with evaluation only.")
            args.emit_labels = False
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    print(f"Evaluating on {len(test_df)} posts...")
    print("-" * 70)
    
    predictions = []
    ground_truth = []
    errors = []
    
    for idx, row in test_df.iterrows():
        url = row['URL']
        expected_labels = json.loads(row['Labels'])
        
        try:
            predicted_labels = labeler.moderate_post(url)
            predictions.append(predicted_labels)
            ground_truth.append(expected_labels)
            
            # Emit labels if requested
            if args.emit_labels and labeler_client and len(predicted_labels) > 0:
                try:
                    label_post(client, labeler_client, url, predicted_labels)
                    print(f"✓ Emitted labels {predicted_labels} to {url}")
                except Exception as e:
                    print(f"⚠️  Failed to emit labels to {url}: {e}")
        except Exception as e:
            errors.append((url, str(e)))
            print(f"Error processing {url}: {e}")
            predictions.append([])
            ground_truth.append(expected_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)
    
    # Print overall results only
    print("\n" + "=" * 70)
    print("OVERALL EVALUATION (Positive = any risk label)")
    print("=" * 70)
    print(f"Total posts evaluated: {metrics['total']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("\n" + "=" * 70)
    
    return metrics


if __name__ == "__main__":
    main()

