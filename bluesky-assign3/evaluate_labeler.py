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
    Calculate accuracy, precision, recall, and F1 score
    
    Args:
        predictions: List of predicted label lists
        ground_truth: List of ground truth label lists
        
    Returns:
        dict with metrics
    """
    # Flatten labels for binary classification
    # We'll treat it as: fraudulent-recruitment vs others
    tp_fraud = 0  # True positives for fraudulent
    fp_fraud = 0  # False positives for fraudulent
    fn_fraud = 0  # False negatives for fraudulent
    tn_fraud = 0  # True negatives for fraudulent
    
    tp_susp = 0  # True positives for suspicious
    fp_susp = 0  # False positives for suspicious
    fn_susp = 0  # False negatives for suspicious
    
    correct_exact = 0  # Exact match (all labels match)
    
    for pred, truth in zip(predictions, ground_truth):
        pred_set = set(pred)
        truth_set = set(truth)
        
        # Exact match
        if pred_set == truth_set:
            correct_exact += 1
        
        # Binary classification for fraudulent
        pred_fraud = "fraudulent-recruitment" in pred_set
        truth_fraud = "fraudulent-recruitment" in truth_set
        
        if pred_fraud and truth_fraud:
            tp_fraud += 1
        elif pred_fraud and not truth_fraud:
            fp_fraud += 1
        elif not pred_fraud and truth_fraud:
            fn_fraud += 1
        else:
            tn_fraud += 1
        
        # Binary classification for suspicious
        pred_susp = "suspicious-recruitment" in pred_set
        truth_susp = "suspicious-recruitment" in truth_set
        
        if pred_susp and truth_susp:
            tp_susp += 1
        elif pred_susp and not truth_susp:
            fp_susp += 1
        elif not pred_susp and truth_susp:
            fn_susp += 1
    
    # Calculate metrics for fraudulent
    precision_fraud = tp_fraud / (tp_fraud + fp_fraud) if (tp_fraud + fp_fraud) > 0 else 0
    recall_fraud = tp_fraud / (tp_fraud + fn_fraud) if (tp_fraud + fn_fraud) > 0 else 0
    f1_fraud = 2 * (precision_fraud * recall_fraud) / (precision_fraud + recall_fraud) if (precision_fraud + recall_fraud) > 0 else 0
    
    # Calculate metrics for suspicious
    precision_susp = tp_susp / (tp_susp + fp_susp) if (tp_susp + fp_susp) > 0 else 0
    recall_susp = tp_susp / (tp_susp + fn_susp) if (tp_susp + fn_susp) > 0 else 0
    f1_susp = 2 * (precision_susp * recall_susp) / (precision_susp + recall_susp) if (precision_susp + recall_susp) > 0 else 0
    
    # Overall accuracy (exact match)
    accuracy = correct_exact / len(predictions) if len(predictions) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'fraudulent': {
            'precision': precision_fraud,
            'recall': recall_fraud,
            'f1': f1_fraud,
            'tp': tp_fraud,
            'fp': fp_fraud,
            'fn': fn_fraud,
            'tn': tn_fraud,
        },
        'suspicious': {
            'precision': precision_susp,
            'recall': recall_susp,
            'f1': f1_susp,
            'tp': tp_susp,
            'fp': fp_susp,
            'fn': fn_susp,
        },
        'total': len(predictions),
        'exact_matches': correct_exact,
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
            
            # Print mismatches
            if set(predicted_labels) != set(expected_labels):
                print(f"\nMismatch at row {idx + 1}:")
                print(f"  URL: {url}")
                print(f"  Expected: {expected_labels}")
                print(f"  Predicted: {predicted_labels}")
        except Exception as e:
            errors.append((url, str(e)))
            print(f"Error processing {url}: {e}")
            predictions.append([])
            ground_truth.append(expected_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal posts evaluated: {metrics['total']}")
    print(f"Exact matches: {metrics['exact_matches']}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print(f"\n{'─' * 70}")
    print("FRAUDULENT-RECRUITMENT Detection:")
    print(f"{'─' * 70}")
    print(f"  Precision: {metrics['fraudulent']['precision']:.4f}")
    print(f"  Recall:    {metrics['fraudulent']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['fraudulent']['f1']:.4f}")
    print(f"  TP: {metrics['fraudulent']['tp']}, FP: {metrics['fraudulent']['fp']}, "
          f"FN: {metrics['fraudulent']['fn']}, TN: {metrics['fraudulent']['tn']}")
    
    print(f"\n{'─' * 70}")
    print("SUSPICIOUS-RECRUITMENT Detection:")
    print(f"{'─' * 70}")
    print(f"  Precision: {metrics['suspicious']['precision']:.4f}")
    print(f"  Recall:    {metrics['suspicious']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['suspicious']['f1']:.4f}")
    print(f"  TP: {metrics['suspicious']['tp']}, FP: {metrics['suspicious']['fp']}, "
          f"FN: {metrics['suspicious']['fn']}")
    
    if errors:
        print(f"\n{'─' * 70}")
        print(f"Errors encountered: {len(errors)}")
        for url, error in errors[:5]:  # Show first 5 errors
            print(f"  {url}: {error}")
    
    print("\n" + "=" * 70)
    
    return metrics


if __name__ == "__main__":
    main()

