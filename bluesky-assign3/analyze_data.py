"""
Quick data analysis script to understand your dataset
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("DATA ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Handle different column names
    if 'text ' in df.columns:
        df.rename(columns={'text ': 'text'}, inplace=True)
    if 'Label by Human' in df.columns:
        df.rename(columns={'Label by Human': 'label'}, inplace=True)
    
    print(f"\nDataset: {args.data}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'label' not in df.columns:
        print("\n⚠ WARNING: No 'label' column found!")
        print("Available columns:", df.columns.tolist())
        return
    
    # Label distribution
    print("\n--- Label Distribution ---")
    df_clean = df.dropna(subset=['text', 'label'])
    print(f"Clean samples: {len(df_clean)} (removed {len(df) - len(df_clean)} with missing data)")
    
    label_counts = df_clean['label'].value_counts().sort_index()
    print("\nBy count:")
    for label, count in label_counts.items():
        label_name = ['Legitimate', 'Suspicious', 'Fraudulent'][int(label)]
        print(f"  {label_name} ({label}): {count:3d} ({count/len(df_clean)*100:.1f}%)")
    
    # Minimum samples per class
    min_count = label_counts.min()
    print(f"\nSmallest class: {min_count} samples")
    
    # Recommendations for splitting
    print("\n--- Train/Test Split Recommendations ---")
    
    splits_to_try = [0.5, 0.6, 0.7, 0.8]
    print("\nWith different train/test splits:")
    print(f"{'Split':<10} {'Train':<10} {'Test':<10} {'Min Test/Class':<20}")
    print("-" * 50)
    
    for split in splits_to_try:
        train_size = int(len(df_clean) * split)
        test_size = len(df_clean) - train_size
        min_test_per_class = int(min_count * (1 - split))
        print(f"{split:.1f}/{1-split:.1f}     {train_size:<10} {test_size:<10} ~{min_test_per_class}")
    
    print("\n⚠ For reliable evaluation, aim for at least 5 samples per class in test set")
    
    # Recommended split
    recommended_split = 0.7
    for split in splits_to_try:
        min_test = int(min_count * (1 - split))
        if min_test >= 5:
            recommended_split = split
            break
    
    print(f"\n✓ Recommended: --train_size {recommended_split}")
    
    # Text length analysis
    if 'text' in df_clean.columns:
        print("\n--- Text Length Analysis ---")
        df_clean['text_length'] = df_clean['text'].str.len()
        
        print(f"Average text length: {df_clean['text_length'].mean():.0f} characters")
        print(f"Median text length: {df_clean['text_length'].median():.0f} characters")
        print(f"Min: {df_clean['text_length'].min()}, Max: {df_clean['text_length'].max()}")
        
        print("\nBy label:")
        for label in sorted(df_clean['label'].unique()):
            label_name = ['Legitimate', 'Suspicious', 'Fraudulent'][int(label)]
            avg_len = df_clean[df_clean['label'] == label]['text_length'].mean()
            print(f"  {label_name} ({label}): {avg_len:.0f} chars (avg)")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate current performance expectations
    majority_class_pct = (label_counts.max() / len(df_clean)) * 100
    print(f"\nMajority class baseline: {majority_class_pct:.1f}%")
    print("(Always predicting the most common class)")
    print(f"\nYour model achieved: 67.65%")
    
    if 67.65 > majority_class_pct:
        improvement = 67.65 - majority_class_pct
        print(f"✓ Model is {improvement:.1f}% better than baseline!")
    else:
        print("⚠ Model is barely better than baseline - needs improvement")
    
    print("\n--- Action Items ---")
    print("1. Try stratified split: use train_model_improved.py")
    print(f"2. Use larger training set: --train_size {recommended_split} or higher")
    print("3. Collect more data for underrepresented classes")
    if min_count < 10:
        print(f"4. ⚠ CRITICAL: Smallest class has only {min_count} samples - collect more!")

if __name__ == '__main__':
    main()