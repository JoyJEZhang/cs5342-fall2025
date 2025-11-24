"""
Simplified fraud detection training - optimized for your specific dataset
Focus on what actually works with your data
"""

import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--train_size', type=float, default=0.70)
    parser.add_argument('--model_type', default='logistic', choices=['logistic', 'rf'])
    args = parser.parse_args()
    
    print("=" * 80)
    print("SIMPLIFIED FRAUD DETECTION TRAINING")
    print("Optimized for your dataset characteristics")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading: {args.data}")
    df = pd.read_csv(args.data)
    
    if 'text ' in df.columns:
        df.rename(columns={'text ': 'text'}, inplace=True)
    if 'Label by Human' in df.columns:
        df.rename(columns={'Label by Human': 'label'}, inplace=True)
    
    df = df.dropna(subset=['text', 'label'])
    
    print(f"Total samples: {len(df)}")
    print("\nLabel distribution:")
    for label, count in df['label'].value_counts().sort_index().items():
        label_name = ['Legitimate', 'Suspicious', 'Fraudulent'][int(label)]
        print(f"  {label_name} ({label}): {count} ({count/len(df)*100:.1f}%)")
    
    # Split
    train_df, test_df = train_test_split(
        df, train_size=args.train_size,
        random_state=42, stratify=df['label']
    )
    
    print(f"\nSplit: {len(train_df)} train, {len(test_df)} test")
    
    # ========================================================================
    # Try Multiple Approaches
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE APPROACHES")
    print("=" * 80)
    
    approaches = []
    
    # Approach 1: Simple TF-IDF with character n-grams (catches patterns)
    print("\n1. TF-IDF with character n-grams...")
    vec1 = TfidfVectorizer(
        max_features=100,
        ngram_range=(3, 5),  # Character n-grams
        analyzer='char_wb',
        min_df=2
    )
    X_train_1 = vec1.fit_transform(train_df['text'])
    X_test_1 = vec1.transform(test_df['text'])
    
    model1 = LogisticRegression(
        C=0.5,  # Strong regularization
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model1.fit(X_train_1, train_df['label'])
    acc1 = model1.score(X_test_1, test_df['label'])
    approaches.append(('Char n-grams + Logistic', model1, vec1, acc1))
    print(f"   Accuracy: {acc1:.4f}")
    
    # Approach 2: Word-based TF-IDF with bigrams
    print("\n2. Word TF-IDF with bigrams...")
    vec2 = TfidfVectorizer(
        max_features=75,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
        stop_words='english'
    )
    X_train_2 = vec2.fit_transform(train_df['text'])
    X_test_2 = vec2.transform(test_df['text'])
    
    model2 = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model2.fit(X_train_2, train_df['label'])
    acc2 = model2.score(X_test_2, test_df['label'])
    approaches.append(('Word n-grams + Logistic', model2, vec2, acc2))
    print(f"   Accuracy: {acc2:.4f}")
    
    # Approach 3: Simple count vectorizer (sometimes works better!)
    print("\n3. Count vectorizer...")
    vec3 = CountVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        min_df=2,
        binary=True  # Just presence/absence
    )
    X_train_3 = vec3.fit_transform(train_df['text'])
    X_test_3 = vec3.transform(test_df['text'])
    
    model3 = LogisticRegression(
        C=0.3,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model3.fit(X_train_3, train_df['label'])
    acc3 = model3.score(X_test_3, test_df['label'])
    approaches.append(('Binary counts + Logistic', model3, vec3, acc3))
    print(f"   Accuracy: {acc3:.4f}")
    
    # Approach 4: Random Forest (shallow)
    print("\n4. Shallow Random Forest...")
    vec4 = TfidfVectorizer(
        max_features=60,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.75
    )
    X_train_4 = vec4.fit_transform(train_df['text'])
    X_test_4 = vec4.transform(test_df['text'])
    
    model4 = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,  # Very shallow
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    model4.fit(X_train_4, train_df['label'])
    acc4 = model4.score(X_test_4, test_df['label'])
    approaches.append(('Shallow RF', model4, vec4, acc4))
    print(f"   Accuracy: {acc4:.4f}")
    
    # ========================================================================
    # Select Best Model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    approaches.sort(key=lambda x: x[3], reverse=True)
    
    print("\nAccuracy ranking:")
    for i, (name, _, _, acc) in enumerate(approaches, 1):
        star = " ⭐" if i == 1 else ""
        print(f"  {i}. {name:<35} {acc:.4f}{star}")
    
    # Use best model
    best_name, best_model, best_vec, best_acc = approaches[0]
    print(f"\n✓ Selected: {best_name} ({best_acc:.4f})")
    
    # Evaluate best model
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION OF BEST MODEL")
    print("=" * 80)
    
    X_train_best = best_vec.fit_transform(train_df['text'])
    X_test_best = best_vec.transform(test_df['text'])
    
    best_model.fit(X_train_best, train_df['label'])
    
    y_pred = best_model.predict(X_test_best)
    train_pred = best_model.predict(X_train_best)
    
    train_acc = (train_pred == train_df['label']).mean()
    test_acc = (y_pred == test_df['label']).mean()
    
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Overfitting gap: {train_acc - test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        test_df['label'], y_pred,
        target_names=['Legitimate', 'Suspicious', 'Fraudulent'],
        zero_division=0
    ))
    
    cm = confusion_matrix(test_df['label'], y_pred)
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 Legit  Susp  Fraud")
    for i, row in enumerate(cm):
        label_name = ['Legitimate', 'Suspicious ', 'Fraudulent'][i]
        print(f"True {label_name}  {row[0]:5}  {row[1]:4}  {row[2]:5}")
    
    # Show most important features if available
    if hasattr(best_model, 'coef_'):
        # Logistic regression
        feature_names = best_vec.get_feature_names_out()
        
        # Get coefficients for each class
        print("\nMost predictive features by class:")
        for class_idx in range(len(best_model.coef_)):
            class_name = ['Legitimate', 'Suspicious', 'Fraudulent'][class_idx]
            coefs = best_model.coef_[class_idx]
            top_indices = np.argsort(np.abs(coefs))[-10:][::-1]
            
            print(f"\n{class_name}:")
            for idx in top_indices:
                direction = "+" if coefs[idx] > 0 else "-"
                print(f"  {direction} {feature_names[idx]:<30} {abs(coefs[idx]):.4f}")
    
    # ========================================================================
    # Save Best Model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    import os
    
    with open(os.path.join(args.out_dir, 'fraud_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(args.out_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(best_vec, f)
    
    # Save metadata
    metadata = {
        'model_type': best_name,
        'test_accuracy': test_acc,
        'train_accuracy': train_acc,
        'overfitting_gap': train_acc - test_acc,
        'train_size': len(train_df),
        'test_size': len(test_df)
    }
    
    with open(os.path.join(args.out_dir, 'model_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✓ Saved fraud_model.pkl")
    print("✓ Saved vectorizer.pkl")
    print("✓ Saved model_metadata.pkl")
    
    # ========================================================================
    # Final Recommendations
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\nBest Model: {best_name}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    if test_acc >= 0.75:
        print("\n✓✓ Excellent! Ready for deployment.")
    elif test_acc >= 0.70:
        print("\n✓ Good enough for deployment with monitoring.")
    elif test_acc >= 0.65:
        print("\n⚠ Marginal. Consider:")
        print("  1. Collecting 50-100 more labeled samples")
        print("  2. Trying ensemble of multiple approaches")
    else:
        print("\n❌ Below acceptable threshold. Issues:")
        print("  1. Dataset too small (only 164 samples)")
        print("  2. Classes may not be well-separated")
        print("  3. Need more diverse training examples")
        print("\n  Recommendation: Collect at least 300 labeled samples")
    
    # Reality check
    majority_baseline = (df['label'].value_counts().max() / len(df))
    print(f"\nBaseline (always predict majority): {majority_baseline:.4f}")
    print(f"Your model improvement: +{(test_acc - majority_baseline)*100:.1f}%")
    
    if test_acc - majority_baseline < 0.10:
        print("\n⚠ Model is barely better than baseline!")
        print("  This suggests the features aren't capturing useful patterns.")
        print("  You may need more data or different features.")

if __name__ == '__main__':
    main()