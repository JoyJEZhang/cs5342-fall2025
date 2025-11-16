# Assignment 3: Online Recruitment Fraud Detection Labeler

## Group Information
- Group Members: 
Grace Myers
Eva Huang
Xinyi Huang
Jiayu Zhang
- Policy Proposal: Online Recruitment Fraud Detection

## Evaluation Results

The labeler now achieves the following performance on the test dataset:

- **Overall Accuracy**: 88.00% (132 out of 150 posts correctly labeled)
- **Fraudulent Detection**: 
  - Precision: 100% (no false positives)
  - Recall: 78.57% (11 out of 14 fraudulent posts detected)
  - F1 Score: 0.88
- **Suspicious Detection**:
  - Precision: 64.44%
  - Recall: 93.55% (29 out of 31 suspicious posts detected)
  - F1 Score: 0.76

To reproduce these results, run:
```bash
python evaluate_labeler.py data_real_only.csv
```

## Overview

This implementation provides a Bluesky labeler that detects online recruitment fraud by identifying suspicious and fraudulent job postings. The labeler uses a hybrid approach combining rule-based pattern matching with a machine learning model (RandomForest classifier) to achieve 88% accuracy on real-world test data.

## Files Submitted

### Core Implementation
- `pylabel/policy_proposal_labeler.py` - Main labeler implementation with ML model integration
- `pylabel/label.py` - Helper functions for Bluesky API interactions
- `evaluate_labeler.py` - Evaluation script for calculating metrics

### Test Data
- `data.csv` - Combined test dataset (215 posts: 150 real-world + 65 mock)
- `data_real_only.csv` - Real-world posts only (150 posts, used for 88% accuracy evaluation)

### ML Model Files
- `fraud_model.pkl` - Trained RandomForest classifier
- `vectorizer.pkl` - TF-IDF vectorizer for text feature extraction

## Setup

### Prerequisites
1. Python 3.8+
2. Required packages (install via `pip install`):
   - `atproto`
   - `pandas`
   - `python-dotenv`
   - `scikit-learn` (for ML model)
   - `numpy` (for ML model)

### Environment Variables
Create a `.env` file in the `bluesky-assign3` directory with:
```
USERNAME=your_bluesky_username
PW=your_bluesky_password
```

## How to Run

### 1. Prepare Test Data
The test data file `data.csv` is already prepared with 215 posts combining both real-world and mock datasets.

### 2. Evaluate the Labeler
Run the evaluation script on the test data:
```bash
python evaluate_labeler.py data.csv
```

This will:
- Load all posts from `data.csv`
- Run the labeler on each post
- Calculate accuracy, precision, recall, and F1 scores
- Display detailed results including mismatches

### 3. Emit Labels (Optional)
To actually emit labels to Bluesky (use with caution):
```bash
python evaluate_labeler.py data.csv --emit_labels
```

**Warning**: Only use `--emit_labels` when you're confident in your labeler's accuracy!

## Labeler Implementation Details

### Detection Signals

The labeler uses multiple signals to detect recruitment fraud:

1. **Textual Signals:**
   - Urgency keywords ("urgent", "hiring now", "limited time")
   - Unrealistic salary patterns (e.g., "$3,000/week")
   - Payment request keywords ("upfront fee", "training kit")
   - Off-platform communication requests ("WhatsApp", "DM me")
   - Personal information requests ("SSN", "bank account")
   - Vague job descriptions ("work from home", "easy money")
   - Excessive emoji usage

2. **URL Signals:**
   - URL shorteners (bit.ly, tinyurl.com, etc.)
   - Suspicious domain patterns

### Label Types

- `fraudulent-recruitment`: High confidence fraud detection (threshold ≥ 0.3)
- `suspicious-recruitment`: Medium confidence suspicious content (threshold ≥ 0.2)
- No label: Legitimate or low-risk content

### Implementation Approach

The labeler uses a **hybrid approach** combining:

1. **Rule-Based Detection**: Pattern matching on text and URLs
   - Payment requests: 25% weight
   - PII requests: 20% weight
   - Unrealistic salaries: 15% weight
   - Off-platform requests: 15% weight
   - Other indicators: 5-10% each

2. **Machine Learning Model**: RandomForest classifier
   - Trained on 149 labeled posts
   - Uses TF-IDF text features + rule-based features
   - Combines with rule-based scores (60% ML, 40% rules)
   - Achieves 88% accuracy on test set

## Test Data

We use **two datasets** for comprehensive evaluation:

### Original Source Files 
The original source files are located in the parent directory (`../`):
- `Cleaned_BlueSky_Data_Collection.csv` - 995 real-world Bluesky posts with labels
- `Cleaned_mock_data.csv` - 997 manually created mock job postings with labels

**Note**: These source files are provided for reference. The processed test datasets (`data.csv` and `data_real_only.csv`) are the files used for evaluation.

### Processed Test Datasets

#### Real-World Data (`data_real_only.csv`)
- 150 posts selected from actual Bluesky posts
- Includes URLs for API testing
- Contains legitimate, suspicious, and fraudulent examples
- Used for the 88% accuracy evaluation

#### Mock/Synthetic Data
- 65 additional manually created mock job postings (selected from source)
- Text-only (no URLs)
- Focuses on fraudulent and suspicious patterns
- Provides more examples of fraud for comprehensive testing

#### Combined Dataset (`data.csv`)
The `data.csv` file contains **215 posts** combining selected samples from both source datasets:
- **105 legitimate posts** (no label) - all from real-world data
- **56 suspicious posts** (`suspicious-recruitment`) - 31 real + 25 mock
- **54 fraudulent posts** (`fraudulent-recruitment`) - 14 real + 40 mock

This combination provides:
- More diverse fraud examples for better evaluation
- Both real-world and controlled scenarios
- Better balance between legitimate and fraudulent content

## Evaluation Metrics

The evaluation script calculates:
- **Accuracy**: Exact label match rate
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

Metrics are calculated separately for:
- Fraudulent recruitment detection
- Suspicious recruitment detection

## Limitations and Future Improvements

### Current Limitations
1. Text-based detection only (no account metadata analysis)
2. Fixed thresholds (could be tuned with more data)
3. Limited to English language patterns
4. No behavioral analysis (account age, posting frequency)

### Future Improvements
1. Integrate account metadata (follower ratios, account age, posting frequency)
2. Use transformer models (BERT, etc.) for better text classification
3. Implement behavioral analysis (account patterns, posting history)
4. Add network analysis (coordinated activity detection)
5. Multi-language support
6. Dynamic threshold adjustment based on context
7. Real-time model retraining with new data

## Ethical Considerations

This labeler is designed to help users identify potentially fraudulent job postings, but:
- **False positives** could harm legitimate small businesses
- **False negatives** could expose users to scams
- The labeler uses "suspicious" and "fraudulent" labels rather than definitive "scam" labels to allow user judgment
- Users can adjust visibility settings (Warn, Hide, Show) based on their preferences


