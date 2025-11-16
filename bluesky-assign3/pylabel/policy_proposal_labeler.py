"""Implementation of recruitment fraud detection labeler for Assignment 3 Part 1"""

import re
import pickle
import os
from typing import List, Optional
from atproto import Client
from .label import post_from_url

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Using rule-based approach only.")

# Label constants
FRAUDULENT_LABEL = "fraudulent-recruitment"
SUSPICIOUS_LABEL = "suspicious-recruitment"

# Thresholds
FRAUDULENT_THRESHOLD = 0.3  # Lowered to catch more fraudulent posts
SUSPICIOUS_THRESHOLD = 0.2  # Lowered to catch more suspicious posts

class PolicyProposalLabeler:
    """Labeler for detecting online recruitment fraud on Bluesky"""

    def __init__(self, client: Client, use_ml: bool = True):
        """
        Initialize the recruitment fraud labeler
        
        Args:
            client: ATProto Client instance
            use_ml: Whether to use ML model (if available and trained)
        """
        self.client = client
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        self._initialize_detection_patterns()
        self._initialize_ml_model()
    
    def _initialize_detection_patterns(self):
        """Initialize detection patterns and keywords"""
        
        # Urgency keywords (common in scams)
        self.urgency_keywords = [
            r'\b(urgent|immediate|asap|hiring now|apply now|limited time|act fast)\b',
            r'\b(start (today|immediately|right away))\b',
            r'\b(guaranteed|guarantee)\b',
        ]
        
        # Unrealistic salary patterns
        self.salary_patterns = [
            r'\$(\d{1,3},?\d{3})[\s/]*(week|day|hour)',  # $3,000/week, $500/day
            r'(\d{1,3},?\d{3})[\s/]*(week|day|hour)',  # 3000/week
            r'earn\s+\$?\d+[,\d]*\s*(per|a)\s*(week|day|hour)',
        ]
        
        # Suspicious payment/request patterns
        self.payment_keywords = [
            r'\b(upfront|application fee|training kit|background check fee|processing fee)\b',
            r'\b(send (money|payment|deposit|fee))\b',
            r'\b(pay (first|before|to start|for training))\b',
            r'\b(deposit|down payment|registration fee)\b',
        ]
        
        # Off-platform communication requests
        self.off_platform_patterns = [
            r'\b(whatsapp|telegram|signal|dm me|message me|contact me)\b',
            r'\b(move (to|conversation)|off (platform|site))\b',
            r'\b(send (email|dm|message) (to|at))\b',
        ]
        
        # Personal information requests
        self.pii_keywords = [
            r'\b(social security|ssn|ss#|driver.?s license|bank (account|routing|info))\b',
            r'\b(personal (info|information|data|details))\b',
            r'\b(credit card|debit card|account number)\b',
        ]
        
        # Vague job description indicators
        self.vague_patterns = [
            r'\b(work from home|remote work|online job|easy money|quick cash)\b',
            r'\b(no experience (needed|required)|beginners welcome|anyone can do)\b',
            r'\b(flexible (hours|schedule|work)|set your own (hours|schedule))\b',
        ]
        
        # URL shorteners (suspicious)
        self.url_shorteners = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
            'buff.ly', 'short.link', 'cutt.ly', 'rebrand.ly'
        ]
        
        # Excessive emoji patterns (common in spam)
        self.emoji_pattern = re.compile(r'[🚀💼🏢🌍📁💰✅🔗📲💻✨🏡🌐📌]')
    
    def _initialize_ml_model(self):
        """Initialize ML model if available"""
        self.ml_model = None
        self.vectorizer = None
        
        if not self.use_ml:
            return
        
        # Try to load pre-trained model if it exists
        model_path = os.path.join(os.path.dirname(__file__), '..', 'fraud_model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("✓ Loaded pre-trained ML model")
            except Exception as e:
                print(f"Warning: Could not load ML model: {e}")
                self.use_ml = False
    
    def _extract_post_text(self, post) -> str:
        """Extract text content from a post"""
        try:
            if hasattr(post, 'value'):
                if hasattr(post.value, 'text'):
                    return post.value.text or ""
                elif hasattr(post.value, 'record') and hasattr(post.value.record, 'text'):
                    return post.value.record.text or ""
            return ""
        except Exception:
            return ""
    
    def _extract_links(self, post) -> List[str]:
        """Extract links from a post"""
        links = []
        try:
            text = self._extract_post_text(post)
            # Extract URLs from text
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            links = re.findall(url_pattern, text)
            
            # Also check embed if present
            if hasattr(post, 'value') and hasattr(post.value, 'embed'):
                embed = post.value.embed
                if hasattr(embed, 'external') and embed.external:
                    if hasattr(embed.external, 'uri'):
                        links.append(embed.external.uri)
        except Exception:
            pass
        return links
    
    def _check_text_patterns(self, text: str) -> dict:
        """
        Check text against various fraud indicators
        
        Returns:
            dict with scores for different fraud indicators
        """
        text_lower = text.lower()
        scores = {
            'urgency': 0,
            'unrealistic_salary': 0,
            'payment_request': 0,
            'off_platform': 0,
            'pii_request': 0,
            'vague_description': 0,
            'excessive_emoji': 0,
        }
        
        # Check urgency keywords
        for pattern in self.urgency_keywords:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores['urgency'] += 0.2
        
        # Check unrealistic salary patterns
        for pattern in self.salary_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract number
                num_str = match[0] if isinstance(match, tuple) else match
                num_str = num_str.replace(',', '')
                try:
                    num = int(num_str)
                    period = match[1] if isinstance(match, tuple) else 'week'
                    # Flag unrealistic salaries (e.g., $3000/week for entry level)
                    if period in ['week', 'day'] and num > 2000:
                        scores['unrealistic_salary'] += 0.3
                    elif period == 'hour' and num > 50:
                        scores['unrealistic_salary'] += 0.3
                except ValueError:
                    pass
        
        # Check payment request keywords
        for pattern in self.payment_keywords:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores['payment_request'] += 0.4
        
        # Check off-platform communication
        for pattern in self.off_platform_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores['off_platform'] += 0.3
        
        # Check PII requests
        for pattern in self.pii_keywords:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores['pii_request'] += 0.5
        
        # Check vague descriptions
        for pattern in self.vague_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores['vague_description'] += 0.15
        
        # Check excessive emojis
        emoji_count = len(self.emoji_pattern.findall(text))
        if emoji_count > 5:
            scores['excessive_emoji'] = min(0.3, emoji_count * 0.05)
        
        return scores
    
    def _check_urls(self, links: List[str]) -> dict:
        """Check URLs for suspicious patterns"""
        scores = {
            'url_shortener': 0,
            'suspicious_domain': 0,
        }
        
        for link in links:
            link_lower = link.lower()
            
            # Check for URL shorteners
            for shortener in self.url_shorteners:
                if shortener in link_lower:
                    scores['url_shortener'] += 0.3
                    break
            
            # Check for suspicious patterns in domain
            if re.search(r'(apply|job|work|hiring|career)', link_lower):
                # If it's a job-related link but uses shortener, more suspicious
                if scores['url_shortener'] > 0:
                    scores['suspicious_domain'] += 0.2
        
        return scores
    
    def _calculate_fraud_score(self, text_scores: dict, url_scores: dict) -> float:
        """
        Calculate overall fraud score from individual indicators
        
        Returns:
            float between 0 and 1 representing fraud likelihood
        """
        # Weighted combination of scores
        weights = {
            'urgency': 0.1,
            'unrealistic_salary': 0.15,
            'payment_request': 0.25,  # High weight - strong fraud indicator
            'off_platform': 0.15,
            'pii_request': 0.20,  # High weight - strong fraud indicator
            'vague_description': 0.05,
            'excessive_emoji': 0.05,
            'url_shortener': 0.05,
            'suspicious_domain': 0.05,
        }
        
        total_score = 0.0
        for key, score in {**text_scores, **url_scores}.items():
            if key in weights:
                total_score += min(score, 1.0) * weights[key]
        
        return min(total_score, 1.0)
    
    def _extract_ml_features(self, text: str, text_scores: dict, url_scores: dict) -> np.ndarray:
        """Extract features for ML model"""
        if not self.use_ml or self.vectorizer is None:
            return None
        
        # Combine rule-based scores with text features
        features = []
        
        # Rule-based scores (normalized)
        features.extend([
            text_scores.get('urgency', 0),
            text_scores.get('unrealistic_salary', 0),
            text_scores.get('payment_request', 0),
            text_scores.get('off_platform', 0),
            text_scores.get('pii_request', 0),
            text_scores.get('vague_description', 0),
            text_scores.get('excessive_emoji', 0),
            url_scores.get('url_shortener', 0),
            url_scores.get('suspicious_domain', 0),
        ])
        
        # Text-based features (TF-IDF)
        try:
            text_features = self.vectorizer.transform([text]).toarray()[0]
            features.extend(text_features.tolist())
        except:
            # If vectorizer fails, just use rule-based features
            pass
        
        return np.array(features).reshape(1, -1)
    
    def _predict_with_ml(self, features: np.ndarray) -> tuple:
        """
        Predict using ML model
        
        Returns:
            (fraud_probability, suspicious_probability)
        """
        if not self.use_ml or self.ml_model is None:
            return None, None
        
        try:
            # Get probability predictions
            if hasattr(self.ml_model, 'predict_proba'):
                probs = self.ml_model.predict_proba(features)[0]
                # Assuming binary classification: [legitimate, fraudulent]
                # Or multi-class: [legitimate, suspicious, fraudulent]
                if len(probs) == 2:
                    return probs[1], probs[1] * 0.7  # Fraud prob, suspicious prob
                elif len(probs) == 3:
                    return probs[2], probs[1]  # Fraud prob, suspicious prob
            else:
                prediction = self.ml_model.predict(features)[0]
                return float(prediction == 2), float(prediction == 1)
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None, None
    
    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given URL
        
        Args:
            url: URL of the Bluesky post
            
        Returns:
            List of labels to apply (empty list if no labels)
        """
        try:
            # Retrieve post
            post = post_from_url(self.client, url)
            
            # Extract text and links
            text = self._extract_post_text(post)
            if not text:
                return []
            
            links = self._extract_links(post)
            
            # Check patterns (rule-based)
            text_scores = self._check_text_patterns(text)
            url_scores = self._check_urls(links)
            
            # Calculate rule-based fraud score
            rule_score = self._calculate_fraud_score(text_scores, url_scores)
            
            # ML prediction if available
            ml_fraud_prob = None
            ml_susp_prob = None
            
            if self.use_ml:
                features = self._extract_ml_features(text, text_scores, url_scores)
                if features is not None:
                    ml_fraud_prob, ml_susp_prob = self._predict_with_ml(features)
            
            # Combine rule-based and ML scores
            if ml_fraud_prob is not None:
                # Weighted combination: 60% ML, 40% rule-based
                combined_fraud_score = 0.6 * ml_fraud_prob + 0.4 * rule_score
                combined_susp_score = 0.6 * ml_susp_prob + 0.4 * rule_score
            else:
                # Use rule-based only
                combined_fraud_score = rule_score
                combined_susp_score = rule_score * 0.7  # Suspicious is lower threshold
            
            # Determine labels based on thresholds
            labels = []
            if combined_fraud_score >= FRAUDULENT_THRESHOLD:
                labels.append(FRAUDULENT_LABEL)
            elif combined_susp_score >= SUSPICIOUS_THRESHOLD:
                labels.append(SUSPICIOUS_LABEL)
            
            return labels
            
        except Exception as e:
            # Log error but don't crash - return no labels on error
            print(f"Error processing post {url}: {e}")
            return []
