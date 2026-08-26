import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocess import clean_text
from ml.evaluate import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "support_tickets.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_all_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("SupportAI: Initiating ML Training & Evaluation Pipeline")
    print("=" * 60)
    
    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} records across {df['category'].nunique()} categories.")
    
    # 2. Clean Text
    df['cleaned_text'] = df['ticket_text'].apply(clean_text)
    
    # -------------------------------------------------------------
    # CATEGORY CLASSIFICATION: Logistic Regression vs Random Forest
    # -------------------------------------------------------------
    print("\n--- 1. Training & Comparing Category Classifiers ---")
    X = df['cleaned_text']
    y_cat = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.20, random_state=42, stratify=y_cat
    )
    
    # Vectorizer for category
    cat_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True
    )
    X_train_vec = cat_vectorizer.fit_transform(X_train)
    X_test_vec = cat_vectorizer.transform(X_test)
    
    cat_classes = sorted(list(y_cat.unique()))
    
    # Algorithm 1: Logistic Regression
    print("Training Logistic Regression...")
    lr_cat = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
    lr_cat.fit(X_train_vec, y_train)
    lr_metrics = evaluate_model(lr_cat, X_test_vec, y_test, class_names=cat_classes)
    
    # Algorithm 2: Random Forest
    print("Training Random Forest...")
    rf_cat = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42)
    rf_cat.fit(X_train_vec, y_train)
    rf_metrics = evaluate_model(rf_cat, X_test_vec, y_test, class_names=cat_classes)
    
    print(f"\n[Category Evaluation Results]")
    print(f"Logistic Regression -> Accuracy: {lr_metrics['accuracy'] * 100:.2f}%, F1-Score: {lr_metrics['f1_weighted'] * 100:.2f}%")
    print(f"Random Forest       -> Accuracy: {rf_metrics['accuracy'] * 100:.2f}%, F1-Score: {rf_metrics['f1_weighted'] * 100:.2f}%")
    
    # Select Best Model for Production
    if lr_metrics['f1_weighted'] >= rf_metrics['f1_weighted']:
        best_cat_model = lr_cat
        best_algo_name = "Logistic Regression"
    else:
        best_cat_model = rf_cat
        best_algo_name = "Random Forest"
    
    print(f"=> Selected Best Model for Category Classification: {best_algo_name}")
    
    # Save Category Model & Vectorizer
    with open(os.path.join(MODELS_DIR, "category_model.pkl"), "wb") as f:
        pickle.dump(best_cat_model, f)
    with open(os.path.join(MODELS_DIR, "category_vectorizer.pkl"), "wb") as f:
        pickle.dump(cat_vectorizer, f)
        
    # Also save the other model so users can toggle or test both if desired
    with open(os.path.join(MODELS_DIR, "lr_category_model.pkl"), "wb") as f:
        pickle.dump(lr_cat, f)
    with open(os.path.join(MODELS_DIR, "rf_category_model.pkl"), "wb") as f:
        pickle.dump(rf_cat, f)
        
    # -------------------------------------------------------------
    # PRIORITY PREDICTION MODEL
    # -------------------------------------------------------------
    print("\n--- 2. Training Priority Prediction Model ---")
    y_prio = df['priority']
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X, y_prio, test_size=0.20, random_state=42, stratify=y_prio
    )
    
    prio_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)
    X_train_p_vec = prio_vectorizer.fit_transform(X_train_p)
    X_test_p_vec = prio_vectorizer.transform(X_test_p)
    
    prio_classes = ["Low", "Medium", "High"]
    prio_model = LogisticRegression(C=1.2, max_iter=1000, random_state=42)
    prio_model.fit(X_train_p_vec, y_train_p)
    prio_metrics = evaluate_model(prio_model, X_test_p_vec, y_test_p, class_names=prio_classes)
    print(f"Priority Model -> Accuracy: {prio_metrics['accuracy'] * 100:.2f}%, F1-Score: {prio_metrics['f1_weighted'] * 100:.2f}%")
    
    with open(os.path.join(MODELS_DIR, "priority_model.pkl"), "wb") as f:
        pickle.dump(prio_model, f)
    with open(os.path.join(MODELS_DIR, "priority_vectorizer.pkl"), "wb") as f:
        pickle.dump(prio_vectorizer, f)

    # -------------------------------------------------------------
    # SENTIMENT ANALYSIS MODEL
    # -------------------------------------------------------------
    print("\n--- 3. Training Sentiment Analysis Model ---")
    y_sent = df['sentiment']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sent, test_size=0.20, random_state=42, stratify=y_sent
    )
    
    sent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)
    X_train_s_vec = sent_vectorizer.fit_transform(X_train_s)
    X_test_s_vec = sent_vectorizer.transform(X_test_s)
    
    sent_classes = ["Negative", "Neutral", "Positive"]
    sent_model = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
    sent_model.fit(X_train_s_vec, y_train_s)
    sent_metrics = evaluate_model(sent_model, X_test_s_vec, y_test_s, class_names=sent_classes)
    print(f"Sentiment Model -> Accuracy: {sent_metrics['accuracy'] * 100:.2f}%, F1-Score: {sent_metrics['f1_weighted'] * 100:.2f}%")
    
    with open(os.path.join(MODELS_DIR, "sentiment_model.pkl"), "wb") as f:
        pickle.dump(sent_model, f)
    with open(os.path.join(MODELS_DIR, "sentiment_vectorizer.pkl"), "wb") as f:
        pickle.dump(sent_vectorizer, f)

    # -------------------------------------------------------------
    # SAVE FULL METRICS REPORT FOR DASHBOARD & PERFORMANCE PAGE
    # -------------------------------------------------------------
    comparison_summary = {
        "dataset": {
            "total_samples": len(df),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "categories_count": len(cat_classes),
            "categories": cat_classes,
            "category_distribution": df['category'].value_counts().to_dict(),
            "priority_distribution": df['priority'].value_counts().to_dict(),
            "sentiment_distribution": df['sentiment'].value_counts().to_dict()
        },
        "selected_best_model": best_algo_name,
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression",
                "task": "Category Classification",
                "hyperparameters": {"C": 1.5, "max_iter": 1000, "solver": "lbfgs"},
                "metrics": lr_metrics
            },
            "random_forest": {
                "name": "Random Forest Classifier",
                "task": "Category Classification",
                "hyperparameters": {"n_estimators": 150, "max_depth": 25, "criterion": "gini"},
                "metrics": rf_metrics
            },
            "priority_classifier": {
                "name": "Logistic Regression (Priority)",
                "task": "Priority Prediction",
                "metrics": prio_metrics
            },
            "sentiment_classifier": {
                "name": "Logistic Regression (Sentiment)",
                "task": "Sentiment Analysis",
                "metrics": sent_metrics
            }
        }
    }
    
    metrics_file = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=4)
        
    print(f"\nAll models, vectorizers, and metrics successfully exported to {MODELS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    train_all_models()
