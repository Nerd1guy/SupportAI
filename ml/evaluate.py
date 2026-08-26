import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluates a trained classifier on test data and returns detailed metrics.
    """
    y_pred = model.predict(X_test)
    
    # Calculate overall metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    if class_names is None:
        class_names = sorted(list(set(y_test) | set(y_pred)))
    
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, labels=class_names, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for i, cls in enumerate(class_names):
        per_class_metrics[cls] = {
            "precision": round(float(prec_per_class[i]), 4),
            "recall": round(float(rec_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
            "support": int(support[i])
        }
        
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class_metrics,
        "classes": [str(c) for c in class_names],
        "confusion_matrix": cm.tolist()
    }
