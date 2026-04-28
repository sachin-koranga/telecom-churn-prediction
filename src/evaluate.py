import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def apply_threshold(search, x_test, threshold=0.7):
    # TO find best Threshold for recall and precision

    y_pred = search.predict(x_test)
    y_prob = search.predict_proba(x_test)[:, 1]
    
    
    y_pred = (y_prob > 0.72).astype(int)
    return y_pred


def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{cr}")

    return {
        "Accuracy": acc,
        "Confusion Matrix": cm,
        "Classification_Report": cr
    }