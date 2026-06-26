"""Метрики для оценки моделей"""

from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


def compute_macro_f1(y_true, y_pred):
    """Вычислить macro-F1"""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_baseline_f1(df, gold_col='pos_gold', pred_col='stanza_uni'):
    """Вычислить macro-F1 для Stanza baseline"""
    return compute_macro_f1(df[gold_col], df[pred_col])


def print_classification_report(y_true, y_pred, labels=None):
    """Вывести classification report"""
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    return report


def get_confusion_matrix(y_true, y_pred, labels=None):
    """Получить confusion matrix"""
    return confusion_matrix(y_true, y_pred, labels=labels)


def save_results(experiment_name, model_name, features_used, macro_f1, filepath='experiments/results.csv'):
    """Сохранить результаты эксперимента в CSV"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    row = pd.DataFrame([{
        'experiment': experiment_name,
        'model': model_name,
        'features': features_used,
        'macro_f1': macro_f1
    }])
    
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        row = pd.concat([existing, row], ignore_index=True)
    
    row.to_csv(filepath, index=False)
    print(f"✅ Результаты сохранены: {filepath}")