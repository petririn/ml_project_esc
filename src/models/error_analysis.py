"""Анализ ошибок лучшей модели и визуализация результатов"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.data.loader import load_raw_data, preprocess_data
from src.features.feature_engineering import create_all_features
from src.models.tfidf_lr import train_tfidf_lr
from src.models.catboost_model import train_catboost


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Построить и сохранить confusion matrix"""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix сохранён: {save_path}")


def plot_results_comparison(results_path='experiments/results.csv'):
    """Построить сравнение моделей"""
    if not os.path.exists(results_path):
        print(f"Файл результатов не найден: {results_path}")
        return

    df = pd.read_csv(results_path)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='macro_f1', hue='features')
    plt.title('Сравнение моделей по macro-F1')
    plt.xlabel('Модель')
    plt.ylabel('macro-F1')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('experiments/models_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Сравнение моделей сохранено: experiments/models_comparison.png")


def analyze_errors(df, feature_cols, model_name='TF-IDF + LR'):
    """Полный анализ ошибок модели."""

    if model_name == 'TF-IDF + LR':
        results = train_tfidf_lr(df, feature_cols, 'full_analysis')
    else:
        results = train_catboost(df, feature_cols, 'full_analysis')

    y_test = results['y_test']
    y_pred = results['y_pred']
    test_indices = results['test_indices'] 

    n_test = len(y_test)
    test_df = df.loc[test_indices].copy().reset_index(drop=True)

    analysis_df = pd.DataFrame({
        'token': test_df['token'].values,
        'lemma': test_df['lemma'].values,
        'pos_stanza': test_df['pos_stanza'].values,
        'pos_gold': y_test.values,
        'pos_pred': y_pred,
        'stanza_correct': (test_df['pos_stanza'].values == y_test.values).astype(int),
        'model_correct': (y_pred == y_test.values).astype(int)
    })

    total = len(analysis_df)
    stanza_errors = (analysis_df['stanza_correct'] == 0).sum()
    model_errors = (analysis_df['model_correct'] == 0).sum()
    fixed_by_model = ((analysis_df['stanza_correct'] == 0) &
                      (analysis_df['model_correct'] == 1)).sum()
    new_errors = ((analysis_df['stanza_correct'] == 1) &
                  (analysis_df['model_correct'] == 0)).sum()

    print(f'''
          Всего токенов в тесте: {total}
          Ошибок Stanza: {stanza_errors} ({stanza_errors/total*100:.1f}%)
          Ошибок модели: {model_errors} ({model_errors/total*100:.1f}%)
          Исправлено моделью: {fixed_by_model}
          Новых ошибок модели: {new_errors}
          ''')

    fixed = analysis_df[
        (analysis_df['stanza_correct'] == 0) &
        (analysis_df['model_correct'] == 1)
    ]
    print(f"ПРИМЕРЫ ИСПРАВЛЕННЫХ ОШИБОК STANZA (первые 5):")
    for _, row in fixed.head(5).iterrows():
        print(f"  '{row['token']}' (лемма: {row['lemma']})")
        print(f"    Stanza: {row['pos_stanza']:6} → Gold: {row['pos_gold']:6} → Модель: {row['pos_pred']:6}")

    new_err = analysis_df[
        (analysis_df['stanza_correct'] == 1) &
        (analysis_df['model_correct'] == 0)
    ]
    print(f"ПРИМЕРЫ НОВЫХ ОШИБОК МОДЕЛИ (первые 5):")
    for _, row in new_err.head(5).iterrows():
        print(f"  '{row['token']}' (лемма: {row['lemma']})")
        print(f"    Stanza: {row['pos_stanza']:6} → Gold: {row['pos_gold']:6} → Модель: {row['pos_pred']:6}")

    plot_confusion_matrix(
        y_test, y_pred,
        f'Confusion Matrix: {model_name}',
        f'experiments/confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    )

    print(f'''
          Classification Report: {classification_report(y_test, y_pred, zero_division=0)}
          ''')

    os.makedirs('experiments', exist_ok=True)
    analysis_df.to_csv('experiments/error_analysis.csv', index=False, encoding='utf-8')
    print(f"Анализ ошибок сохранён: experiments/error_analysis.csv")

    return analysis_df


def full_analysis():
    """Полный анализ: обучение лучшей модели + визуализация"""

    df = load_raw_data()
    df = preprocess_data(df)

    df_feat, feature_cols = create_all_features(df, 'full')

    analyze_errors(df_feat, feature_cols, 'TF-IDF + LR')

    plot_results_comparison()

if __name__ == '__main__':
    full_analysis()