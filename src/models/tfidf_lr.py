"""Модель: TF-IDF + Logistic Regression"""

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils.metrics import compute_macro_f1, save_results
from src.utils.constants import RANDOM_SEED, TEST_SIZE


def train_tfidf_lr(df, feature_cols, iteration_name='tfidf_lr'):
    """
    Обучить TF-IDF + Logistic Regression.
    Совместимо с пайплайном пост-редактирования.
    """
    print(f'''
          Модель: TF-IDF + Logistic Regression
          Итерация: {iteration_name}
          ''')

    from src.features.feature_engineering import prepare_features_for_tfidf

    # Подготавливаем признаки
    df_prep = prepare_features_for_tfidf(df, feature_cols)

    X = df_prep['text_features']
    y = df_prep['pos_gold']

    # Разделение на train/test (стратифицированное)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"\n Train size: {len(X_train)}, Test size: {len(X_test)}")

    # TF-IDF векторизация
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"TF-IDF матрица: {X_train_tfidf.shape}")

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
        class_weight='balanced'
    )

    print("Обучение модели...")
    lr.fit(X_train_tfidf, y_train)

    # Предсказания
    y_pred = lr.predict(X_test_tfidf)

    # Метрики
    f1 = compute_macro_f1(y_test, y_pred)

    print(f'''
          macro-F1: {f1:.4f}
          Classification Report: {classification_report(y_test, y_pred, zero_division=0)}
          ''')

    # Сохраняем результаты эксперимента
    features_str = ', '.join(feature_cols)
    save_results(
        experiment_name=iteration_name,
        model_name='TF-IDF + LR',
        features_used=features_str,
        macro_f1=round(f1, 4)
    )

    # Сохраняем лучшую модель (только для full итерации)
    if iteration_name.endswith('full'):
        save_trained_model(tfidf, lr)

    return {
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'model': lr,
        'tfidf': tfidf,
        'test_indices': X_test.index
    }


def save_trained_model(tfidf, model, filepath='experiments/trained_models/tfidf_lr_full.pkl'):
    """
    Сохранить обученную модель TF-IDF + LR.
    
    Args:
        tfidf: TfidfVectorizer - векторизатор
        model: LogisticRegression - обученная модель
        filepath: путь для сохранения
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({
            'tfidf': tfidf,
            'model': model
        }, f)
    print(f"Модель сохранена: {filepath}")


def load_trained_model(filepath='experiments/trained_models/tfidf_lr_full.pkl'):
    """
    Загрузить обученную модель TF-IDF + LR.
    
    Returns:
        tuple: (tfidf_vectorizer, logistic_regression_model)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Модель не найдена: {filepath}\n"
            "Сначала обучите модель с итерацией 'full':\n"
            "  python -m src.models.tfidf_lr"
        )
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    tfidf = data['tfidf']
    model = data['model']
    
    print(f"Загружена модель: {filepath}")
    return tfidf, model


if __name__ == '__main__':
    from src.data.loader import load_raw_data, preprocess_data
    from src.features.feature_engineering import create_all_features
    
    df = load_raw_data()
    df = preprocess_data(df)
    
    # Тест для разных итераций
    for iteration, feat_cols in [
        ('pos_only', ['pos_stanza']),
        ('pos_context', ['pos_stanza', 'prev_pos', 'next_pos']),
        ('full', ['token', 'lemma', 'pos_stanza', 'prev_token', 'next_token', 'prev_pos', 'next_pos', 'token_length'])
    ]:
        df_feat, _ = create_all_features(df, iteration)
        train_tfidf_lr(df_feat, feat_cols, iteration_name=f'tfidf_lr_{iteration}')