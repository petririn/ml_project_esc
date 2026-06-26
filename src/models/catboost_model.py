"""Модель: CatBoost Classifier"""

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils.metrics import compute_macro_f1, save_results
from src.utils.constants import RANDOM_SEED, TEST_SIZE

import pickle
import os 

def train_catboost(df, feature_cols, iteration_name='catboost'):
    from src.features.feature_engineering import prepare_features_for_catboost
    
    df_prep, cat_cols, num_cols = prepare_features_for_catboost(df, feature_cols)
    
    # Фильтруем cat_cols только существующими колонками
    cat_cols = [col for col in cat_cols if col in df_prep.columns]
    
    X = df_prep[feature_cols]
    y = df_prep['pos_gold']
    
    # Разделение на train/test (стратифицированное)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    
    print(f'''
          Train size: {len(X_train)}, Test size: {len(X_test)}
          Категориальные признаки: {cat_cols}
          Числовые признаки: {num_cols}
          ''')
    
    # CatBoost (параметры для слабого ноутбука)
    model = CatBoostClassifier(
        iterations=200,  # мало итераций
        depth=4,         # небольшая глубина
        learning_rate=0.1,
        random_seed=RANDOM_SEED,
        verbose=False,   # не выводить прогресс
        cat_features=cat_cols if cat_cols else None
    )
    
    print("Обучение модели...")
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики
    f1 = compute_macro_f1(y_test, y_pred)
    
    print(f'''
          macro-F1: {f1:.4f}
          Classification Report: {classification_report(y_test, y_pred, zero_division=0)}
          ''')
    
    # Сохраняем результаты
    features_str = ', '.join(feature_cols)
    save_results(
        experiment_name=iteration_name,
        model_name='CatBoost',
        features_used=features_str,
        macro_f1=round(f1, 4)
    )
    if iteration_name.endswith('full'):
        save_trained_model(model)

    return {
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'model': model,
        'test_indices': X_test.index
    }

def save_trained_model(model, filepath='experiments/trained_models/catboost_full.pkl'):
    """Сохранить обученную модель CatBoost."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    print(f" Модель сохранена: {filepath}")

def load_trained_model(filepath='experiments/trained_models/catboost_full.pkl'):
    """Загрузить обученную модель CatBoost."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(filepath)
    print(f"Загружена модель: {filepath}")
    return model


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
        train_catboost(df_feat, feat_cols, iteration_name=f'catboost_{iteration}')