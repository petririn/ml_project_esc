"""Инженерия признаков для POS post-editing"""

import pandas as pd
import numpy as np

def create_context_features(df):
    """
    Создать контекстные признаки с учётом границ предложений:
    - prev_token, next_token
    - prev_pos, next_pos
    
    Контекст берётся ТОЛЬКО в пределах одного предложения!
    """
    df = df.copy()
    
    # Сдвигаем в пределах каждого предложения
    df['prev_token'] = df.groupby('sentence_id')['token'].shift(1)
    df['next_token'] = df.groupby('sentence_id')['token'].shift(-1)
    df['prev_pos'] = df.groupby('sentence_id')['pos_stanza'].shift(1)
    df['next_pos'] = df.groupby('sentence_id')['pos_stanza'].shift(-1)
    
    # Для первого токена предложения prev = START, для последнего next = END
    df['prev_token'] = df['prev_token'].fillna('START')
    df['next_token'] = df['next_token'].fillna('END')
    df['prev_pos'] = df['prev_pos'].fillna('START')
    df['next_pos'] = df['next_pos'].fillna('END')
    
    return df


def create_all_features(df, iteration='full'):
    """
    Создать все признаки в зависимости от итерации эксперимента.
    
    Args:
        df: DataFrame с данными
        iteration: 
            - 'pos_only': только pos_stanza
            - 'pos_context': pos_stanza + контекст
            - 'full': все признаки (pos + контекст + lemma + token_length)
    
    Returns:
        DataFrame с признаками и target
    """
    df = df.copy()
    
    # Базовые признаки
    df['token_length'] = df['token'].str.len()
    
    # Контекстные признаки (с учётом границ предложений)
    df = create_context_features(df)
    
    # Выбираем признаки в зависимости от итерации
    if iteration == 'pos_only':
        feature_cols = ['pos_stanza']
    elif iteration == 'pos_context':
        feature_cols = ['pos_stanza', 'prev_pos', 'next_pos']
    elif iteration == 'full':
        feature_cols = ['token', 'lemma', 'pos_stanza', 
                       'prev_token', 'next_token', 
                       'prev_pos', 'next_pos', 'token_length']
    else:
        raise ValueError(f"Неизвестная итерация: {iteration}")
    
    print(f"Созданы признаки для итерации '{iteration}': {feature_cols}")
    
    return df, feature_cols


def prepare_features_for_tfidf(df, feature_cols):
    """
    Подготовить признаки для TF-IDF + Logistic Regression.
    Конкатенируем категориальные признаки в одну строку.
    
    ОПТИМИЗАЦИЯ: используем векторизованное сложение строк вместо apply
    """
    df = df.copy()
    
    # Заполняем пропуски
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna('UNK').astype(str)
    
    # векторизованное сложение строк
    df['text_features'] = ''
    for col in feature_cols:
        df['text_features'] += col + '_' + df[col] + ' '
    
    # Убираем последний пробел
    df['text_features'] = df['text_features'].str.strip()
    
    return df


def prepare_features_for_catboost(df, feature_cols):
    """
    Подготовить признаки для CatBoost.
    Возвращает DataFrame с признаками и список категориальных колонок.
    """
    df = df.copy()
    
    # Определяем категориальные и числовые признаки
    cat_cols = [col for col in feature_cols if col != 'token_length']
    num_cols = [col for col in feature_cols if col == 'token_length']
    
    # Заполняем пропуски
    for col in cat_cols:
        df[col] = df[col].fillna('UNK').astype(str)
    
    for col in num_cols:
        df[col] = df[col].fillna(0)
    
    return df, cat_cols, num_cols


if __name__ == '__main__':
    from src.data.loader import load_raw_data, preprocess_data
    
    df = load_raw_data()
    df = preprocess_data(df)
    
    # Тест для разных итераций
    for iteration in ['pos_only', 'pos_context', 'full']:
        print(f"Тест итерации: {iteration}")
        df_feat, feature_cols = create_all_features(df, iteration)
        print(f"Признаки: {feature_cols}")
        print(df_feat[feature_cols + ['pos_gold']].head())