"""Загрузка и первичная обработка данных"""

import pandas as pd
import os
import numpy as np
from src.utils.constants import DATA_RAW_PATH, DATA_PROCESSED_PATH


def load_raw_data(filepath=DATA_RAW_PATH):
    """
    Загрузить исходные данные из Excel файла.
    """
    xl = pd.ExcelFile(filepath)
    sheet_name = xl.sheet_names[0]
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    print(f'''
          Загружено {len(df)} токенов
          Колонки: {list(df.columns)}
          ''')
    
    return df


def add_sentence_id(df):
    """
    Добавить номер предложения на основе пунктуации.
    """
    df = df.copy()
    
    sentence_enders = ['.', '!', '?']
    df['is_sentence_end'] = df['token'].isin(sentence_enders)
    df['sentence_id'] = df['is_sentence_end'].shift(1).fillna(0).cumsum().astype(int)
    
    print(f"Определено {df['sentence_id'].nunique()} предложений")
    
    return df


def preprocess_data(df):
    """
    Первичная предобработка данных.
    
    ВАЖНО: в исходных данных есть ДВЕ колонки POS:
    - pos_gold (родные русские теги: SPRO, APRO, PRED...)
    - pos_gold_uni (Universal Dependencies: PRON, DET, AUX...)
    
    Чтобы избежать конфликта имён, сначала переименовываем родные теги,
    затем переименовываем унифицированные в pos_gold.
    """
    df = df.copy()
    
    #  Сохраняем родные теги под другим именем (или удаляем, если не нужны)
    if 'pos_gold' in df.columns:
        df = df.rename(columns={'pos_gold': 'pos_gold_native'})
    
    #  Переименовываем унифицированные теги в pos_gold (наш target)
    df = df.rename(columns={
        'lemma_manual': 'lemma',
        'stanza_uni': 'pos_stanza',
        'pos_gold_uni': 'pos_gold'  # теперь это безопасно
    })
    
    # Обработка пропусков
    df['token'] = df['token'].fillna('')
    df['lemma'] = df['lemma'].fillna('')
    df['pos_stanza'] = df['pos_stanza'].fillna('UNK')
    df['pos_gold'] = df['pos_gold'].fillna('UNK')
    
    # Создаём признак длины токена
    df['token_length'] = df['token'].str.len()
    
    # Определяем границы предложений
    df = add_sentence_id(df)
    
    print(f'''
          Колонки после обработки: {list(df.columns)}
          Распределение POS (gold, унифицированные): {df['pos_gold'].value_counts().head(15)}
          ''')
    
    return df


def validate_data(df):
    """Проверить качество данных"""
    
    null_counts = df[['token', 'pos_gold', 'pos_stanza']].isnull().sum()
    if null_counts.any():
        print(f"Найдены пропуски:\n{null_counts[null_counts > 0]}")
    
    print(f'''
          Уникальные pos_gold: {df['pos_gold'].nunique()}
          Уникальные pos_stanza: {df['pos_stanza'].nunique()}
          ''')
    
    gold_tags = set(df['pos_gold'].unique())
    stanza_tags = set(df['pos_stanza'].unique())
    
    print(f'''
          Теги в pos_gold (uni): {sorted(gold_tags)}
          Теги в pos_stanza (uni): {sorted(stanza_tags)}
          ''')
    
    if not gold_tags.issubset(stanza_tags) and not stanza_tags.issubset(gold_tags):
        print(f'''
              Теги только в pos_gold: {gold_tags - stanza_tags}
              Только в pos_stanza: {stanza_tags - gold_tags}
              ''')
    else:
        print("Системы тегов совместимы")
    
    return df


def save_processed_data(df, filename='processed_data.csv'):
    """Сохранить обработанные данные"""
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    filepath = os.path.join(DATA_PROCESSED_PATH, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Данные сохранены: {filepath}")
    return filepath


def load_processed_data(filename='processed_data.csv'):
    """Загрузить обработанные данные"""
    filepath = os.path.join(DATA_PROCESSED_PATH, filename)
    return pd.read_csv(filepath)


if __name__ == '__main__':
    # Тест загрузки
    df = load_raw_data()
    df = preprocess_data(df)
    validate_data(df)
    save_processed_data(df)