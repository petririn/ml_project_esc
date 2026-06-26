"""Разведочный анализ данных (EDA)"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_pos_distribution(df, pos_col='pos_gold', title='POS Distribution'):
    """Анализ распределения POS-тегов"""
    counts = df[pos_col].value_counts()
    print(f"\nВсего уникальных тегов: {len(counts)}")
    print(counts.head(15))
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    counts.head(20).plot(kind='bar', color='steelblue')
    plt.title(title)
    plt.xlabel('POS Tag')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'data/processed/{pos_col}_distribution.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return counts


def analyze_stanza_errors(df):
    """Анализ ошибок Stanza"""
    df = df.copy()
    df['stanza_match_correct'] = (df['pos_gold'] == df['pos_stanza']).astype(int)
    
    total = len(df)
    correct = df['stanza_match_correct'].sum()
    errors = total - correct
    accuracy = correct / total * 100
    
    print(f'''
    Всего токенов: {total}
    Правильно размечено: {correct} ({accuracy:.2f}%)
    Ошибок: {errors} ({100-accuracy:.2f}%)
    ''')
    
    # Какие POS Stanza путает чаще всего
    errors_df = df[df['stanza_match_correct'] == 0]
    
    print(f"\nТоп-10 ошибок (gold -> stanza):")
    error_pairs = list(zip(errors_df['pos_gold'], errors_df['pos_stanza']))
    error_counter = Counter(error_pairs)
    
    for (gold, stanza), count in error_counter.most_common(10):
        print(f"  {gold:8} -> {stanza:8} : {count} раз")
    
    return errors_df


def analyze_token_lengths(df):
    """Анализ длин токенов"""
    
    lengths = df['token_length']
    
    print(f'''
    Минимальная длина: {lengths.min()}
    Максимальная длина: {lengths.max()}
    Средняя длина: {lengths.mean():.2f}
    Медиана: {lengths.median():.0f}
    ''')
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=30, color='steelblue', edgecolor='black')
    plt.title('Distribution of Token Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('data/processed/token_length_distribution.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return lengths.describe()


def full_eda(df):
    """Полный разведочный анализ"""
    
    # 1. Общая информация
    print(f'''
    Размер: {df.shape}
    Первые 5 строк: {df.head()}
    Информация о типах {df.dtypes}
    Пропуски: {df.isnull().sum()}
    ''')
    
    # 2. Распределение POS
    analyze_pos_distribution(df, 'pos_gold', 'Gold POS Distribution (Universal Dependencies)')
    analyze_pos_distribution(df, 'pos_stanza', 'Stanza POS Distribution (Universal Dependencies)')
    
    # 3. Ошибки Stanza
    analyze_stanza_errors(df)
    
    # 4. Длины токенов
    analyze_token_lengths(df)


if __name__ == '__main__':
    from src.data.loader import load_raw_data, preprocess_data
    
    df = load_raw_data()
    df = preprocess_data(df)
    full_eda(df)