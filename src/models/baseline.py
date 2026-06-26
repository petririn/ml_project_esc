"""Baseline модель - Stanza"""

from src.utils.metrics import compute_macro_f1, save_results


def evaluate_baseline(df):
    """
    Оценить baseline (Stanza).
    
    Returns:
        float: macro-F1 score
    """
    df = df.copy()
    # Пересчитываем корректно
    df['stanza_match_correct'] = (df['pos_gold'] == df['pos_stanza']).astype(int)
    
    f1 = compute_macro_f1(df['pos_gold'], df['pos_stanza'])
    accuracy = df['stanza_match_correct'].mean()
    
    print(f'''
          Stanza macro-F1: {f1:.4f}
          Stanza accuracy: {accuracy:.4f}
          ''')
    
    # Сохраняем результат
    save_results(
        experiment_name='baseline',
        model_name='Stanza',
        features_used='stanza_uni',
        macro_f1=round(f1, 4)
    )
    
    return f1


if __name__ == '__main__':
    from src.data.loader import load_raw_data, preprocess_data
    
    df = load_raw_data()
    df = preprocess_data(df)
    evaluate_baseline(df)