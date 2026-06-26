"""
Пайплайн пост-редактирования POS-разметки на новых данных.
Сравнивает две модели: TF-IDF+LR и CatBoost.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from stanza.pipeline.core import DownloadMethod


for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy"
]:
    os.environ.pop(key, None)

os.environ["NO_PROXY"] = "*"

def load_new_excel(filepath):
    """
    Загрузить новый Excel с расшифровкой.
    
    Поддерживаемые форматы:
    1. Колонка 'text' - сырой текст, нужна токенизация Stanza
    2. Колонка 'token' - уже токенизированные данные (нужна только POS-разметка)
    3. Полная структура как в обучающем датасете (token, lemma, pos_gold и т.д.)
    """
    df = pd.read_excel(filepath)
    
    print(f'''
          Загружено {len(df)} строк
          Колонки: {list(df.columns)}
          ''')
    
    # Определяем формат
    if 'text' in df.columns and 'token' not in df.columns:
        print("Формат: сырой текст (нужна полная токенизация Stanza)")
        return df, 'text'
    elif 'token' in df.columns:
        print("Формат: токенизированные данные")
        return df, 'token'
    else:
        raise ValueError(f"Не найдена колонка 'text' или 'token'. Колонки: {list(df.columns)}")


# РАЗМЕТКА STANZA

def tag_with_stanza(df, text_column='text'):
    """Применить Stanza для POS-разметки и лемматизации."""
    import stanza
    import os
        
    # Проверяем, есть ли модели локально
    stanza_dir = os.path.expanduser('~/stanza_resources')
    models_exist = os.path.exists(os.path.join(stanza_dir, 'ru'))
    
    if models_exist:
        print(f"Найдены локальные модели в {stanza_dir}")
        print("Пропускаем скачивание...")
    else:
        print("Модели не найдены локально. Пытаемся скачать...")
        try:
            stanza.download('ru', processors='tokenize,pos,lemma', verbose=False)
        except Exception as e:
            print(f" Не удалось скачать модели: {e}")
            raise
    
    # Загружаем pipeline 
    nlp = stanza.Pipeline(
        lang="ru",
        dir=stanza_dir,
        processors="tokenize,pos,lemma",
        download_method=None,
        verbose=False
        )
    
    all_tokens = []
    
    if text_column == 'text' and 'text' in df.columns:
        for idx, row in df.iterrows():
            text = str(row['text'])
            doc = nlp(text)
            
            for sent_idx, sentence in enumerate(doc.sentences):
                for token_idx, token in enumerate(sentence.tokens):
                    word = token.words[0]
                    all_tokens.append({
                        'sentence_id': sent_idx,
                        'token_id': token_idx + 1,
                        'token': word.text,
                        'lemma': word.lemma,
                        'pos_stanza': word.upos,
                        'pos_stanza_raw': word.xpos
                    })
    else:
        if 'sentence_id' in df.columns:
            grouped = df.groupby('sentence_id')
        else:
            df = df.copy()
            df['sentence_id'] = 0
            grouped = df.groupby('sentence_id')
        
        for sent_id, group in grouped:
            text = ' '.join(group['token'].astype(str))
            doc = nlp(text)
            
            token_idx = 0
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    word = token.words[0]
                    if token_idx < len(group):
                        orig_row = group.iloc[token_idx]
                        all_tokens.append({
                            'sentence_id': sent_id,
                            'token_id': token_idx + 1,
                            'token': word.text,
                            'lemma': word.lemma,
                            'pos_stanza': word.upos,
                            'pos_stanza_raw': word.xpos,
                            'token_original': orig_row['token']
                        })
                        token_idx += 1
    
    result_df = pd.DataFrame(all_tokens)
    print(f"Размечено {len(result_df)} токенов в {result_df['sentence_id'].nunique()} предложениях")
    
    return result_df

#ПОСТ-РЕДАКТИРОВАНИЕ 

def postedit_with_tfidf_lr(df):
    """Пост-редактирование с помощью TF-IDF + LR."""
    from src.models.tfidf_lr import load_trained_model
        
    tfidf, model = load_trained_model()
    
    df = df.copy()
    df['token_length'] = df['token'].str.len()
    df = _add_context_features(df)
    
    feature_cols = ['token', 'lemma', 'pos_stanza', 
                   'prev_token', 'next_token', 
                   'prev_pos', 'next_pos', 'token_length']
    
    # Импортируем prepare_features_for_tfidf
    from src.features.feature_engineering import prepare_features_for_tfidf
    df_prep = prepare_features_for_tfidf(df, feature_cols)
    X = df_prep['text_features']
    
    X_tfidf = tfidf.transform(X)
    y_pred = model.predict(X_tfidf)
    
    df['pos_tfidf_lr'] = y_pred
    print(f"TF-IDF + LR: предсказано {len(y_pred)} тегов")
    
    return df


def postedit_with_catboost(df):
    """Пост-редактирование с помощью CatBoost."""
    from src.models.catboost_model import load_trained_model
        
    model = load_trained_model()
    
    df = df.copy()
    df['token_length'] = df['token'].str.len()
    df = _add_context_features(df)
    
    feature_cols = ['token', 'lemma', 'pos_stanza', 
                   'prev_token', 'next_token', 
                   'prev_pos', 'next_pos', 'token_length']
    
    cat_cols = [col for col in feature_cols if col != 'token_length']
    for col in cat_cols:
        df[col] = df[col].fillna('UNK').astype(str)
    df['token_length'] = df['token_length'].fillna(0)
    
    X = df[feature_cols]
    y_pred = model.predict(X)
    
    # преобразуем в 1D массив
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
    elif hasattr(y_pred, 'ravel'):
        y_pred = y_pred.ravel()
    else:
        y_pred = np.array(y_pred).flatten()
    
    df['pos_catboost'] = y_pred
    print(f"CatBoost: предсказано {len(y_pred)} тегов")
    
    return df

def _add_context_features(df):
    """Добавить контекстные признаки (prev/next token и POS)."""
    df = df.copy()
    df['prev_token'] = df.groupby('sentence_id')['token'].shift(1).fillna('START')
    df['next_token'] = df.groupby('sentence_id')['token'].shift(-1).fillna('END')
    df['prev_pos'] = df.groupby('sentence_id')['pos_stanza'].shift(1).fillna('START')
    df['next_pos'] = df.groupby('sentence_id')['pos_stanza'].shift(-1).fillna('END')
    return df


# СРАВНЕНИЕ РЕЗУЛЬТАТОВ 

def compare_results(df, gold_col='pos_gold'):
    """
    Сравнить модели между собой и с gold (если есть).
    
    Returns:
        dict: статистика сравнения
    """
    results = {}
    
    # Если есть gold разметка - считаем метрики
    if gold_col in df.columns:
        print(f"Сравнение с gold разметкой ({gold_col}):")
        
        for model_col in ['pos_stanza', 'pos_tfidf_lr', 'pos_catboost']:
            if model_col in df.columns:
                f1 = f1_score(df[gold_col], df[model_col], average='macro', zero_division=0)
                acc = (df[gold_col] == df[model_col]).mean()
                results[model_col] = {'f1': f1, 'accuracy': acc}
                print(f"  {model_col:15}: macro-F1={f1:.4f}, accuracy={acc:.4f}")
    
    # Попарное сравнение моделей
    print(f"Попарное совпадение предсказаний моделей:")
    model_cols = [col for col in ['pos_tfidf_lr', 'pos_catboost'] if col in df.columns]
    
    for i, col1 in enumerate(model_cols):
        for col2 in model_cols[i+1:]:
            agreement = (df[col1] == df[col2]).mean()
            print(f"  {col1} vs {col2}: {agreement:.2%} совпадений")
    
    # Сравнение со Stanza
    if 'pos_stanza' in df.columns:
        print(f"Сравнение моделей со Stanza:")
        for model_col in model_cols:
            if model_col in df.columns:
                agreement = (df['pos_stanza'] == df[model_col]).mean()
                changes = (df['pos_stanza'] != df[model_col]).sum()
                print(f"  {model_col} vs Stanza: {agreement:.2%} совпадений ({changes} изменений)")
    
    # Сколько токенов обе модели предсказали одинаково
    if len(model_cols) == 2:
        all_agree = (df[model_cols[0]] == df[model_cols[1]]).mean()
        print(f"Обе модели согласны: {all_agree:.2%} токенов")
    
    return results


def show_correction_examples(df, gold_col='pos_gold'):
    """Показать примеры изменений, сделанных моделями."""
    
    model_cols = [col for col in ['pos_tfidf_lr', 'pos_catboost'] if col in df.columns]
    
    for model_col in model_cols:
        print(f"{model_col}:")
        
        if gold_col is not None and gold_col in df.columns:
            # Если есть gold - показываем исправления ошибок Stanza
            fixed = df[
                (df['pos_stanza'] != df[gold_col]) & 
                (df[model_col] == df[gold_col])
            ]
            print(f"Исправлено ошибок Stanza: {len(fixed)}")
            
            if len(fixed) > 0:
                print(f"  Примеры (первые 5):")
                for _, row in fixed.head(5).iterrows():
                    print(f"    '{row['token']}' (лемма: {row['lemma']})")
                    print(f"      Stanza: {row['pos_stanza']:6} → {model_col}: {row[model_col]:6} (gold: {row[gold_col]})")
            
            new_err = df[
                (df['pos_stanza'] == df[gold_col]) & 
                (df[model_col] != df[gold_col])
            ]
            print(f"Новых ошибок: {len(new_err)}")
        else:
            # Если нет gold - показываем все изменения от Stanza
            changes = df[df['pos_stanza'] != df[model_col]]
            print(f"Всего изменений от Stanza: {len(changes)}")
            
            if len(changes) > 0:
                print(f"Примеры изменений (первые 10):")
                for _, row in changes.head(10).iterrows():
                    print(f"    '{row['token']}' (лемма: {row['lemma']})")
                    print(f"      Stanza: {row['pos_stanza']:6} → {model_col}: {row[model_col]:6}")
            
            # Статистика по типам изменений
            if len(changes) > 0:
                print(f"Типы изменений:")
                change_types = changes.groupby(['pos_stanza', model_col]).size().reset_index(name='count')
                change_types = change_types.sort_values('count', ascending=False)
                for _, row in change_types.head(5).iterrows():
                    print(f"    {row['pos_stanza']:6} → {row[model_col]:6} : {row['count']} раз")


def plot_comparison(df, gold_col='pos_gold', save_path='experiments/models_comparison_new.png'):
    """Построить график сравнения моделей."""
    if gold_col not in df.columns:
        print("️  Нет gold разметки - пропускаем визуализацию метрик")
        return
    
    model_cols = [col for col in ['pos_stanza', 'pos_tfidf_lr', 'pos_catboost'] if col in df.columns]
    
    f1_scores = []
    acc_scores = []
    names = []
    
    for col in model_cols:
        f1 = f1_score(df[gold_col], df[col], average='macro', zero_division=0)
        acc = (df[gold_col] == df[col]).mean()
        f1_scores.append(f1)
        acc_scores.append(acc)
        names.append(col.replace('pos_', '').upper())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, f1_scores, width, label='macro-F1', color='steelblue')
    ax1.set_title('macro-F1 по моделям')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    ax2.bar(x + width/2, acc_scores, width, label='Accuracy', color='coral')
    ax2.set_title('Accuracy по моделям')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"График сохранён: {save_path}")


# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ 

def save_results(df, output_path='data/processed/corrected_tags_new.xlsx'):
    """Сохранить результаты с POS от всех моделей."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Выбираем нужные колонки
    cols_to_save = ['sentence_id', 'token_id', 'token', 'lemma', 'pos_stanza']
    
    if 'pos_tfidf_lr' in df.columns:
        cols_to_save.append('pos_tfidf_lr')
    if 'pos_catboost' in df.columns:
        cols_to_save.append('pos_catboost')
    if 'pos_gold' in df.columns:
        cols_to_save.append('pos_gold')
    
    df[cols_to_save].to_excel(output_path, index=False)
    print(f"\n Результаты сохранены: {output_path}")


# ГЛАВНЫЙ ПАЙПЛАЙН 

def full_postedit_pipeline(new_data_path, use_gold=True, models=None):
    """
    Полный пайплайн пост-редактирования.
    
    Args:
        new_data_path: путь к новому Excel
        use_gold: использовать ли gold разметку для оценки
        models: список моделей для применения (по умолчанию ['tfidf_lr', 'catboost'])
    """
    if models is None:
        models = ['tfidf_lr', 'catboost']
    
    print(f"Модели: {', '.join(models)}")
  
    
    # Загрузка
    print("Загрузка новых данных...")
    df_new, format_type = load_new_excel(new_data_path)
    
    # Разметка Stanza
    print("Разметка Stanza (POS + леммы)...")
    if format_type == 'text':
        df_tagged = tag_with_stanza(df_new, text_column='text')
    else:
        df_tagged = tag_with_stanza(df_new, text_column='token')
    
    # Если в новом файле есть gold разметка - добавляем
    if 'pos_gold' in df_new.columns and len(df_new) == len(df_tagged):
        df_tagged['pos_gold'] = df_new['pos_gold'].values
        print(f"Добавлена gold разметка ({len(df_tagged)} токенов)")
    elif 'pos_gold_uni' in df_new.columns and len(df_new) == len(df_tagged):
        # Если есть унифицированные теги - используем их
        df_tagged['pos_gold'] = df_new['pos_gold_uni'].values
        print(f"Добавлена gold разметка из pos_gold_uni ({len(df_tagged)} токенов)")
    
    # Применение моделей
    print("Пост-редактирование моделями...")
    
    df_result = df_tagged.copy()
    
    if 'tfidf_lr' in models:
        try:
            df_result = postedit_with_tfidf_lr(df_result)
        except Exception as e:
            print(f"Ошибка TF-IDF+LR: {e}")
            import traceback
            traceback.print_exc()
    
    if 'catboost' in models:
        try:
            df_result = postedit_with_catboost(df_result)
        except Exception as e:
            print(f"Ошибка CatBoost: {e}")
            import traceback
            traceback.print_exc()
    
    # Сравнение
    print("Сравнение результатов...")
    gold_col = 'pos_gold' if (use_gold and 'pos_gold' in df_result.columns) else None
    
    if gold_col:
        stats = compare_results(df_result, gold_col)
    else:
        print("Gold разметка отсутствует - только попарное сравнение моделей")
        stats = compare_results(df_result)
    
    # Примеры исправлений
    show_correction_examples(df_result, gold_col=None)
    
    # Визуализация
    if gold_col:
        plot_comparison(df_result, gold_col)
    
    # Сохранение
    save_results(df_result)
    
    return df_result, stats