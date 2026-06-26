"""
Скрипт для пост-редактирования POS-разметки новых данных.
Сравнивает две модели: TF-IDF+LR и CatBoost.

Использование:
    python run_postedit.py path/to/new_data.xlsx
    
Опции:
    --models tfidf_lr,catboost   какие модели использовать
    --no-gold                    не использовать gold разметку
"""

import sys
import argparse
from src.pipeline.postedit import full_postedit_pipeline

import os

for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy"
]:
    os.environ.pop(key, None)

os.environ["NO_PROXY"] = "*"

def main():
    parser = argparse.ArgumentParser(description='Пост-редактирование POS-разметки')
    parser.add_argument('data_path', help='Путь к новому Excel файлу')
    parser.add_argument('--models', default='tfidf_lr,catboost',
                        help='Модели через запятую: tfidf_lr,catboost')
    parser.add_argument('--no-gold', action='store_true',
                        help='Не использовать gold разметку для оценки')
    
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(',')]
    
    print(f'''
          Данные: {args.data_path}
          Модели: {models}
          Gold разметка: {'нет' if args.no_gold else 'да (если есть)'}
          ''')
    
    df_result, stats = full_postedit_pipeline(
        new_data_path=args.data_path,
        use_gold=not args.no_gold,
        models=models
    )
    
    print(f"Итоговая таблица результатов:")
    if stats:
        for model, metrics in stats.items():
            f1 = metrics.get('f1', 0)
            acc = metrics.get('accuracy', 0)
            print(f"  {model:15}: macro-F1={f1:.4f}, accuracy={acc:.4f}")


if __name__ == '__main__':
    main()