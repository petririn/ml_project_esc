# Постредактирование автоматической морфологической разметки устной русской речи

## Автор проекта

**Ирина Петрова** 

Проект является продолжением сравнительного анализа морфологических парсеров для устной русской речи.

## Задача
Улучшить качество POS-разметки расшифровок устной русской речи, полученной автоматическим парсером Stanza, путём пост-редактирования с помощью моделей машинного обучения, учитывающих лексический и контекстуальный контекст.

## Гипотеза
Автоматическая разметка Stanza может быть улучшена с помощью моделей машинного обучения, использующих контекстные признаки.

## Данные
- Источник: ручная расшифровок устной русской речи (2476 токенов, 340 предложений)
- Target: `pos_gold` (золотой стандарт POS)
- Features: token, lemma, POS от Stanza, контекст (prev/next token и POS), длина токена

## Baseline
Stanza (автоматическая разметка, парсер был выбран в результате сравнения нескольких парсеров на этой же выборке)

## Модели
1. TF-IDF + Logistic Regression
2. CatBoostClassifier

## Метрика
macro-F1

```mermaid
flowchart LR
    A[Raw Data<br/>Excel: stanza_and_manual_pos.xlsx] --> B[EDA<br/>POS distribution, errors]
    B --> C[ Stanza Baseline<br/>auto POS tagging]
    C --> D[️ Feature Engineering<br/>token, lemma, context, length]
    D --> E[TF-IDF + LR<br/>3 iterations]
    D --> F[CatBoost<br/>3 iterations]
    E --> G[Evaluation<br/>macro-F1]
    F --> G
    G --> H[Comparison<br/>vs Baseline]
    H --> I[Error Analysis<br/>confusion matrix]
    I --> J[Post-editing<br/>new data pipeline]

## Запуск
```bash
1. Установка зависимостей
pip install -r requirements.txt

2. Обучение моделей на обучающем датасете
# Загрузка и валидация данных
python -m src.data.loader

# Разведочный анализ (EDA)
python -m src.data.eda

# Оценка baseline (Stanza)
python -m src.models.baseline

# Обучение TF-IDF + Logistic Regression (3 итерации)
python -m src.models.tfidf_lr

# Обучение CatBoost (3 итерации)
python -m src.models.catboost_model

После обучения в experiments/trained_models/ появятся файлы:
tfidf_lr_full.pkl
catboost_full.pkl

3. Пост-редактирование новых данных
# Все модели (TF-IDF+LR и CatBoost)
python run_postedit.py data/raw/new_data.xlsx --no-gold

# Только одна модель
python run_postedit.py data/raw/new_data.xlsx --models tfidf_lr

# С gold-разметкой (если есть колонка pos_gold)
python run_postedit.py data/raw/new_data.xlsx

4. Анализ ошибок
python -m src.models.error_analysis

## Использование ИИ

Проект выполнен индивидуально. В процессе работы использовался **Qwen3.7** для:
- Генерации архитектурной схемы (Mermaid)
- Написания шаблонного кода модулей
- Отладки ошибок 
- Для настройки `.gitignore`
- Адаптации кода под конкретные данные (исправление системы POS-тегов, контекстных признаков)

Весь код был проверен, адаптирован под конкретные данные и запущен вручную. ИИ-артефакты удалены, логика прокомментирована. Все решения (выбор метрики, признаков, моделей) принимались самостоятельно.