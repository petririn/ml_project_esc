"""Константы проекта"""

import os

# Базовый путь проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Колонки датасета — ИСПРАВЛЕНО: используем унифицированные теги
COL_TOKEN = 'token'
COL_LEMMA = 'lemma_manual'
COL_POS_GOLD = 'pos_gold_uni'      # ✅ Унифицированные теги ручной разметки
COL_POS_STANZA = 'stanza_uni'      # ✅ Унифицированные теги Stanza
COL_STANZA_MATCH = 'stanza_match'

# Признаки
CAT_FEATURES = ['token', 'lemma', 'pos_stanza', 'prev_token', 'next_token', 'prev_pos', 'next_pos']
NUM_FEATURES = ['token_length']

# Пути
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data/raw/stanza_and_manual_pos.xlsx')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data/processed')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'experiments/results.csv')