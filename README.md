## Data for ML

Мультиагентная система для полного цикла подготовки данных и обучения модели на текстовых данных.
Реализована как набор Python-агентов + Cursor Agent Skills (совместимо с Claude Code).

---

## Быстрый старт

```bash
pip install -r requirements.txt
```

Создать `.env`:
```
OPENROUTER_API_KEY=sk-or-...      # обязательно — авторазметка и выбор модели
KAGGLE_USERNAME=your_username     # для загрузки датасетов с Kaggle
KAGGLE_KEY=your_key               # https://www.kaggle.com/settings → API → Create New Token
LABEL_STUDIO_API_KEY=...          # для ручной разметки спорных примеров (опционально)
```

Запустить через скрипт:
```bash
python .claude/skills/data-pipeline/scripts/run_pipeline.py "sentiment analysis"
```

Или через Cursor / Claude Code:
```
/data-pipeline sentiment analysis
```

---

## Пайплайн


| Шаг | Агент | Кто решает | Главный результат на диске |
|-----|-------|------------|----------------------------|
| 1 | DataCollector | человек (источники) | `data/raw/combined.parquet` + EDA |
| 2 | DataDetective | человек (стратегия чистки) | `data/cleaned/cleaned.parquet` + отчёт качества |
| 3 | DataAnnotation | человек (таксономия, Label Studio) | `data/labeled/labeled.parquet` + метрики разметки |
| 4 | ActiveLearner | автономно | `data/active_learning/REPORT.md`, `learning_curve.png`, `history_*.json` |
| 5 | ModelTrainer | автономно | `models/final_model.pkl`, `data/model/MODEL_REPORT.md` |
| — | — | — | В конце: **`PIPELINE_REPORT.md`** в корне проекта |


---

## Агенты

### 1. DataCollectionAgent

Поиск датасетов из нескольких источников с унификацией в единую схему.

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config="config.yaml")
df = agent.run(sources=[
    {"type": "hf_dataset", "name": "imdb"},
    {"type": "rss", "url": "https://feeds.bbci.co.uk/news/rss.xml"},
])
```

**4 типа источников:**

| Тип | Описание |
|-----|----------|
| `hf_dataset` | HuggingFace Hub — через metadata API, без скачивания |
| `kaggle_dataset` | Kaggle (требует `KAGGLE_USERNAME` + `KAGGLE_KEY`) |
| `scrape` | Web scraping через BeautifulSoup |
| `rss` / `api` | RSS/Atom ленты или публичные JSON API |

Агент ищет подходящие датасеты через скрипты `search_hf.py` / `search_kaggle.py`, валидирует кандидатов (только метаданные, без загрузки), объединяет отобранные источники в единый DataFrame.

**Единая схема:**

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст |
| `label` | str | Метка класса (может быть пустой) |
| `source` | str | Идентификатор источника (`huggingface:imdb`) |
| `collected_at` | str | Временная метка ISO 8601 |

Настройки в `config.yaml`: `max_samples_per_source`, включение/отключение типов источников.

### 2. DataQualityAgent

Обнаружение и устранение проблем качества данных.

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df)
```

**Обнаруживает 10 типов проблем:**
- Пропущенные значения (nulls в `text` и метаданных, не в `label`)
- Точные дубликаты — одинаковые тексты
- Выбросы по длине — IQR-метод на количестве символов
- Пустые / whitespace-only тексты
- Несогласованные метки (`pos` vs `positive` vs `1`)
- Near-дубликаты — тексты, отличающиеся только пунктуацией
- Encoding noise — HTML-entities, мусорная разметка
- Экстремальные длины (< 20 или > 5000 символов)
- Смешение языков
- Качество по источнику — один источник может давать особенно шумные тексты

**3 стратегии чистки** (`agent.fix(df, strategy)`):

| Стратегия | Пропуски | Дубликаты | Выбросы | Пустые |
|-----------|----------|-----------|---------|--------|
| `aggressive` | Удалить строки | Удалить | Удалить (IQR) | Удалить |
| `balanced` | Заполнить `""` | Удалить | Удалить (z > 3) | Удалить |
| `conservative` | Заполнить `""` | Оставить | Оставить | Оставить |

Нормализация меток применяется **до** стратегии — `positive → pos`, `1 → pos`, `negative → neg`, `0 → neg`.

`agent.compare(df_before, df_after)` — сравнивает метрики до и после чистки.

### 3. AnnotationAgent

Автоматическая разметка текстов через OpenRouter API с контролем качества и интеграцией с Label Studio.

```python
from agents.annotation_agent import AnnotationAgent

agent = AnnotationAgent()
```

**LLM анализирует датасет и предлагает таксономию** — читает 50–100 семплов, оценивает существующие метки, может предложить новые классы, которые лучше соответствуют данным.

**4 варианта использования:**
1. **Разметить всё заново** — с предложенной таксономией
2. **Разметить заново с корректировкой** — пользователь изменяет классы перед запуском
3. **Разметить только неразмеченные** — существующие метки сохраняются (`label_unlabeled.py`)
4. **Оставить метки как есть** — заполнить только пустые строки

Каждая строка получает:

| Колонка | Описание |
|---------|----------|
| `auto_label` | Присвоенная метка |
| `confidence` | Уверенность модели (0.0–1.0) |
| `is_disputed` | `True` если confidence < порога (по умолчанию 0.7) |

**Контроль качества** (`check_quality.py`):
- Cohen's κ — согласованность с исходными метками (если есть)
- Статистики confidence (mean, median, std)
- Количество спорных примеров (low confidence)

**Интеграция с Label Studio:**
- `export_ls.py` — экспорт спорных примеров в JSON с pre-annotations
- `start_ls.py` — запуск Label Studio, создание проекта, импорт задач
- `import_ls.py` — импорт исправлений обратно (confidence → 1.0)

### 4. ActiveLearningAgent

Эксперимент: сколько размеченных примеров реально нужно? Сравнивает стратегии умного отбора.

```python
from agents.active_learning_agent import ActiveLearningAgent

agent = ActiveLearningAgent()
recommendation = agent.select_model(df_labeled, task_type="sentiment analysis")
# → {"model": "logreg", "seed_size": 20, "reasoning": "..."}
```

**OpenRouter API выбирает модель и seed** — отправляет статистику датасета (размер, классы, дисбаланс, длины текстов) и получает рекомендацию.

**3 доступные модели (TF-IDF + sklearn):**

| Ключ | Модель | Когда подходит |
|------|--------|----------------|
| `logreg` | Logistic Regression | Хороший default для текстов |
| `logreg_balanced` | LogReg с `class_weight='balanced'` | Несбалансированные классы |
| `svm` | Linear SVM + Platt scaling | Высокоразмерные данные |

**3 стратегии выборки** (`agent.compare_strategies`):

| Стратегия | Формула | Идея |
|-----------|---------|------|
| `entropy` | H(p) = −Σ pᵢ log pᵢ | Максимальная неопределённость |
| `margin` | p₁ − p₂ (разница top-2) | Модель колеблется между двумя классами |
| `random` | Случайная выборка | Baseline для сравнения |

**Цикл:** фиксированный split (seed + pool + test 20%) → обучение на seed → query из pool → добавление → повторение. На каждой итерации — свежая модель, оценка accuracy + F1 macro.

**Генерирует:** `data/active_learning/learning_curve.png`, `REPORT.md` с анализом экономии, `history_*.json`.

### 5. ModelTrainerAgent

Обучение финальной модели на всех размеченных данных. OpenRouter API выбирает оптимальную модель.

```python
from agents.model_trainer_agent import ModelTrainerAgent

agent = ModelTrainerAgent()
recommendation = agent.select_model(df, task_type="sentiment analysis")
# → {"model": "logreg_balanced", "reasoning": "..."}
result = agent.run(parquet_path="data/labeled/labeled.parquet", task_type="sentiment analysis")
```

**Полный цикл `run()`:**
1. OpenRouter анализирует данные + результаты AL → выбирает модель
2. Stratified train/test split 80/20
3. Обучение TF-IDF (15 000 features, uni+bigrams) + выбранная модель
4. Оценка: accuracy, F1 macro/weighted, precision, recall
5. Визуализации: confusion matrix, per-class F1, top features
6. Сохранение в `models/final_model.pkl` (joblib) + `model_config.json`
7. Генерация `data/model/MODEL_REPORT.md`

---

## Структура проекта

Ниже — ориентир по каталогам и коду. Крупные артефакты (`*.parquet`, `*.pkl`, многие графики) часто не коммитятся: смотри `.gitignore`. После первого прогона пайплайна появятся `data/**`, `models/final_model.pkl`, `PIPELINE_REPORT.md`.

```
data_for_ml/
├── README.md
├── config.yaml
├── requirements.txt
├── PIPELINE_REPORT.md              # итоговый отчёт после полного пайплайна
│
├── agents/                         # классы агентов
│   ├── __init__.py
│   ├── data_collection_agent.py
│   ├── data_quality_agent.py
│   ├── annotation_agent.py
│   ├── active_learning_agent.py
│   ├── model_trainer_agent.py
│   └── openrouter_client.py
│
├── scripts/                        # вспомогательные CLI-утилиты (вне скиллов)
│   ├── run_eda.py
│   ├── make_quality_notebook.py
│   └── test_openrouter_client.py
│
├── data/                           # данные и отчёты по шагам
│   ├── raw/                        # combined.parquet, по источникам
│   ├── cleaned/
│   ├── labeled/
│   ├── eda/                        # REPORT.md, графики EDA
│   ├── detective/                  # QUALITY_REPORT.md, problems.json, графики
│   ├── annotation/                 # метрики, Label Studio, графики
│   ├── active_learning/            # REPORT.md, learning_curve.png, history_*.json
│   └── model/                      # MODEL_REPORT.md, metrics.json, графики
│
├── models/                         # final_model.pkl (bundle), model_config.json
│
├── notebooks/                      # ноутбуки с результатами по шагам
│   ├── eda.ipynb
│   ├── data_quality.ipynb
│   ├── annotation.ipynb
│   ├── al_experiment.ipynb
│   └── model_training.ipynb
│
└── .claude/skills/                 # SKILL.md + скрипты для Cursor / Claude Code
    ├── data-collector/
    │   ├── SKILL.md
    │   └── scripts/
    ├── data-detective/
    │   ├── SKILL.md
    │   └── script/                 # detect, fix, compare, visualize, run_notebook
    ├── data-annotation/
    │   ├── SKILL.md
    │   └── scripts/
    ├── active-learner/
    │   ├── SKILL.md
    │   └── scripts/
    ├── model-trainer/
    │   ├── SKILL.md
    │   └── scripts/
    └── data-pipeline/
        ├── SKILL.md
        └── scripts/
            └── run_pipeline.py     # точка входа «всё сразу» без чата
```

Локально (не в репозитории): `.env` с ключами, виртуальное окружение (`.venv/`), при необходимости `.claude/settings.local.json`.
