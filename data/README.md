# Данные

Положите сюда `events.csv` из датасета
[Retailrocket Recommender System](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Через Kaggle CLI:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data --unzip
```

Ожидаемая структура:

```
data/
├── README.md
└── events.csv      # 2 756 101 строка, 5 колонок
```

Исходный CSV/архив **не коммитится** в репозиторий (лицензия CC BY-NC-SA 4.0,
размер ≈ 90 МБ). В git попадают только код, тесты и отчёт.
