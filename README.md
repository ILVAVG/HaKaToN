# Прогноз режима электрической сети (хакатон)

Модель предсказывает установившийся режим сети 35 узлов / 80 линий:
флаг `regime_valid`, `V` и `Delta` по узлам, токи `I_max` и `I_zarg` по линиям.

Основное решение — sklearn MLP + постобработка (усреднение параллельных линий,
снап квази-константных узлов), финальный инференс — на plain-numpy,
чтобы не зависеть от версии sklearn.

## Структура

```
baseline/
├─ baseline_solution.py      # точка входа для проверки/сабмита
├─ requirements.txt
├─ README.md
├─ src/                      # библиотеки (импортируются скриптами и entrypoint)
│  ├─ grid_lib.py            # данные, сплиты, метрики, параллельные группы
│  ├─ grid_mlp.py            # sklearn MLP-пайплайн + numpy-инференс
│  └─ gnn_model.py           # графовая модель (эксперимент)
├─ scripts/                  # обучение и эксперименты
│  ├─ train_v2.py            # финальное обучение (sklearn)
│  ├─ export_numpy_bundle.py # экспорт в plain-numpy бандл
│  ├─ exp_stage1.py          # быстрые победы + диагностика
│  ├─ exp_stage3.py          # GNN (медленно)
│  ├─ exp_stage4.py          # валидация 3 seed + blocks + extrapolation
│  └─ physics_check.py       # проверка физ. модели сети
├─ data/                     # обучающие CSV (100к опытов)
├─ models/                   # веса
│  ├─ weights_v2_numpy.joblib  # ОСНОВНОЙ (numpy, версионно-независимый)
│  └─ weights_v2.joblib        # sklearn-версия (фоллбэк)
├─ my_test/                  # маленький тест с разметкой
├─ predictions/              # выдача (pred_nodes.csv, pred_lines.csv)
├─ reports/                  # отчёт, метрики (REPORT_v2.md, физика, json)
├─ docs/                     # задание, итоги анализа, конспект статьи
├─ notebooks/                # EDA-ноутбуки
└─ archive/                  # устаревшее (v1, старые веса, GNN-чекпоинты)
```

## Быстрый старт

```bash
# окружение для обучения (Python 3.10 + torch/PyG)
py -3.10 -m venv .venv310
.venv310\Scripts\python -m pip install -r requirements.txt

# инференс не требует torch/sklearn-классов: только numpy/pandas/joblib
python baseline_solution.py --nodes my_test/nodes.csv --lines my_test/lines.csv --out-dir predictions
```

`baseline_solution.py` берёт `models/weights_v2_numpy.joblib`. Если его нет —
`models/weights_v2.joblib`, иначе тривиальный фоллбэк. Для невалидных режимов
выдаются нули.

## Обучение и эксперименты

```bash
.venv310\Scripts\python scripts\exp_stage1.py            # быстрые победы
.venv310\Scripts\python scripts\physics_check.py         # проверка физики
.venv310\Scripts\python scripts\exp_stage4.py            # валидация
.venv310\Scripts\python scripts\train_v2.py --config after
.venv310\Scripts\python scripts\export_numpy_bundle.py
```

Зависимости обучения — в `requirements.txt`. Скрипты сами добавляют `src/` в путь.

## Итоги и ограничения

См. `reports/REPORT_v2.md` (честные метрики, провалы, changelog) и
`reports/physics_check.md`. Кратко: флаг F1≈0.995 на случайном сплите, но есть
сдвиг распределения по ID (блочный сплит F1≈0.87); физика сети не воспроизведена;
GNN на CPU недоучена и не используется.
