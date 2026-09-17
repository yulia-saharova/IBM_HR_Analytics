# HR Attrition Prediction — IBM HR Analytics

## 1. Цель
Задача — заранее находить сотрудников с высоким риском увольнения, чтобы HR успевал вмешаться (пересмотр нагрузки, грейда, командировок), а не оформлял уход по факту.
Метрика успеха для бизнеса — не accuracy, а умение ловить уходящих при приемлемой цене ложных тревог. Поэтому главные метрики — `PR-AUC` и `F1`, вспомогательная — `ROC-AUC`.

## 2. Данные
- Источник: Kaggle `IBM HR Analytics Employee Attrition` (`data/WA_Fn-UseC_-HR-Employee-Attrition.csv`).
- Размер сырых: `1470 строк x 35 колонок` (26 числовых `int64`, 9 категориальных `str`).
- После очистки: `1470 x 31` (`data/cleaned_data.csv`).
- Пропусков: `0`. Полных дубликатов: `0`.
- Удалены 4 столбца: `EmployeeNumber` (ID, 1470 уникальных), `Over18` (константа `Y`), `StandardHours` (константа `80`), `EmployeeCount` (константа `1`).
- Распределение target `Attrition -> target 0/1`: `No (остался) = 1233 (83.9%)`, `Yes (ушёл) = 237 (16.1%)` — сильный дисбаланс.

![Распределение target](img/distribution_target.png)

## 3. Что сделано по шагам
- `01_data_exploration_and_cleaning.ipynb` — описание 35 признаков, `info/describe/value_counts`, проверка пропусков и констант, удаление 4 мусорных столбцов, сохранение подготовленного датасета `cleaned_data.csv`.
- `02_visualization_and_analysis.ipynb` — EDA: баланс классов, `describe()`, ящики с усами числовых vs target, гистограммы с KDE, корреляционная матрица, срезы категориальных с долями увольнений. Графики сохранены в `img/`.
- `03_feature_engineering_and_prediction.ipynb` — удаление шумовых признаков по ставкам `Daily/Hourly/Monthly` (корреляция с target `-0.057/-0.007/+0.015`), 5 новых флагов  (`IsYoung, HighRisk, JobHopping, BadWorkLife, FarFromHome`), `stratify`-сплит `train 1176 (16.16%) / test 294 (15.99%)`, `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`), 4 пайплайна, выбор лучшей по `PR-AUC`, тюнинг `GridSearchCV`.

## 4. Визуализация и выводы
Числовые признаки vs target — у ушедших сдвиг в сторону меньшего стажа, грейда и дохода:

![Числовые признаки](img/int_features.png)

Корреляции со знаком — все топ-10 отрицательные (`-0.17…-0.10`): `TotalWorkingYears, JobLevel, YearsInCurrentRole, MonthlyIncome, Age, YearsWithCurrManager`. Дольше работаешь, выше грейд и доход — ниже риск.

![Корреляции](img/heatmap_corr.png)

Категориальные группы риска: 
- `OverTime Yes 30.5% vs No 10.4%`, 
- `Travel_Frequently 24.9% vs Non-Travel 8.0%`, 
- `Sales 20.6% / HR 19.0% vs R&D 13.8%`, `Sales Representative 39.8%`, `Laboratory Technician 23.9%`, `Human Resources (сфера) 25.9%` на маленькой группе 27 строк.

![Категориальные признаки](img/categ_features.png)

## 5. Модели и результаты
Сравнение на одном сплите (`random_state=42`):
- `Logistic Regression: F1 0.4923, ROC-AUC 0.8150, PR-AUC 0.5931, precision 0.3855, recall 0.6809`
- `Random Forest: F1 0.4898, ROC-AUC 0.7878, PR-AUC 0.4455`
- `XGBoost: F1 0.4359, ROC-AUC 0.8100, PR-AUC 0.5059`
- `Decision Tree: F1 0.4333, ROC-AUC 0.6518, PR-AUC 0.3517`

Лучшая по `PR-AUC/F1` — `Logistic Regression`. Ее тюнили через `GridSearchCV (cv=5, scoring=average_precision)` по сетке `C / penalty / solver / class_weight`.
Финал на тесте после тюнинга: `precision 0.62 (ушедшие) / recall 0.38 / F1 0.47, ROC-AUC 0.8176, PR-AUC 0.6098, accuracy 0.86`. Читается так: из 10 помеченных реально уйдут ~6, поймаем ~4 из 10 уходящих.
Важности `|coef_|` совпадают с EDA: переработки, роль, командировки, срок с повышения, стаж, число смен компаний.

## 6. Структура и запуск
```
01_data_exploration_and_cleaning.ipynb
02_visualization_and_analysis.ipynb
03_feature_engineering.ipynb
data/*.csv, img/*.png
requirements.txt
```
Запуск:
```
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
jupyter lab
# затем Kernel -> Restart & Run All по порядку 01 -> 02 -> 03
```

## 7. Ограничения и дальше
- Всего 1470 строк, оценка точечная на одном сплите; маленькие срезы шумят.
- Порог 0.5 не калиброван под бюджет HR — двигать по precision/recall.
