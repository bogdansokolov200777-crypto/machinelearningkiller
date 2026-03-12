import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV

# Загрузка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# 0: malignant (злокачественная), 1: benign (доброкачественная)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Масштабирование признаков (важно для логистической регрессии)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Для подбора порога дополнительно разделим обучающую выборку на train и validation
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 1. Baseline: логистическая регрессия без взвешивания классов
model_baseline = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
model_baseline.fit(X_train_part, y_train_part)
y_pred_baseline = model_baseline.predict(X_test_scaled)
recall_baseline = recall_score(y_test, y_pred_baseline, pos_label=0)  # recall для класса 0 (malignant)
print("=== Baseline (без взвешивания) ===")
print(f"Recall для malignant: {recall_baseline:.4f}")
print(classification_report(y_test, y_pred_baseline, target_names=data.target_names))
print("-" * 60)

# 2. Модель с автоматическим взвешиванием классов (class_weight='balanced')
model_balanced = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
model_balanced.fit(X_train_part, y_train_part)
y_pred_balanced = model_balanced.predict(X_test_scaled)
recall_balanced = recall_score(y_test, y_pred_balanced, pos_label=0)
print("=== Со взвешиванием классов (balanced) ===")
print(f"Recall для malignant: {recall_balanced:.4f}")
print(classification_report(y_test, y_pred_balanced, target_names=data.target_names))
print("-" * 60)

# 3. Подбор порога классификации для максимизации recall по malignant
#    (используем валидационную выборку)
# Получаем вероятности для класса 0 (malignant) на валидации
probs_val = model_balanced.predict_proba(X_val)[:, 0]  # вероятность malignant

# Перебираем пороги от 0.1 до 0.9 и ищем максимум recall для класса 0
thresholds = np.linspace(0.1, 0.9, 50)
best_thresh = 0.5
best_recall_val = 0.0
recalls_val = []

for thresh in thresholds:
    # Если вероятность malignant >= thresh, предсказываем 0, иначе 1
    pred_val = (probs_val >= thresh).astype(int)
    recall_val = recall_score(y_val, pred_val, pos_label=0)
    recalls_val.append(recall_val)
    if recall_val > best_recall_val:
        best_recall_val = recall_val
        best_thresh = thresh

print(f"Лучший порог на валидации: {best_thresh:.3f}, recall на валидации: {best_recall_val:.4f}")

# Применяем лучший порог к тесту
probs_test = model_balanced.predict_proba(X_test_scaled)[:, 0]
y_pred_thresh = (probs_test >= best_thresh).astype(int)
recall_thresh = recall_score(y_test, y_pred_thresh, pos_label=0)
print("=== Взвешенная модель + оптимальный порог ===")
print(f"Recall для malignant на тесте: {recall_thresh:.4f}")
print(classification_report(y_test, y_pred_thresh, target_names=data.target_names))
print("-" * 60)

# 4. Отбор признаков + взвешивание + подбор порога
# Используем SelectKBest для отбора 10 лучших признаков
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_part, y_train_part)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test_scaled)

# Обучаем модель с class_weight='balanced' на отобранных признаках
model_sel = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
model_sel.fit(X_train_selected, y_train_part)

# Подбор порога на валидации
probs_val_sel = model_sel.predict_proba(X_val_selected)[:, 0]
best_thresh_sel = 0.5
best_recall_val_sel = 0.0
for thresh in thresholds:
    pred_val_sel = (probs_val_sel >= thresh).astype(int)
    recall_val_sel = recall_score(y_val, pred_val_sel, pos_label=0)
    if recall_val_sel > best_recall_val_sel:
        best_recall_val_sel = recall_val_sel
        best_thresh_sel = thresh

print(f"Лучший порог на валидации (отбор признаков): {best_thresh_sel:.3f}, recall: {best_recall_val_sel:.4f}")

# Оценка на тесте
probs_test_sel = model_sel.predict_proba(X_test_selected)[:, 0]
y_pred_sel_thresh = (probs_test_sel >= best_thresh_sel).astype(int)
recall_sel = recall_score(y_test, y_pred_sel_thresh, pos_label=0)
print("=== Отбор признаков (топ-10) + взвешивание + порог ===")
print(f"Recall для malignant на тесте: {recall_sel:.4f}")
print(classification_report(y_test, y_pred_sel_thresh, target_names=data.target_names))
print("-" * 60)




# Выводы: выбираем лучшую стратегию по recall для malignant на тесте
results = {
    'Baseline': recall_baseline,
    'Balanced': recall_balanced,
    'Balanced + threshold': recall_thresh,
    'Feature selection + threshold': recall_sel,
}

best_strategy = max(results, key=results.get)
print("\n=== Сравнение стратегий (recall для malignant) ===")
for name, score in results.items():
    print(f"{name}: {score:.4f}")
print(f"\nЛучшая стратегия: {best_strategy} с recall = {results[best_strategy]:.4f}")