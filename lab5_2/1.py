import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Загрузка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# 0: 'malignant' (злокачественная), 1: 'benign' (доброкачественная)

X.head()

print(X.isnull().sum())

# Трейн тест сплит
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализация и обучение модели
# C — параметр регуляризации (чем меньше, тем сильнее штраф за большие веса)
model = LogisticRegression(C=1.0, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Предсказание
y_pred = model.predict(X_test_scaled)

# Оценка результатов
print(f"Общая точность (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nДетальный отчет по метрикам:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Интерпретация: какие признаки важнее всего?
weights = pd.Series(model.coef_[0], index=data.feature_names)
top_weights = weights.sort_values(ascending=False).head(5)
print("\nТоп-5 признаков, влияющих на решение о доброкачественности:")
print(top_weights)