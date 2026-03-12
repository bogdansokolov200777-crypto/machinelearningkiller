import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# Загружаем датасет с рукописными цифрами
digits = load_digits()
X = digits.data  # Матрица признаков (64 пикселя на каждое изображение 8x8)
y = digits.target


# Превращаем задачу в бинарную:
# Пусть 1 — это цифра '5', а 0 — любая другая цифра
y_binary = (y == 5).astype(int)

print(f"Всего объектов: {len(X)}")
print(f"Всего цифр '5': {sum(y_binary)}")

# Трейн тест сплит
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Линейным моделям необходимо масштабирование пикселей (от 0..16 к -1..1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучаем логистическую регрессию
# class_weight='balanced' важен, так как "пятерок" в 9 раз меньше, чем остальных цифр
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)


# Оценка
y_pred = model.predict(X_test_scaled)

print("Отчет о классификации:")
print(classification_report(y_test, y_pred, target_names=['Не пятёрка', 'Пятёрка']))

# 7. Посмотрим на веса (какие пиксели важнее всего для распознавания цифры 5)
# Мы можем восстановить форму 8x8 из весов модели
coef_image = model.coef_.reshape(8, 8)
print("\nМатрица весов (влияние пикселей):")
print(np.round(coef_image, 1))