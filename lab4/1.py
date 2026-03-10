import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing
import ssl
import certifi


# Теперь загрузка должна работать
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print(f"Формат данных: {df.shape}")
print(df.head())


# Подготовка для градиентного спуска
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values.reshape(-1, 1)

# Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Конвертация в тензоры PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Создание модели с регуляризацией
# L2-регуляризация в PyTorch задается параметром weight_decay в оптимизаторе
model = nn.Linear(X_train.shape[1], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01) # weight_decay = L2
criterion = nn.MSELoss()

# Цикл обучения (Batch Gradient Descent)
epochs = 500
history = []

for epoch in range(epochs):
    # Forward
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

plt.figure(figsize=(15, 5))

# График обучения
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title("Процесс обучения (MSE)")
plt.xlabel("Эпоха")

# Анализ остатков
with torch.no_grad():
    test_preds = model(X_test_t).numpy()
    residuals = y_test - test_preds

plt.subplot(1, 2, 2)
plt.scatter(test_preds, residuals, alpha=0.3, color='teal')
plt.axhline(0, color='red', linestyle='--')
plt.title("Анализ остатков (Residual Analysis)")
plt.xlabel("Предсказанная цена")
plt.ylabel("Ошибка")
plt.show()

# Корреляционная матрица для EDA
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляция признаков")
plt.show()