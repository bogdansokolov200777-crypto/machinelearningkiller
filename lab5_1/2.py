import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(x_train)
X_test_std = sc.transform(x_test)

model1 = LogisticRegression(C=0.01, penalty='l1', solver='saga', max_iter=10000)
model2 = LogisticRegression(C = 0.01, penalty='l2')

model1.fit(X_train_std, y_train)
model2.fit(X_train_std, y_train)
model1.score(X_train_std, y_train)
model2.score(X_test_std, y_test)
# Оцениваем точность на тестовых данных для обеих моделей
accuracy_l1 = model1.score(X_test_std, y_test)
accuracy_l2 = model2.score(X_test_std, y_test)

print(f"\nТочность L1 модели: {accuracy_l1:.4f}")
print(f"Точность L2 модели: {accuracy_l2:.4f}")
print(model1.score(X_train_std, y_train))
print(model2.score(X_train_std, y_train))
# Получаем и выводим веса
weights_l1 = model1.coef_
weights_l2 = model2.coef_

print("Веса L1 модели , C=0.01):")
print(weights_l1)
print("\nВеса L2 модели (с сильной регуляризацией, C=0.01):")
print(weights_l2)