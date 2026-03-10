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

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование признаков
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Список значений С
C_values = [0.001, 0.01, 0.1, 1, 10]

# Цикл по значениям C
for C in C_values:
    # Используем solver='saga' для L1 с многоклассовой классификацией
    model = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=10000)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)

    # Получаем коэффициенты
    coef = model.coef_
    n_zero = np.sum(coef == 0)
    print(f"C = {C}, нулевых коэффициентов: {n_zero}, точность: {acc:.4f}")