import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#Загрузка данных
df = pd.read_csv('house_price_regression_dataset.csv')
target = df.values[:, -1]
df.drop("House_Price", axis=1, inplace=True)

df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df4 = df.copy()
df5 = df.copy()
print(df.head())

#Предобработка данных
print(df.isna().any())

#Загрузка модели, обучение
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#1 модель
df1.drop("Num_Bathrooms", axis=1, inplace=True)
print(df1.head())
model1 = LinearRegression()
y = np.copy(target)
X1 = df1.values
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y,
                                                    train_size=0.8,
                                                    random_state=42)
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

#2 модель
df2.drop(["Num_Bathrooms", "Garage_Size"], axis=1, inplace=True)
print(df2.head())
model2 = LinearRegression()
y2 = np.copy(target)
X2 = df2.values
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,
                                                    train_size=0.8,
                                                    random_state=42)
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

#3 модель
from sklearn.linear_model import Ridge
X3 = df3.values
y3 = np.copy(target)
model3 = Ridge(alpha=1.0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3,
                                                    train_size=0.8,
                                                    random_state=42)
model3.fit(X3_train, y3_train)
y3_pred = model3.predict(X3_test)


#4 модель
from sklearn.linear_model import Lasso
X4 = df4.values
y4 = np.copy(target)
model4 = Lasso(alpha=0.1)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4,
                                                    train_size=0.8,
                                                    random_state=42)
model4.fit(X4_train, y4_train)
y4_pred = model4.predict(X4_test)

#5 модель
df5.drop(["Garage_Size", "Neighborhood_Quality"], axis=1, inplace=True)
print(df5.head())
y5 = np.copy(target)
X5 = df5.values
X5_train, X5_test, y5_train, y5_test = train_test_split(
    X5, y5,
    train_size=0.8,
    random_state=42
)
scaler5 = StandardScaler()
X5_train_scaled = scaler5.fit_transform(X5_train)
X5_test_scaled = scaler5.transform(X5_test)
model5 = LinearRegression()
model5.fit(X5_train_scaled, y5_train)
y5_pred = model5.predict(X5_test_scaled)

#Расчет метрик
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def print_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'\n{name}:')
    print(f'  MSE:  {mse:,.2f}')
    print(f'  RMSE: {rmse:,.2f}')
    print(f'  MAE:  {mae:,.2f}')
    print(f'  R²:   {r2:.4f}')
    return r2

r2_1 = print_metrics('Модель 1 (без Num_Bathrooms)', y1_test, y1_pred)
r2_2 = print_metrics('Модель 2 (без Num_Bathrooms, Garage_Size)', y2_test, y2_pred)
r2_3 = print_metrics('Модель 3 (Ridge Regression)', y3_test, y3_pred)
r2_4 = print_metrics('Модель 4 (Lasso Regression)', y4_test, y4_pred)
r2_5 = print_metrics('Модель 5 (без Garage_Size, Neighborhood_Quality + StandartScaler)', y5_test, y5_pred)