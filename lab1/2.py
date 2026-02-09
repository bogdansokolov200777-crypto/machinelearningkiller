import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Таргет: Купит ли клиент товар
df1 = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50],
    'ID_System': [np.nan, 102, np.nan, 105, np.nan, 107],
    'Target': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes']
})
imputer1 = SimpleImputer(strategy='constant', fill_value=-999)
df1[['ID_System']] = imputer1.fit_transform(df1[['ID_System']])
df1['Target'] = df1['Target'].map({'No': 0, 'Yes': 1})
print(df1,"\n")

# Таргет: Уровень подписки (Basic < Silver < Gold — с порядком)
df2 = pd.DataFrame({
    'City': ['Moscow', 'Moscow', 'London', 'Moscow', np.nan, 'Moscow', 'London'],
    'Age': [20, 25, 30, 35, 40, 45, 50],
    'Target': ['Basic', 'Basic', 'Silver', 'Silver', 'Gold', 'Gold', 'Gold']
})
imputer2 = SimpleImputer(strategy='most_frequent')
df2[['City']] = imputer2.fit_transform(df2[['City']])
df2['Target'] = df2['Target'].map({'Basic': 0, 'Silver': 1, 'Gold': 2})
print(df2, "\n")

# Таргет: Группа здоровья (A < B < C — с порядком)
df3 = pd.DataFrame({
    'Pulse': [70, 72, 75, np.nan, 68, 71, 73, 74],
    'Temp': [36.6, 36.7, 36.8, 36.6, 36.9, 36.6, 36.7, 36.8],
    'Target': ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'C']
})

imputer3 = SimpleImputer(strategy='median')
df3[['Pulse']] = imputer3.fit_transform(df3[['Pulse']])
df3['Target'] = df3['Target'].map({'A': 0, 'B': 1, 'C': 2})
print(df3, "\n")

# Таргет: Прошел проверку безопасности (Да/Нет)
df4 = pd.DataFrame({
    'Days_Since_Last_Incident': [10, 5, 20, np.nan, 15, 30],
    'Risk_Score': [0.1, 0.2, 0.1, 0.4, 0.2, 0.1],
    'Target': ['Safe', 'Safe', 'Warning', 'Safe', 'Safe', 'Warning']
})

imputer4 = SimpleImputer(strategy='median')
df4[['Days_Since_Last_Incident']] = imputer4.fit_transform(df4[['Days_Since_Last_Incident']])
df4['Target'] = df4['Target'].map({'Safe': 0, 'Warning': 1})
print(df4, "\n")

# Таргет: Кредитный рейтинг (Low < High — с порядком)
df5 = pd.DataFrame({
    'Bonus_Points': [100, 500, np.nan, 200, np.nan, 800],
    'Salary_K': [50, 100, 40, 120, 30, 150],
    'Target': ['Low', 'High', 'Low', 'High', 'Low', 'High']
})
imputer5 = SimpleImputer(strategy='median')
df5[['Bonus_Points']] = imputer5.fit_transform(df5[['Bonus_Points']])
df5['Target'] = df5['Target'].map({'Low': 0, 'High': 1})
print(df5)