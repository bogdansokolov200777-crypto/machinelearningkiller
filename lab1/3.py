import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
df6 = pd.DataFrame({
    'Completion_Pct': [10, 25, 45, 50, 75, 85, 95, 100],
    'Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8],
    'Target': ['Low', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High']
})
scaler = MinMaxScaler()
df6[['Completion_Pct', 'Experience_Years']] = scaler.fit_transform(
    df6[['Completion_Pct', 'Experience_Years']]
)
encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df6['Target'] = encoder.fit_transform(df6[['Target']])
print(df6)

df7 = pd.DataFrame({
    'Income_K': [30, 35, 40, 45, 50, 42, 38, 1000],
    'Credit_Score': [600, 620, 640, 610, 650, 630, 615, 800],
    'Target': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
})
scaler = StandardScaler()
df7[['Income_K', 'Credit_Score']] = scaler.fit_transform(
    df7[['Income_K', 'Credit_Score']]
)
df7['Target'] = df7['Target'].map({'No': 0, 'Yes': 1})
print(df7)