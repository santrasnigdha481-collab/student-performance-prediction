import pandas as pd

df = pd.read_csv("C:\\Users\\santr\\OneDrive\\Desktop\\python\\Student-Performance-Prediction\\Pass-Fail Data.csv")

print(df.head())
print(df.info())
print(df.describe())