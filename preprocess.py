import pandas as pd

# Load Dataset
df = pd.read_csv("C:\\Users\\santr\\OneDrive\\Desktop\\python\\Student-Performance-Prediction\\Pass-Fail Data.csv")

# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicates (if any)
df = df.drop_duplicates()

# Features (Input)
X = df[['attendance_pct',
        'homework_pct',
        'midterm_score',
        'study_hours_per_week']]

# Target (Output)
y = df['pass']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nPreprocessing Completed Successfully!")