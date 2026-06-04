import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("C:\\Users\\santr\\OneDrive\\Desktop\\python\\Student-Performance-Prediction\\Pass-Fail Data.csv")

# Features
X = df[['attendance_pct',
        'homework_pct',
        'midterm_score',
        'study_hours_per_week']]

# Target
y = df['pass']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Trained Successfully!")
print("Accuracy:", accuracy)


import joblib

joblib.dump(model, "student_model.pkl")

print("Model Saved Successfully!")