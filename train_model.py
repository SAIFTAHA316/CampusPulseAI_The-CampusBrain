import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
import joblib
import os
import json

# -----------------------------
# 1. Load REAL Dataset
# -----------------------------

train_df = pd.read_csv("data/datatraining.txt")
test_df = pd.read_csv("data/datatest.txt")

# Drop date column
train_df = train_df.drop(columns=["date"])
test_df = test_df.drop(columns=["date"])

X_train = train_df.drop(columns=["Occupancy"])
y_train = train_df["Occupancy"]

X_test = test_df.drop(columns=["Occupancy"])
y_test = test_df["Occupancy"]

# -----------------------------
# 2. Train Model
# -----------------------------

model = LGBMClassifier()
model.fit(X_train, y_train)

print("Model trained successfully.")

# -----------------------------
# 3. Evaluate Model
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Model Performance ===")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# Save metrics to JSON
os.makedirs("model", exist_ok=True)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open("model/model_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to model/model_metrics.json")

# -----------------------------
# 4. Save Model
# -----------------------------

joblib.dump(model, "model/occupancy_model.pkl")

print("Model saved successfully in model/occupancy_model.pkl")