import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ------------------------
# LOAD DATASET
# ------------------------
df = pd.read_csv("rf_dataset.csv")

# If your column is output_label → rename it
if "output_label" in df.columns:
    df.rename(columns={"output_label": "label"}, inplace=True)

# Convert labels to 0/1 (VERY IMPORTANT)
df["label"] = df["label"].astype(int)

y_true = df["label"].values

# ------------------------
# LOAD MODEL + VECTORIZER
# ------------------------
rf_model = joblib.load("../models/bot_model.pkl")
vectorizer = joblib.load("../models/ngram_vectorizer.pkl")

# ------------------------
# HANDLE TEXT COLUMN
# ------------------------
if "text" not in df.columns:
    df["text"] = "default text"

text_vec = vectorizer.transform(df["text"])

# ------------------------
# NUMERIC FEATURES
# ------------------------
required_cols = [
    "mouse_moves",
    "clicks",
    "avg_typing_speed",
    "scroll_depth",
    "time_spent"
]

# If any column missing → fill with 0
for col in required_cols:
    if col not in df.columns:
        df[col] = 0

num_features = df[required_cols].fillna(0).values

# ------------------------
# COMBINE FEATURES
# ------------------------
X = hstack([text_vec, num_features])

# ------------------------
# FINAL PREDICTION (NO ERROR)
# ------------------------
y_pred = (rf_model.predict_proba(X)[:,1] > 0.3).astype(int).ravel()  # ✅ correct (NOT predict_proba)

# ------------------------
# METRICS
# ------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

# ------------------------
# OUTPUT
# ------------------------
print("\n===== FINAL SYSTEM EVALUATION =====")
print("Accuracy :", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall   :", round(recall, 2))

print("\nConfusion Matrix:")
print(cm)