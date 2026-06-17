import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("rf_dataset.csv")

# Use ONLY Human & Bot
df = df[df["label"].isin(["Human", "Bot"])]

# Separate numeric and text features
X_num = df[[
    "mouse_moves",
    "clicks",
    "avg_typing_speed",
    "scroll_depth",
    "time_spent"
]].astype(float)

X_text = df["text"].astype(str)

y = df["label"].map({"Human": 0, "Bot": 1})

# Split
X_train_num, X_test_num, X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_num, X_text, y, test_size=0.2, random_state=42
)

# N-gram vectorizer
vectorizer = CountVectorizer(ngram_range=(2, 3))

X_train_text_vec = vectorizer.fit_transform(X_train_text)

# Combine numeric + text features
X_train = hstack([X_train_text_vec, X_train_num.values])

# Train Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/bot_model.pkl")
joblib.dump(vectorizer, "../models/ngram_vectorizer.pkl")

print("Model trained successfully ")
