import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv("lstm_dataset.csv")

# ------------------------
# FEATURES & LABEL
# ------------------------
X = df.drop("output_label", axis=1).values
y = df["output_label"].values

# ------------------------
# NORMALIZATION
# ------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ------------------------
# RESHAPE FOR LSTM
# (samples, timesteps, features)
# ------------------------
X = X.reshape((X.shape[0], X.shape[1], 1))

# ------------------------
# TRAIN TEST SPLIT
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# MODEL
# ------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------
# TRAIN
# ------------------------
model.fit(X_train, y_train, epochs=10, batch_size=16)

# ------------------------
# SAVE MODEL
# ------------------------
model.save("../models/lstm_model.h5")

print("✅ LSTM Model trained successfully")