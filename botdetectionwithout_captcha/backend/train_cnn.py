import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("cnn_dataset.csv")

# Use behavioral numeric features
X = df[[
    "mouse_moves",
    "clicks",
    "avg_typing_speed",
    "scroll_depth",
    "time_spent"
]].values

# Normalize values (important)
X = X / np.max(X, axis=0)

# Reshape for CNN (samples, timesteps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Labels
y = df["label"].map({"Human": 0, "Bot": 1}).values

# -----------------------
# Build CNN model
# -----------------------
model = Sequential([
    Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(5,1)),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# Train
# -----------------------
model.fit(X, y, epochs=15, batch_size=8)

# -----------------------
# Save model
# -----------------------
model.save("../models/cnn_model.h5")

print("CNN model trained successfully")