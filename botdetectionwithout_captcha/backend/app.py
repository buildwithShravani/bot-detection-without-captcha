import random
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
from scipy.sparse import hstack
from tensorflow.keras.models import load_model

# ------------------------
# Flask Setup
# ------------------------
app = Flask(__name__)
CORS(app)

# ------------------------
# FIX FRONTEND PATH (VERY IMPORTANT)
# ------------------------
FRONTEND_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend")
)

# ------------------------
# LOAD MODELS
# ------------------------
try:
    rf_model = joblib.load("../models/bot_model.pkl")
    vectorizer = joblib.load("../models/ngram_vectorizer.pkl")
    lstm_model = load_model("../models/lstm_model.h5")
    cnn_model = load_model("../models/cnn_model.h5")
    print("✅ All models loaded successfully")
except Exception as e:
    print("❌ Model loading error:", e)

# ------------------------
# SERVE FRONTEND
# ------------------------
@app.route("/")
def home():
    return send_from_directory(FRONTEND_FOLDER, "index.htm")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_FOLDER, path)

# ------------------------
# PREDICT API
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # ------------------------
        # RANDOM FOREST + NGRAM
        # ------------------------
        text = data.get("text", "")
        text_vec = vectorizer.transform([text])

        num_features = [[
            float(data.get("mouse_moves", 0)),
            float(data.get("clicks", 0)),
            float(data.get("avg_typing_speed", 0)),
            float(data.get("scroll_depth", 0)),
            float(data.get("time_spent", 0))
        ]]

        rf_features = hstack([text_vec, num_features])

        rf_probs = rf_model.predict_proba(rf_features)[0]
        rf_confidence = float(max(rf_probs))

        # ------------------------
        # LSTM
        # ------------------------
        typing_seq = data.get("typing_seq", [0]*10)

        if len(typing_seq) < 10:
            typing_seq = ([0]*(10-len(typing_seq))) + typing_seq
        else:
            typing_seq = typing_seq[-10:]

        seq_array = np.array(typing_seq).reshape((1,10,1))
        lstm_confidence = float(lstm_model.predict(seq_array, verbose=0)[0][0])
        if np.isnan(lstm_confidence):
            lstm_confidence = 0.5

        # ------------------------
        # CNN
        # ------------------------
        cnn_input = np.array([[
            float(data.get("mouse_moves", 0)),
            float(data.get("clicks", 0)),
            float(data.get("avg_typing_speed", 0)),
            float(data.get("scroll_depth", 0)),
            float(data.get("time_spent", 0))
        ]])

        max_val = np.max(cnn_input)
        if max_val == 0:
            max_val = 1

        cnn_input = cnn_input / max_val
        cnn_input = cnn_input.reshape((1,5,1))

        cnn_confidence = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
        if np.isnan(cnn_confidence):
            cnn_confidence = 0.5

        # ------------------------
        # HUMAN SIGNAL CHECK
        # ------------------------
        human_signals = 0
        if data.get("mouse_moves",0) > 400: human_signals += 1
        if data.get("clicks",0) > 15: human_signals += 1
        if data.get("scroll_depth",0) > 800: human_signals += 1
        if data.get("time_spent",0) > 30: human_signals += 1
        if len(text) > 6: human_signals += 1

        # ------------------------
        # FUSION
        # ------------------------
        final_confidence = (
            0.5 * rf_confidence +
            0.3 * lstm_confidence +
            0.2 * cnn_confidence
        )
        variation = random.uniform(-0.05, 0.05)
        final_confidence += variation
        

        if np.isnan(final_confidence):
            final_confidence = 0.5

        if human_signals >= 3:
            final_confidence -= random.uniform(0.05,0.15)

        if(
            data.get("mouse_moves",0) > 50 and
            data.get("scroll_depth",0) > 100 and
            data.get("time_spent",0) > 5
        ):
            final_confidence -= random.uniform(0.02, 0.08)
            final_confidence = max(0,min(1, final_confidence))

        # ------------------------
        # FINAL RESULT
        # ------------------------
        if final_confidence >= 0.9:
            result = "Bot"
            action = "BLOCK"
        elif final_confidence >= 0.8:
            result = "Bot"
            action = "RESTRICT"
        else:
            result = "Human"
            action = "ALLOW"

        return jsonify({
            "result": result,
            "confidence": round(final_confidence, 2),
            "action": action
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)})

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)