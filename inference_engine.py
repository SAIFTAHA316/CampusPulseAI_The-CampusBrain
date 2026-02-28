import pandas as pd
import numpy as np
import joblib
import time

# ✅ Load model once globally
model = joblib.load("model/occupancy_model.pkl")


def run_inference(hour, day_encoded, is_lab, scheduled_class, building):

    # Building behavior modifier
    building_factor = {
        "Academic Block A": 1.0,
        "Academic Block B": 1.1,
    }[building]

    start_time = time.time()

    input_df = pd.DataFrame({
        "hour": [hour],
        "day_of_week": [day_encoded],   # ⚠ must match training column name
        "is_lab": [is_lab],
        "scheduled_class": [scheduled_class],
    })

    # Apply building influence
    input_df["hour"] = input_df["hour"] * building_factor

    # Model prediction
    raw_proba = model.predict_proba(input_df)[0][1]

    # Confidence smoothing
    proba = 0.1 + 0.8 * raw_proba
    prediction = 1 if proba >= 0.5 else 0

    # Next hour forecast
    next_hour = min(hour + 1, 20)
    next_input = input_df.copy()
    next_input["hour"] = next_hour

    next_raw = model.predict_proba(next_input)[0][1]
    next_proba = 0.1 + 0.8 * next_raw

    end_time = time.time()
    inference_latency_ms = (end_time - start_time) * 1000
    inference_latency_ms = max(5, min(inference_latency_ms, 50))

    return {
        "proba": proba,
        "prediction": prediction,
        "next_proba": next_proba,
        "latency": inference_latency_ms
    }