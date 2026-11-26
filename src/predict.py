
import joblib
import pandas as pd
from pathlib import Path

# Locate the project directory
PROJECT_PATH = Path(__file__).resolve().parent.parent

# Load the trained model
MODEL_PATH = PROJECT_PATH / "models" / "random_forest_duration_model.joblib"
model = joblib.load(MODEL_PATH)


def predict_duration(encounter_class, code, payer, start):
    """
    Predict recorded encounter duration in minutes.

    Parameters:
        encounter_class: Encounter category
        code: Encounter code
        payer: Payer identifier
        start: Encounter start datetime
    """

    start = pd.to_datetime(start, utc=True)

    input_data = pd.DataFrame([{
        "ENCOUNTERCLASS": encounter_class,
        "CODE": code,
        "PAYER": payer,
        "START_YEAR": start.year,
        "START_MONTH": start.month,
        "START_DAY": start.day,
        "START_HOUR": start.hour,
        "START_DAYOFWEEK": start.dayofweek
    }])

    prediction = model.predict(input_data)[0]

    return round(float(prediction), 2)


if __name__ == "__main__":
    predicted_minutes = predict_duration(
        encounter_class="ambulatory",
        code="185347001",
        payer="b1c428d6-4f07-31e0-90f0-68ffa6ff8c76",
        start="2022-03-15 10:30:00"
    )

    print(f"Predicted encounter duration: {predicted_minutes} minutes")
