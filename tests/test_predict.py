
import sys
from pathlib import Path

# Make the src folder accessible
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_PATH / "src"))

from predict import predict_duration


def test_prediction_is_numeric():
    prediction = predict_duration(
        "ambulatory",
        "185347001",
        "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76",
        "2022-03-15 10:30:00"
    )

    assert isinstance(prediction, float)


def test_prediction_is_positive():
    prediction = predict_duration(
        "ambulatory",
        "185347001",
        "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76",
        "2022-03-15 10:30:00"
    )

    assert prediction > 0


def test_unseen_encounter_category():
    prediction = predict_duration(
        "new_category",
        "185347001",
        "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76",
        "2022-03-15 10:30:00"
    )

    assert prediction > 0
