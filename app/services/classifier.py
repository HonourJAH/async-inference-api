import joblib
import os

_model = None


def get_model():
    """Loads the model only when explicitly called."""
    global _model
    if _model is None:
        # Check if file exists to prevent hard crashes during testing
        if not os.path.exists("model.joblib"):
            raise FileNotFoundError(
                "Model file missing. Are you in a test environment?"
            )
        _model = joblib.load("model.joblib")
    return _model


def predict(text: str) -> dict:
    """Run the text classification pipeline on a single input.

    Returns the predicted category and the confidence score
    so the worker has everything it needs to store the result.
    """

    data = get_model()
    pipeline = data["pipeline"]
    categories = data["categories"]

    result = pipeline.predict([text])
    probs = pipeline.predict_proba([text])
    category = categories[result[0]]
    confidence = probs[0][result[0]]

    return {
        "category": category,
        "confidence": round(float(confidence), 4),
    }
