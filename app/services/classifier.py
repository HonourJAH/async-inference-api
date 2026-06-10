import joblib

data = joblib.load("model.joblib")
pipeline = data["pipeline"]
categories = data["categories"]


def predict(text: str) -> dict:
    """Run the text classification pipeline on a single input.

    Returns the predicted category and the confidence score
    so the worker has everything it needs to store the result.
    """
    result = pipeline.predict([text])
    probs = pipeline.predict_proba([text])
    category = categories[result[0]]
    confidence = probs[0][result[0]]

    return {
        "category": category,
        "confidence": round(float(confidence), 4),
    }
