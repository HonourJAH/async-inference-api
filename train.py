import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

cats = [
    "sci.space",
    "alt.atheism",
    "talk.religion.misc",
    "soc.religion.christian",
    "sci.med",
]

train = fetch_20newsgroups(
    subset="train", categories=cats, remove=("headers", "footers", "quotes")
)

pipeline = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
            ),
        ),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)


param_grid = {
    "vectorizer__max_features": [10000, 50000],
    "vectorizer__ngram_range": [(1, 1), (1, 2)],
    "classifier__C": [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(train.data, train.target)

joblib.dump(
    {"pipeline": grid.best_estimator_, "categories": train.target_names}, "model.joblib"
)
print("Model saved to model.joblib")
