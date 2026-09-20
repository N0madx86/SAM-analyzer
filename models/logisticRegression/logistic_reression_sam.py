import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


class LogisticSAM:

    def __init__(self):

        self.vectorizer = CountVectorizer()

        self.model = LogisticRegression(
            max_iter=1000
        )

    def train(self, x_train, y_train):

        x_train_vectors = (
            self.vectorizer.fit_transform(x_train)
        )

        self.model.fit(
            x_train_vectors,
            y_train
        )

    def predict(self, text):
        x_vector = self.vectorizer.transform(
            [text]
        )

        probabilities = self.model.predict_proba(
            x_vector
        )[0]

        prediction_index = probabilities.argmax()

        prediction = self.model.classes_[
            prediction_index
        ]

        confidence = probabilities[
            prediction_index
        ]

        return {
            "sentiment": str(prediction),
            "confidence": float(confidence)
        }

    def save(self, path):

        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "model": self.model
            },
            path
        )

    def load(self, path):

        checkpoint = joblib.load(path)

        self.vectorizer = checkpoint["vectorizer"]
        self.model = checkpoint["model"]