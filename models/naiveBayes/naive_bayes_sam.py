import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class NaiveBayesSAM:

    def __init__(self):

        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, x, y):

        x_vectors = self.vectorizer.fit_transform(x)

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            x_vectors,
            y,
            test_size=0.3,
            random_state=42
        )

        self.model.fit(
            self.x_train,
            self.y_train
        )

    def evaluate(self):

        predictions = self.model.predict(
            self.x_test
        )

        accuracy = accuracy_score(
            self.y_test,
            predictions
        )

        precision = precision_score(
            self.y_test,
            predictions,
            pos_label="positive"
        )

        recall = recall_score(
            self.y_test,
            predictions,
            pos_label="positive"
        )

        f1 = f1_score(
            self.y_test,
            predictions,
            pos_label="positive"
        )

        matrix = confusion_matrix(
            self.y_test,
            predictions
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report")

        print(
            classification_report(
                self.y_test,
                predictions,
                target_names=[
                    "Negative",
                    "Positive"
                ]
            )
        )

        print("\nConfusion Matrix")

        print(matrix)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def predict(self, text):
        vector = self.vectorizer.transform(
            [text]
        )

        probabilities = self.model.predict_proba(
            vector
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