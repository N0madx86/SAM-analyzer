from models.bert.predictor import BERTPredictor
from models.lstm.predictor import LSTMPredictor

from models.logisticRegression.predictor import (
    predict as logistic_predict
)

from models.naiveBayes.predictor import (
    predict as naive_bayes_predict
)

from models.aggregation import SentimentAggregator


class SentimentEngine:

    def __init__(self):

        print("Loading sentiment models...")

        self.bert = BERTPredictor()
        self.lstm = LSTMPredictor()

        self.aggregator = SentimentAggregator()

        print("All models loaded")

    def predict(self, text):

        predictions = {
            "bert": self.bert.predict(text),

            "lstm": self.lstm.predict(text),

            "logistic_regression": (
                logistic_predict(text)
            ),

            "naive_bayes": (
                naive_bayes_predict(text)
            )
        }

        final_result = (
            self.aggregator.aggregate(
                predictions
            )
        )

        return {
            "models": predictions,
            "overall": final_result
        }