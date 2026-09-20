from preprocess import clean_text

from models.naiveBayes.naive_bayes_sam import (
    NaiveBayesSAM
)

from models.naiveBayes.config import (
    MODEL_PATH
)


class NaiveBayesPredictor:

    def __init__(self):

        self.model = NaiveBayesSAM()

        self.model.load(
            MODEL_PATH
        )

    def predict(self, text):

        text = clean_text(text)

        return self.model.predict(
            text
        )


predictor = NaiveBayesPredictor()


def predict(text):

    return predictor.predict(text)