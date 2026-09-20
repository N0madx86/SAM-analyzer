from preprocess import clean_text

from models.logisticRegression.logistic_reression_sam import (
    LogisticSAM
)

from models.logisticRegression.config import (
    MODEL_PATH
)


class LogisticPredictor:

    def __init__(self):

        self.model = LogisticSAM()

        self.model.load(
            MODEL_PATH
        )

    def predict(self, text):

        text = clean_text(text)

        return self.model.predict(
            text
        )


predictor = LogisticPredictor()


def predict(text):

    return predictor.predict(text)