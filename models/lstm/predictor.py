import torch

from models.lstm.config import (
    MAX_FEATURES,
    MIN_FREQ,
    MAX_LENGTH,
    MODEL_PATH,
    VOCAB_PATH
)

from models.lstm.lstm_sam import LSTMSAM
from models.lstm.tokenizer import Tokenizer

from preprocess import clean_text


class LSTMPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Tokenizer
        self.tokenizer = Tokenizer(
            max_features=MAX_FEATURES,
            min_freq=MIN_FREQ,
            max_length=MAX_LENGTH
        )

        self.tokenizer.load_vocab(
            VOCAB_PATH
        )

        print("LSTM vocabulary loaded")

        # Model
        self.model = LSTMSAM(
            vocab_size=len(
                self.tokenizer.word_to_index
            )
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        self.model.eval()

        print("LSTM model loaded")

    def predict(self, text):

        text = clean_text(text)

        sequence = (
            self.tokenizer.text_to_sequence(text)
        )

        padded = self.tokenizer.pad_sequences(
            [sequence]
        )

        x = torch.tensor(
            padded,
            dtype=torch.long
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model(x)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0, prediction
            ].item()

        sentiment = (
            "positive"
            if prediction == 1
            else "negative"
        )

        return {
            "sentiment": sentiment,
            "confidence": confidence
        }