import torch

from models.bert.config import MODEL_PATH
from models.bert.bert_sam import BERTSAM
from models.bert.tokenizer import BERTTokenizer


class BERTPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = BERTTokenizer()

        self.model = BERTSAM().to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        self.model.eval()


    def predict(self, text):

        encoded = self.tokenizer.encode(
            text
        )

        input_ids = encoded["input_ids"].to(
            self.device
        )

        attention_mask = encoded["attention_mask"].to(
            self.device
        )

        # Add batch dimension
        if input_ids.dim() == 1:

            input_ids = input_ids.unsqueeze(0)

            attention_mask = (
                attention_mask.unsqueeze(0)
            )

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probabilities = torch.softmax(
                outputs.logits,
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