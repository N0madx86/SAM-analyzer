from transformers import AutoTokenizer

from models.bert.config import (
    MODEL_NAME,
    MAX_LENGTH
)


class BERTTokenizer:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

    def encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )