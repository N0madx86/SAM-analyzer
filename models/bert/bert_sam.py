import torch.nn as nn

from transformers import AutoModelForSequenceClassification

from models.bert.config import (
    MODEL_NAME,
    NUM_LABELS
)


class BERTSAM(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )