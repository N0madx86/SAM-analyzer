import torch


class BERTCollator:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, batch):

        input_ids = [
            item["input_ids"]
            for item in batch
        ]

        attention_masks = [
            item["attention_mask"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]

        encoded = self.tokenizer.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks
            },
            padding=True,
            return_tensors="pt"
        )

        encoded["labels"] = torch.tensor(
            labels,
            dtype=torch.long
        )

        return encoded