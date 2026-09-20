import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoded = self.tokenizer.encode(
            self.texts[idx]
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": self.labels[idx]
        }