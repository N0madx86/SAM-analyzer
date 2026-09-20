import torch
import torch.nn as nn



class LSTMSAM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=128,
            padding_idx=0
        )

        #Bi-directional lstm
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        #Attention layer
        self.attention = nn.Linear(
            in_features=256,
            out_features=1
        )

        self.classifier = nn.Linear(
            in_features=256,
            out_features=2
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        x = self.embedding(x)  # Shape: [batch_size, sequence_length, 128]

        # output shape: [batch_size, sequence_length, hidden_size]
        # hidden shape: [1, batch_size, hidden_size]
        output, (hidden, cell) = self.lstm(x)

        scores = self.attention(output)

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            weights * output,
            dim=1
        )

        context = self.dropout(context)

        # Calculate final logits
        predictions = self.classifier(context)

        return predictions

if __name__ == "__main__":
    pass
