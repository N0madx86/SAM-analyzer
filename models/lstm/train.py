import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset
from preprocess import clean_text

from models.lstm.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_FEATURES,
    MIN_FREQ,
    MAX_LENGTH,
    RANDOM_STATE,
    PATIENCE,
    MODEL_PATH,
    VOCAB_PATH
)

from models.lstm.dataset import ReviewData
from models.lstm.lstm_sam import LSTMSAM
from models.lstm.tokenizer import Tokenizer
from models.lstm.trainer import train_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Dataset
    data = load_dataset()
    print("Dataset loaded")

    data["clean_review"] = data["review"].apply(clean_text)

    data["label"] = data["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    # Tokenizer
    tokenizer = Tokenizer(
        max_features=MAX_FEATURES,
        min_freq=MIN_FREQ,
        max_length=MAX_LENGTH
    )

    tokenizer.build_vocab(
        data["clean_review"]
    )

    tokenizer.save_vocab(
        VOCAB_PATH
    )

    print("LSTM vocabulary saved")

    data["sequence"] = data["clean_review"].apply(
        tokenizer.text_to_sequence
    )

    padded_sequences = tokenizer.pad_sequences(
        data["sequence"].tolist()
    )

    # Split
    x_train, x_temp, y_train, y_temp = train_test_split(
        padded_sequences,
        data["label"].tolist(),
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    x_validation, _, y_validation, _ = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # Tensors
    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_validation = torch.tensor(
        x_validation,
        dtype=torch.long
    )

    y_validation = torch.tensor(
        y_validation,
        dtype=torch.long
    )

    # DataLoaders
    train_loader = DataLoader(
        ReviewData(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        ReviewData(x_validation, y_validation),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Model
    model = LSTMSAM(
        vocab_size=len(tokenizer.word_to_index)
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    print("\nTraining Started\n")

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=PATIENCE,
        model_path=MODEL_PATH,
        device=device
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()