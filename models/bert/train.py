import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset_loader import load_dataset
from models.bert.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    PATIENCE,
    MAX_GRAD_NORM,
    MODEL_PATH,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE
)
from models.bert.tokenizer import BERTTokenizer
from models.bert.dataset import BERTDataset
from models.bert.collator import BERTCollator
from models.bert.bert_sam import BERTSAM
from models.bert.trainer import train_model



def main():

    # ======================================================
    # Device
    # ======================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ======================================================
    # Dataset
    # ======================================================

    data = load_dataset()

    print("Dataset loaded")

    texts = data["review"]

    labels = data["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    # ======================================================
    # Train / Validation / Test Split
    # ======================================================

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    validation_ratio = (
        VALIDATION_SIZE /
        (TEST_SIZE + VALIDATION_SIZE)
    )

    x_validation, x_test, y_validation, y_test = (
        train_test_split(
            x_temp,
            y_temp,
            test_size=1 - validation_ratio,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
    )

    print(
        f"Training samples   : {len(x_train)}"
    )

    print(
        f"Validation samples : {len(x_validation)}"
    )

    print(
        f"Test samples       : {len(x_test)}"
    )

    # ======================================================
    # Tokenizer
    # ======================================================

    tokenizer = BERTTokenizer()

    print("BERT tokenizer loaded")

    # ======================================================
    # Datasets
    # ======================================================

    train_dataset = BERTDataset(
        x_train,
        y_train,
        tokenizer
    )

    validation_dataset = BERTDataset(
        x_validation,
        y_validation,
        tokenizer
    )

    test_dataset = BERTDataset(
        x_test,
        y_test,
        tokenizer
    )

    # ======================================================
    # Collator
    # ======================================================

    collator = BERTCollator(
        tokenizer
    )

    # ======================================================
    # DataLoaders
    # ======================================================

    loader_args = {
        "batch_size": BATCH_SIZE,
        "collate_fn": collator,
        "num_workers": 2,
        "pin_memory": device.type == "cuda"
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args
    )

    print("DataLoaders created")

    # ======================================================
    # Model
    # ======================================================

    model = BERTSAM().to(device)

    print("BERT model loaded")
    print(
        "Model device:",
        next(model.parameters()).device
    )

    # ======================================================
    # Training
    # ======================================================

    print("\nTraining started")

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        patience=PATIENCE,
        max_grad_norm=MAX_GRAD_NORM,
        model_path=MODEL_PATH
    )


if __name__ == "__main__":
    main()