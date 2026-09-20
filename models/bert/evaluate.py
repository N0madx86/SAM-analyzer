import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset

from models.bert.config import (
    BATCH_SIZE,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    MODEL_PATH
)

from models.bert.tokenizer import BERTTokenizer
from models.bert.dataset import BERTDataset
from models.bert.collator import BERTCollator
from models.bert.bert_sam import BERTSAM

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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

    texts = data["review"]

    labels = data["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    # ======================================================
    # Recreate Same Test Split
    # ======================================================

    _, x_temp, _, y_temp = train_test_split(
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

    _, x_test, _, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1 - validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Test samples: {len(x_test)}")

    # ======================================================
    # Tokenizer
    # ======================================================

    tokenizer = BERTTokenizer()

    # ======================================================
    # Test Dataset
    # ======================================================

    test_dataset = BERTDataset(
        x_test,
        y_test,
        tokenizer
    )

    collator = BERTCollator(
        tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=device.type == "cuda"
    )

    # ======================================================
    # Load Model
    # ======================================================

    model = BERTSAM().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print("Best checkpoint loaded")

    # ======================================================
    # Evaluation
    # ======================================================

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predicted = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            actuals.extend(
                labels.cpu().tolist()
            )

    # ======================================================
    # Metrics
    # ======================================================

    accuracy = accuracy_score(
        actuals,
        predictions
    )

    precision = precision_score(
        actuals,
        predictions
    )

    recall = recall_score(
        actuals,
        predictions
    )

    f1 = f1_score(
        actuals,
        predictions
    )

    matrix = confusion_matrix(
        actuals,
        predictions
    )

    print("\nTest Results")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            actuals,
            predictions,
            target_names=[
                "Negative",
                "Positive"
            ]
        )
    )

    print("\nConfusion Matrix")

    print(matrix)


if __name__ == "__main__":
    main()